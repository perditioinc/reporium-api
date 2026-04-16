from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

GRAPH_SNAPSHOT_VERSION = 1
_snapshot_cache: dict[str, Any] | None = None
_snapshot_cache_loaded_at = 0.0
_snapshot_lock = asyncio.Lock()


def _parse_snapshot_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        logger.warning("Could not parse graph snapshot datetime: %s", value)
        return None


def _read_snapshot_from_disk(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_snapshot_from_gcs(bucket_name: str, object_name: str) -> dict[str, Any]:
    from google.cloud import storage

    client = storage.Client(project=settings.gcp_project)
    blob = client.bucket(bucket_name).blob(object_name)
    return json.loads(blob.download_as_text())


def _read_snapshot_payload() -> dict[str, Any] | None:
    if settings.graph_snapshot_local_path:
        path = Path(settings.graph_snapshot_local_path)
        if path.exists():
            return _read_snapshot_from_disk(str(path))
        logger.warning("Graph snapshot local path does not exist: %s", path)

    if settings.graph_snapshot_bucket:
        return _read_snapshot_from_gcs(
            settings.graph_snapshot_bucket,
            settings.graph_snapshot_object,
        )

    return None


async def load_graph_snapshot(force_refresh: bool = False) -> dict[str, Any] | None:
    global _snapshot_cache, _snapshot_cache_loaded_at

    ttl_seconds = max(1, settings.graph_snapshot_cache_ttl_seconds)
    now = time.monotonic()
    if (
        not force_refresh
        and _snapshot_cache is not None
        and (now - _snapshot_cache_loaded_at) < ttl_seconds
    ):
        return _snapshot_cache

    async with _snapshot_lock:
        now = time.monotonic()
        if (
            not force_refresh
            and _snapshot_cache is not None
            and (now - _snapshot_cache_loaded_at) < ttl_seconds
        ):
            return _snapshot_cache

        try:
            snapshot = await asyncio.to_thread(_read_snapshot_payload)
        except Exception as exc:
            if _snapshot_cache is not None:
                logger.warning(
                    "Graph snapshot refresh failed; serving stale in-memory snapshot: %s",
                    exc,
                )
                return _snapshot_cache
            logger.warning("Graph snapshot refresh failed with no stale cache: %s", exc)
            return None

        if snapshot is None:
            return _snapshot_cache

        if snapshot.get("snapshot_version") != GRAPH_SNAPSHOT_VERSION:
            logger.warning(
                "Ignoring graph snapshot with unsupported version %s",
                snapshot.get("snapshot_version"),
            )
            return _snapshot_cache

        _snapshot_cache = snapshot
        _snapshot_cache_loaded_at = time.monotonic()
        return snapshot


def _node_to_response(node: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": node.get("name"),
        "owner": node.get("owner"),
        "description": node.get("description"),
        "category": node.get("primary_category"),
    }


def _node_to_viz(node: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": node.get("name"),
        "owner": node.get("owner"),
        "description": node.get("description"),
        "primary_category": node.get("primary_category"),
        "stars": int(node.get("stars") or 0),
        "stars_log": float(node.get("stars_log") or 0.0),
        "quality": node.get("quality"),
    }


def build_graph_payload_from_snapshot(
    snapshot: dict[str, Any],
    *,
    limit: int,
    min_similarity: float,
    neighbours: int,
    since_interval: str | None = None,
) -> dict[str, Any]:
    node_index = {
        str(node["repo_id"]): node
        for node in snapshot.get("nodes", [])
        if node.get("repo_id")
    }
    if not node_index:
        raise ValueError("Graph snapshot is missing nodes")

    cutoff: datetime | None = None
    if since_interval:
        value_str, unit = since_interval.split(" ", 1)
        value = int(value_str)
        seconds_map = {
            "day": 86400,
            "hour": 3600,
            "minute": 60,
        }
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=value * seconds_map[unit])

    def include_pair(source_id: str, target_id: str) -> bool:
        if cutoff is None:
            return True
        source_dt = _parse_snapshot_datetime(node_index[source_id].get("updated_at"))
        target_dt = _parse_snapshot_datetime(node_index[target_id].get("updated_at"))
        return (source_dt and source_dt >= cutoff) or (target_dt and target_dt >= cutoff)

    edges_by_pair: dict[tuple[str, str], dict[str, Any]] = {}

    for edge in snapshot.get("similarity_edges", []):
        source_id = str(edge.get("source_repo_id") or "")
        target_id = str(edge.get("target_repo_id") or "")
        if not source_id or not target_id:
            continue
        if source_id not in node_index or target_id not in node_index:
            continue
        if int(edge.get("rank") or 999999) > neighbours:
            continue
        weight = float(edge.get("weight") or 0.0)
        if weight < min_similarity:
            continue
        if not include_pair(source_id, target_id):
            continue

        pair_key = tuple(sorted((source_id, target_id)))
        existing = edges_by_pair.get(pair_key)
        if existing and float(existing["weight"]) >= weight:
            continue

        edges_by_pair[pair_key] = {
            "edgeType": "SIMILAR_TO",
            "weight": round(weight, 4),
            "evidence": None,
            "source": _node_to_response(node_index[source_id]),
            "target": _node_to_response(node_index[target_id]),
            "_node_ids": (source_id, target_id),
        }

    for edge in snapshot.get("typed_edges", []):
        source_id = str(edge.get("source_repo_id") or "")
        target_id = str(edge.get("target_repo_id") or "")
        if not source_id or not target_id:
            continue
        if source_id not in node_index or target_id not in node_index:
            continue
        if not include_pair(source_id, target_id):
            continue

        pair_key = tuple(sorted((source_id, target_id)))
        edges_by_pair[pair_key] = {
            "edgeType": edge.get("edge_type", "RELATED_TO"),
            "weight": round(float(edge.get("weight") or 0.5), 4),
            "evidence": None,
            "source": _node_to_response(node_index[source_id]),
            "target": _node_to_response(node_index[target_id]),
            "_node_ids": (source_id, target_id),
        }

    # Apply per-type cap before final limit so ALTERNATIVE_TO (46k edges @ weight=1.0)
    # cannot crowd out DEPENDS_ON (~111 edges) and COMPATIBLE_WITH.
    # Each typed type gets up to 40% of limit; SIMILAR_TO fills the rest.
    typed_cap = max(200, limit // 5)
    all_edges_sorted = sorted(
        edges_by_pair.values(),
        key=lambda edge: (float(edge["weight"]), edge["edgeType"] != "SIMILAR_TO"),
        reverse=True,
    )
    per_type_counts: dict[str, int] = {}
    balanced: list[dict] = []
    deferred_similar: list[dict] = []
    for edge in all_edges_sorted:
        et = edge["edgeType"]
        if et == "SIMILAR_TO":
            deferred_similar.append(edge)
        else:
            if per_type_counts.get(et, 0) < typed_cap:
                balanced.append(edge)
                per_type_counts[et] = per_type_counts.get(et, 0) + 1
    remaining = limit - len(balanced)
    balanced.extend(deferred_similar[:max(0, remaining)])
    edges = balanced[:limit]

    # Drop internal tracking field; we include ALL snapshot nodes below so
    # isolated repos (no edges above threshold) still render in the graph.
    for edge in edges:
        edge.pop("_node_ids", None)

    # KAN-XX: include every public repo in the response, not just those with
    # edges. Previously isolated repos were filtered out, reducing the
    # rendered graph from 1,641 → ~1,307 nodes. Force-layout on the frontend
    # places edgeless nodes at the periphery.
    included_nodes = [_node_to_viz(node) for node in node_index.values()]

    stats = snapshot.get("stats", {})
    return {
        "total": len(edges),
        "total_repos": len(included_nodes),
        "total_public_repos": int(stats.get("total_public_repos") or 0),
        "repos_with_embeddings": int(stats.get("repos_with_embeddings") or 0),
        "edgeTypes": sorted({edge["edgeType"] for edge in edges}),
        "nodes": included_nodes,
        "edges": edges,
        "graph_source": "snapshot",
        "snapshot_generated_at": snapshot.get("generated_at"),
    }
