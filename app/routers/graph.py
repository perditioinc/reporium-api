"""
KAN-83 / KAN-124: Knowledge graph edges endpoint.

Returns repo-to-repo similarity edges computed on the fly from pgvector
embeddings (HNSW cosine similarity). Each repo's top-K nearest neighbours
become graph edges, giving full coverage across the entire library.

Previously read from a static `repo_edges` table populated by a naive
category-matching script.  The new approach uses the existing 384-dim
all-MiniLM-L6-v2 embeddings and the HNSW index for fast ANN queries.
"""

import hashlib
import logging
import math
import re

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.cache import cache as redis_cache
from app.database import get_db
from app.embeddings import get_embedding_model
from app.graph_snapshot import build_graph_payload_from_snapshot, load_graph_snapshot
from app.rate_limit import rate_limit_storage
from app.utils import vec_to_pg

logger = logging.getLogger(__name__)

CACHE_TTL_GRAPH_EDGES = 3600  # 1 hr
CACHE_TTL_GRAPH_SEARCH = 1800  # 30 min
CACHE_TTL_GRAPH_SUBGRAPH = 1800  # 30 min
CACHE_TTL_GRAPH_CLUSTERS = 3600  # 1 hr

_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
_EMBEDDING_DIMENSION = 384

router = APIRouter(tags=["Graph"])
_limiter = Limiter(key_func=get_remote_address, storage_uri=rate_limit_storage)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_since(since: str | None) -> str | None:
    """Parse a duration string like '7d', '24h', '30m' into a Postgres interval.

    Returns None if *since* is falsy or not a recognised pattern.
    """
    if not since:
        return None
    m = re.match(r"^(\d+)([dhm])$", since.strip().lower())
    if not m:
        return None
    value, unit = m.group(1), m.group(2)
    unit_map = {"d": "day", "h": "hour", "m": "minute"}
    return f"{value} {unit_map[unit]}"


def _log_scale_stars(stars: int | None) -> float:
    """Return a log-scaled star value for node sizing (0.0 when no stars)."""
    try:
        stars_value = int(stars or 0)
    except (TypeError, ValueError):
        return 0.0
    if stars_value <= 0:
        return 0.0
    return round(math.log10(stars_value + 1), 4)


def _extract_quality(quality_signals: dict | None) -> float | None:
    """Extract a 0-1 quality score from the quality_signals JSONB.

    Returns the ``overall`` key if present, else None.
    """
    if not isinstance(quality_signals, dict):
        return None
    overall = quality_signals.get("overall")
    if overall is not None:
        return round(float(overall), 4)
    return None


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _rows_to_edges(rows) -> list[dict]:
    """Convert DB rows (with source_*/target_* columns) to edge dicts.

    Handles both similarity rows (have a ``similarity`` column, no ``edge_type``)
    and typed rows from ``repo_edges`` (have ``edge_type`` and ``weight`` columns).
    """
    return [
        {
            "edgeType": (getattr(row, "edge_type", None) or "SIMILAR_TO"),
            "weight": round(float(
                getattr(row, "similarity", None)
                or getattr(row, "weight", None)
                or 0.5
            ), 4),
            "evidence": None,
            "source": {
                "name": row.source_name,
                "owner": row.source_owner,
                "description": row.source_description,
                "category": row.source_category,
            },
            "target": {
                "name": row.target_name,
                "owner": row.target_owner,
                "description": row.target_description,
                "category": row.target_category,
            },
        }
        for row in rows
    ]


def _rows_to_nodes(rows) -> list[dict]:
    """Build unique nodes list from edge rows that contain viz columns."""
    node_map: dict[str, dict] = {}
    for row in rows:
        for prefix in ("source", "target"):
            name = getattr(row, f"{prefix}_name")
            if name not in node_map:
                stars = getattr(row, f"{prefix}_stars", None)
                qs = getattr(row, f"{prefix}_quality_signals", None)
                node_map[name] = {
                    "name": name,
                    "owner": getattr(row, f"{prefix}_owner"),
                    "description": getattr(row, f"{prefix}_description"),
                    "primary_category": getattr(row, f"{prefix}_category"),
                    "stars": _safe_int(stars),
                    "stars_log": _log_scale_stars(stars),
                    "quality": _extract_quality(qs),
                }
    return list(node_map.values())


def _json_graph_response(payload: dict) -> JSONResponse:
    response = JSONResponse(content=payload)
    response.headers["Cache-Control"] = "public, max-age=3600"
    return response


async def _build_graph_payload_from_db(
    db: AsyncSession,
    *,
    limit: int,
    min_similarity: float,
    neighbours: int,
    interval: str | None,
) -> dict:
    """DEPRECATED - This function is dead code, never called. Kept for reference only."""
    # Build optional temporal WHERE clause
    since_clause = ""
    if interval:
        since_clause = (
            "AND (r1.updated_at >= NOW() - :since_interval::interval "
            "OR r2.updated_at >= NOW() - :since_interval::interval)"
        )

    # Use a CTE to find top-K neighbours per repo via HNSW index.
    # The <=> operator returns cosine distance; 1 - distance = similarity.
    # We lateral-join to get the K nearest neighbours per repo efficiently.
    sql = text(f"""
        WITH ranked AS (
            SELECT
                e1.repo_id   AS source_id,
                e2.repo_id   AS target_id,
                1 - (e1.embedding_vec <=> e2.embedding_vec) AS similarity
            FROM repo_embeddings e1
            CROSS JOIN LATERAL (
                SELECT e2_inner.repo_id,
                       e2_inner.embedding_vec
                FROM repo_embeddings e2_inner
                WHERE e2_inner.repo_id != e1.repo_id
                ORDER BY e1.embedding_vec <=> e2_inner.embedding_vec
                LIMIT :neighbours
            ) e2
            WHERE 1 - (e1.embedding_vec <=> e2.embedding_vec) >= :min_sim
        ),
        deduped AS (
            SELECT DISTINCT ON (LEAST(source_id, target_id), GREATEST(source_id, target_id))
                source_id, target_id, similarity
            FROM ranked
            ORDER BY LEAST(source_id, target_id), GREATEST(source_id, target_id),
                     similarity DESC
        ),
        -- Orphan rescue: repos with embeddings but no edges above threshold
        orphan_edges AS (
            SELECT DISTINCT ON (LEAST(e1.repo_id, e2.repo_id), GREATEST(e1.repo_id, e2.repo_id))
                e1.repo_id AS source_id,
                e2.repo_id AS target_id,
                1 - (e1.embedding_vec <=> e2.embedding_vec) AS similarity
            FROM repo_embeddings e1
            CROSS JOIN LATERAL (
                SELECT e2_inner.repo_id,
                       e2_inner.embedding_vec
                FROM repo_embeddings e2_inner
                WHERE e2_inner.repo_id != e1.repo_id
                ORDER BY e1.embedding_vec <=> e2_inner.embedding_vec
                LIMIT 1
            ) e2
            WHERE NOT EXISTS (
                SELECT 1 FROM deduped d
                WHERE d.source_id = e1.repo_id OR d.target_id = e1.repo_id
            )
            ORDER BY LEAST(e1.repo_id, e2.repo_id), GREATEST(e1.repo_id, e2.repo_id),
                     similarity DESC
        ),
        all_edges AS (
            SELECT source_id, target_id, similarity FROM deduped
            UNION
            SELECT source_id, target_id, similarity FROM orphan_edges
        )
        SELECT
            ae.similarity,
            r1.name               AS source_name,
            r1.description        AS source_description,
            r1.primary_category   AS source_category,
            r1.owner              AS source_owner,
            r1.stargazers_count   AS source_stars,
            r1.quality_signals    AS source_quality_signals,
            r1.updated_at         AS source_updated_at,
            r2.name               AS target_name,
            r2.description        AS target_description,
            r2.primary_category   AS target_category,
            r2.owner              AS target_owner,
            r2.stargazers_count   AS target_stars,
            r2.quality_signals    AS target_quality_signals,
            r2.updated_at         AS target_updated_at
        FROM all_edges ae
        JOIN repos r1 ON r1.id = ae.source_id AND r1.is_private = false
        JOIN repos r2 ON r2.id = ae.target_id AND r2.is_private = false
        WHERE 1=1 {since_clause}
        ORDER BY ae.similarity DESC
        LIMIT :limit
    """)

    params: dict = {
        "neighbours": neighbours,
        "min_sim": min_similarity,
        "limit": limit,
    }
    if interval:
        params["since_interval"] = interval

    result = await db.execute(sql, params)
    rows = result.fetchall()

    # Build similarity edges, keyed by (source, target) for dedup
    edges_by_pair: dict[tuple[str, str], dict] = {}
    node_map: dict[str, dict] = {}

    for row in rows:
        key = (row.source_name, row.target_name)
        if key not in edges_by_pair:
            edges_by_pair[key] = {
                "edgeType": "SIMILAR_TO",
                "weight": round(float(row.similarity), 4),
                "evidence": None,
                "source": {
                    "name": row.source_name, "owner": row.source_owner,
                    "description": row.source_description, "category": row.source_category,
                },
                "target": {
                    "name": row.target_name, "owner": row.target_owner,
                    "description": row.target_description, "category": row.target_category,
                },
            }
        if row.source_name not in node_map:
            node_map[row.source_name] = {
                "name": row.source_name, "owner": row.source_owner,
                "description": row.source_description,
                "primary_category": row.source_category,
                "stars": _safe_int(row.source_stars),
                "stars_log": _log_scale_stars(row.source_stars),
                "quality": _extract_quality(row.source_quality_signals),
            }
        if row.target_name not in node_map:
            node_map[row.target_name] = {
                "name": row.target_name, "owner": row.target_owner,
                "description": row.target_description,
                "primary_category": row.target_category,
                "stars": _safe_int(row.target_stars),
                "stars_log": _log_scale_stars(row.target_stars),
                "quality": _extract_quality(row.target_quality_signals),
            }

    # Merge typed relationship edges from repo_edges â€” override SIMILAR_TO when same pair
    try:
        typed_sql = text("""
            SELECT
                re.edge_type,
                re.weight,
                r1.name        AS source_name,
                r1.owner       AS source_owner,
                r1.description AS source_description,
                r1.primary_category AS source_category,
                r1.stargazers_count AS source_stars,
                r1.quality_signals  AS source_quality_signals,
                r2.name        AS target_name,
                r2.owner       AS target_owner,
                r2.description AS target_description,
                r2.primary_category AS target_category,
                r2.stargazers_count AS target_stars,
                r2.quality_signals  AS target_quality_signals
            FROM repo_edges re
            JOIN repos r1 ON r1.id = re.source_repo_id AND r1.is_private = false
            JOIN repos r2 ON r2.id = re.target_repo_id AND r2.is_private = false
            WHERE re.edge_type IN ('ALTERNATIVE_TO', 'COMPATIBLE_WITH', 'DEPENDS_ON', 'EXTENDS')
            ORDER BY re.weight DESC NULLS LAST
            LIMIT 5000
        """)
        typed_result = await db.execute(typed_sql)
        typed_rows = typed_result.fetchall()

        for trow in typed_rows:
            if not isinstance(getattr(trow, "edge_type", None), str):
                continue
            key = (trow.source_name, trow.target_name)
            # Typed edge overrides similarity edge for same pair
            edges_by_pair[key] = {
                "edgeType": trow.edge_type,
                "weight": round(float(trow.weight or 0.5), 4),
                "evidence": None,
                "source": {
                    "name": trow.source_name, "owner": trow.source_owner,
                    "description": trow.source_description, "category": trow.source_category,
                },
                "target": {
                    "name": trow.target_name, "owner": trow.target_owner,
                    "description": trow.target_description, "category": trow.target_category,
                },
            }
            if trow.source_name not in node_map:
                node_map[trow.source_name] = {
                    "name": trow.source_name, "owner": trow.source_owner,
                    "description": trow.source_description,
                    "primary_category": trow.source_category,
                    "stars": _safe_int(trow.source_stars),
                    "stars_log": _log_scale_stars(trow.source_stars),
                    "quality": _extract_quality(trow.source_quality_signals),
                }
            if trow.target_name not in node_map:
                node_map[trow.target_name] = {
                    "name": trow.target_name, "owner": trow.target_owner,
                    "description": trow.target_description,
                    "primary_category": trow.target_category,
                    "stars": _safe_int(trow.target_stars),
                    "stars_log": _log_scale_stars(trow.target_stars),
                    "quality": _extract_quality(trow.target_quality_signals),
                }
    except Exception as exc:
        # repo_edges table may not exist yet â€” degrade gracefully
        logger.warning("Could not fetch typed graph edges: %s", exc)

    edges = list(edges_by_pair.values())
    nodes = list(node_map.values())

    # Count repos with and without embeddings for diagnostics
    counts = await db.execute(text("""
        SELECT
            (SELECT COUNT(*) FROM repos WHERE is_private = false) AS total_public,
            (SELECT COUNT(DISTINCT re.repo_id)
             FROM repo_embeddings re
             JOIN repos r ON r.id = re.repo_id
             WHERE r.is_private = false
               AND re.embedding_vec IS NOT NULL) AS with_embeddings
    """))
    count_row = counts.fetchone()

    return {
        "total": len(edges),
        "total_repos": len(nodes),
        "total_public_repos": count_row.total_public if count_row else 0,
        "repos_with_embeddings": count_row.with_embeddings if count_row else 0,
        "edgeTypes": sorted({e["edgeType"] for e in edges}),
        "nodes": nodes,
        "edges": edges,
        "graph_source": "database",
    }


# ---------------------------------------------------------------------------
# GET /graph/edges — enhanced with nodes array + temporal filter
# ---------------------------------------------------------------------------

@router.get("/graph/edges")
@_limiter.limit("20/minute")
async def get_graph_edges(
    request: Request,
    limit: int = Query(default=500, ge=1, le=10000),
    min_similarity: float = Query(default=0.55, ge=0.0, le=1.0,
                                  description="Minimum cosine similarity threshold"),
    neighbours: int = Query(default=8, ge=1, le=30,
                            description="Max neighbours per repo"),
    since: str | None = Query(default=None,
                              description="Temporal filter, e.g. '7d', '24h', '30m'"),
    db: AsyncSession = Depends(get_db),
):
    """
    Returns knowledge graph edges based on pgvector embedding similarity.
    Each repo is connected to its top-K nearest neighbours above the
    similarity threshold.  Edges are SIMILAR_TO with weight = similarity.

    Includes a ``nodes`` array with cluster metadata for visualization
    (primary_category, log-scaled stars, quality score).

    Optional ``since`` param filters to edges where at least one node was
    added/updated within the given time window (e.g. ``?since=7d``).
    """
    interval = _parse_since(since)

    # --- Redis cache check ---
    cache_key = f"graph_edges:{limit}:{min_similarity}:{neighbours}:{since or 'all'}"
    cached = await redis_cache.get(cache_key)
    if cached is not None:
        return _json_graph_response(cached)

    snapshot = await load_graph_snapshot()
    if snapshot is not None:
        try:
            snapshot_payload = build_graph_payload_from_snapshot(
                snapshot,
                limit=limit,
                min_similarity=min_similarity,
                neighbours=neighbours,
                since_interval=interval,
            )
        except Exception as exc:
            logger.warning("Graph snapshot build failed; falling back to live DB: %s", exc)
        else:
            await redis_cache.set(cache_key, snapshot_payload, ttl=CACHE_TTL_GRAPH_EDGES)
            return _json_graph_response(snapshot_payload)

    # Build optional temporal WHERE clause
    since_clause = ""
    if interval:
        since_clause = (
            "AND (r1.updated_at >= NOW() - :since_interval::interval "
            "OR r2.updated_at >= NOW() - :since_interval::interval)"
        )

    # Use a CTE to find top-K neighbours per repo via HNSW index.
    # The <=> operator returns cosine distance; 1 - distance = similarity.
    # We lateral-join to get the K nearest neighbours per repo efficiently.
    sql = text(f"""
        WITH ranked AS (
            SELECT
                e1.repo_id   AS source_id,
                e2.repo_id   AS target_id,
                1 - (e1.embedding_vec <=> e2.embedding_vec) AS similarity
            FROM repo_embeddings e1
            CROSS JOIN LATERAL (
                SELECT e2_inner.repo_id,
                       e2_inner.embedding_vec
                FROM repo_embeddings e2_inner
                WHERE e2_inner.repo_id != e1.repo_id
                ORDER BY e1.embedding_vec <=> e2_inner.embedding_vec
                LIMIT :neighbours
            ) e2
            WHERE 1 - (e1.embedding_vec <=> e2.embedding_vec) >= :min_sim
        ),
        deduped AS (
            SELECT DISTINCT ON (LEAST(source_id, target_id), GREATEST(source_id, target_id))
                source_id, target_id, similarity
            FROM ranked
            ORDER BY LEAST(source_id, target_id), GREATEST(source_id, target_id),
                     similarity DESC
        ),
        -- Orphan rescue: repos with embeddings but no edges above threshold
        orphan_edges AS (
            SELECT DISTINCT ON (LEAST(e1.repo_id, e2.repo_id), GREATEST(e1.repo_id, e2.repo_id))
                e1.repo_id AS source_id,
                e2.repo_id AS target_id,
                1 - (e1.embedding_vec <=> e2.embedding_vec) AS similarity
            FROM repo_embeddings e1
            CROSS JOIN LATERAL (
                SELECT e2_inner.repo_id,
                       e2_inner.embedding_vec
                FROM repo_embeddings e2_inner
                WHERE e2_inner.repo_id != e1.repo_id
                ORDER BY e1.embedding_vec <=> e2_inner.embedding_vec
                LIMIT 1
            ) e2
            WHERE NOT EXISTS (
                SELECT 1 FROM deduped d
                WHERE d.source_id = e1.repo_id OR d.target_id = e1.repo_id
            )
            ORDER BY LEAST(e1.repo_id, e2.repo_id), GREATEST(e1.repo_id, e2.repo_id),
                     similarity DESC
        ),
        all_edges AS (
            SELECT source_id, target_id, similarity FROM deduped
            UNION
            SELECT source_id, target_id, similarity FROM orphan_edges
        )
        SELECT
            ae.similarity,
            r1.name               AS source_name,
            r1.description        AS source_description,
            r1.primary_category   AS source_category,
            r1.owner              AS source_owner,
            r1.stargazers_count   AS source_stars,
            r1.quality_signals    AS source_quality_signals,
            r1.updated_at         AS source_updated_at,
            r2.name               AS target_name,
            r2.description        AS target_description,
            r2.primary_category   AS target_category,
            r2.owner              AS target_owner,
            r2.stargazers_count   AS target_stars,
            r2.quality_signals    AS target_quality_signals,
            r2.updated_at         AS target_updated_at
        FROM all_edges ae
        JOIN repos r1 ON r1.id = ae.source_id AND r1.is_private = false
        JOIN repos r2 ON r2.id = ae.target_id AND r2.is_private = false
        WHERE 1=1 {since_clause}
        ORDER BY ae.similarity DESC
        LIMIT :limit
    """)

    params: dict = {
        "neighbours": neighbours,
        "min_sim": min_similarity,
        "limit": limit,
    }
    if interval:
        params["since_interval"] = interval

    result = await db.execute(sql, params)
    rows = result.fetchall()

    # Build similarity edges, keyed by (source, target) for dedup
    edges_by_pair: dict[tuple[str, str], dict] = {}
    node_map: dict[str, dict] = {}

    for row in rows:
        key = (row.source_name, row.target_name)
        if key not in edges_by_pair:
            edges_by_pair[key] = {
                "edgeType": "SIMILAR_TO",
                "weight": round(float(row.similarity), 4),
                "evidence": None,
                "source": {
                    "name": row.source_name, "owner": row.source_owner,
                    "description": row.source_description, "category": row.source_category,
                },
                "target": {
                    "name": row.target_name, "owner": row.target_owner,
                    "description": row.target_description, "category": row.target_category,
                },
            }
        if row.source_name not in node_map:
            node_map[row.source_name] = {
                "name": row.source_name, "owner": row.source_owner,
                "description": row.source_description,
                "primary_category": row.source_category,
                "stars": _safe_int(row.source_stars),
                "stars_log": _log_scale_stars(row.source_stars),
                "quality": _extract_quality(row.source_quality_signals),
            }
        if row.target_name not in node_map:
            node_map[row.target_name] = {
                "name": row.target_name, "owner": row.target_owner,
                "description": row.target_description,
                "primary_category": row.target_category,
                "stars": _safe_int(row.target_stars),
                "stars_log": _log_scale_stars(row.target_stars),
                "quality": _extract_quality(row.target_quality_signals),
            }

    # Merge typed relationship edges from repo_edges — override SIMILAR_TO when same pair.
    # Use per-type ranked window to prevent any single type from starving the others
    # (ALTERNATIVE_TO has 46k rows all at weight=1.0 and would fill a flat LIMIT).
    try:
        typed_sql = text("""
            WITH ranked AS (
                SELECT
                    re.edge_type,
                    re.weight,
                    r1.name        AS source_name,
                    r1.owner       AS source_owner,
                    r1.description AS source_description,
                    r1.primary_category AS source_category,
                    r1.stargazers_count AS source_stars,
                    r1.quality_signals  AS source_quality_signals,
                    r2.name        AS target_name,
                    r2.owner       AS target_owner,
                    r2.description AS target_description,
                    r2.primary_category AS target_category,
                    r2.stargazers_count AS target_stars,
                    r2.quality_signals  AS target_quality_signals,
                    ROW_NUMBER() OVER (
                        PARTITION BY re.edge_type
                        ORDER BY re.weight DESC NULLS LAST
                    ) AS rn
                FROM repo_edges re
                JOIN repos r1 ON r1.id = re.source_repo_id AND r1.is_private = false
                JOIN repos r2 ON r2.id = re.target_repo_id AND r2.is_private = false
                WHERE re.edge_type IN ('ALTERNATIVE_TO', 'COMPATIBLE_WITH', 'DEPENDS_ON', 'EXTENDS')
            )
            SELECT
                edge_type, weight,
                source_name, source_owner, source_description, source_category,
                source_stars, source_quality_signals,
                target_name, target_owner, target_description, target_category,
                target_stars, target_quality_signals
            FROM ranked
            WHERE rn <= 2000
            ORDER BY weight DESC NULLS LAST
        """)
        typed_result = await db.execute(typed_sql)
        typed_rows = typed_result.fetchall()

        for trow in typed_rows:
            if not isinstance(getattr(trow, "edge_type", None), str):
                continue
            key = (trow.source_name, trow.target_name)
            # Typed edge overrides similarity edge for same pair
            edges_by_pair[key] = {
                "edgeType": trow.edge_type,
                "weight": round(float(trow.weight or 0.5), 4),
                "evidence": None,
                "source": {
                    "name": trow.source_name, "owner": trow.source_owner,
                    "description": trow.source_description, "category": trow.source_category,
                },
                "target": {
                    "name": trow.target_name, "owner": trow.target_owner,
                    "description": trow.target_description, "category": trow.target_category,
                },
            }
            if trow.source_name not in node_map:
                node_map[trow.source_name] = {
                    "name": trow.source_name, "owner": trow.source_owner,
                    "description": trow.source_description,
                    "primary_category": trow.source_category,
                    "stars": _safe_int(trow.source_stars),
                    "stars_log": _log_scale_stars(trow.source_stars),
                    "quality": _extract_quality(trow.source_quality_signals),
                }
            if trow.target_name not in node_map:
                node_map[trow.target_name] = {
                    "name": trow.target_name, "owner": trow.target_owner,
                    "description": trow.target_description,
                    "primary_category": trow.target_category,
                    "stars": _safe_int(trow.target_stars),
                    "stars_log": _log_scale_stars(trow.target_stars),
                    "quality": _extract_quality(trow.target_quality_signals),
                }
    except Exception as exc:
        # repo_edges table may not exist yet — degrade gracefully
        logger.warning("Could not fetch typed graph edges: %s", exc)

    edges = list(edges_by_pair.values())
    nodes = list(node_map.values())

    # Count repos with and without embeddings for diagnostics
    counts = await db.execute(text("""
        SELECT
            (SELECT COUNT(*) FROM repos WHERE is_private = false) AS total_public,
            (SELECT COUNT(DISTINCT re.repo_id)
             FROM repo_embeddings re
             JOIN repos r ON r.id = re.repo_id
             WHERE r.is_private = false
               AND re.embedding_vec IS NOT NULL) AS with_embeddings,
            (SELECT COUNT(*) FROM repo_edges) AS total_graph_edges
    """))
    count_row = counts.fetchone()

    result_payload = {
        "total": len(edges),
        "total_repos": len(nodes),
        "total_public_repos": count_row.total_public if count_row else 0,
        "repos_with_embeddings": count_row.with_embeddings if count_row else 0,
        "total_knowledge_graph_edges": count_row.total_graph_edges if count_row else 0,
        "edgeTypes": sorted({e["edgeType"] for e in edges}),
        "nodes": nodes,
        "edges": edges,
    }

    # Store in Redis cache
    await redis_cache.set(cache_key, result_payload, ttl=CACHE_TTL_GRAPH_EDGES)

    return _json_graph_response(result_payload)


# ---------------------------------------------------------------------------
# GET /graph/edges/search — semantic graph search
# ---------------------------------------------------------------------------

@router.get("/graph/edges/search")
@_limiter.limit("10/minute")
async def search_graph_edges(
    request: Request,
    query: str = Query(..., min_length=1, description="Natural-language search query"),
    top_k: int = Query(default=10, ge=1, le=50,
                       description="Number of most-similar seed repos to find"),
    neighbours: int = Query(default=3, ge=1, le=20,
                            description="Neighbours to expand per seed repo"),
    min_similarity: float = Query(default=0.5, ge=0.0, le=1.0,
                                  description="Minimum cosine similarity threshold"),
    db: AsyncSession = Depends(get_db),
):
    """
    Semantic graph search: embed a free-text query, find the top-K most
    similar repos via pgvector, then expand each seed's nearest neighbours
    to build a subgraph of edges.  Results are cached in Redis (TTL 30 min).
    """
    # --- Redis cache ---
    query_hash = hashlib.sha256(query.lower().strip().encode()).hexdigest()[:16]
    cache_key = f"graph_search:{query_hash}:{top_k}:{neighbours}:{min_similarity}"
    cached = await redis_cache.get(cache_key)
    if cached is not None:
        return cached

    # 1. Embed the query
    model = get_embedding_model()
    query_vec = model.encode(query)
    vec_str = vec_to_pg(query_vec)

    # 2. Find top_k most similar repos to the query embedding
    # 3. For each seed, find its `neighbours` nearest neighbours
    # 4. Build edges between seeds and their neighbours, plus inter-neighbour
    sql = text("""
        WITH seeds AS (
            SELECT re.repo_id,
                   1 - (re.embedding_vec <=> CAST(:vec AS vector)) AS query_sim
            FROM repo_embeddings re
            JOIN repos r ON r.id = re.repo_id
            WHERE r.is_private = false
              AND re.embedding_vec IS NOT NULL
            ORDER BY re.embedding_vec <=> CAST(:vec AS vector)
            LIMIT :top_k
        ),
        -- Expand each seed's neighbourhood
        expanded AS (
            SELECT
                s.repo_id      AS source_id,
                e2.repo_id     AS target_id,
                1 - (e1.embedding_vec <=> e2.embedding_vec) AS similarity
            FROM seeds s
            JOIN repo_embeddings e1 ON e1.repo_id = s.repo_id
            CROSS JOIN LATERAL (
                SELECT e2_inner.repo_id,
                       e2_inner.embedding_vec
                FROM repo_embeddings e2_inner
                WHERE e2_inner.repo_id != s.repo_id
                ORDER BY e1.embedding_vec <=> e2_inner.embedding_vec
                LIMIT :neighbours
            ) e2
            WHERE 1 - (e1.embedding_vec <=> e2.embedding_vec) >= :min_sim
        ),
        deduped AS (
            SELECT DISTINCT ON (LEAST(source_id, target_id), GREATEST(source_id, target_id))
                source_id, target_id, similarity
            FROM expanded
            ORDER BY LEAST(source_id, target_id), GREATEST(source_id, target_id),
                     similarity DESC
        )
        SELECT
            d.similarity,
            r1.name        AS source_name,
            r1.description AS source_description,
            r1.primary_category AS source_category,
            r1.owner       AS source_owner,
            r2.name        AS target_name,
            r2.description AS target_description,
            r2.primary_category AS target_category,
            r2.owner       AS target_owner
        FROM deduped d
        JOIN repos r1 ON r1.id = d.source_id AND r1.is_private = false
        JOIN repos r2 ON r2.id = d.target_id AND r2.is_private = false
        ORDER BY d.similarity DESC
    """)

    result = await db.execute(sql, {
        "vec": vec_str,
        "top_k": top_k,
        "neighbours": neighbours,
        "min_sim": min_similarity,
    })
    rows = result.fetchall()
    edges = _rows_to_edges(rows)

    # Count distinct repos in the subgraph
    repo_names: set[str] = set()
    for row in rows:
        repo_names.add(row.source_name)
        repo_names.add(row.target_name)

    payload = {
        "query": query,
        "total": len(edges),
        "total_repos": len(repo_names),
        "edgeTypes": ["SIMILAR_TO"],
        "edges": edges,
    }

    await redis_cache.set(cache_key, payload, ttl=CACHE_TTL_GRAPH_SEARCH)
    return payload


# ---------------------------------------------------------------------------
# GET /graph/subgraph/{repo_name} — 2-hop neighbourhood of a specific repo
# ---------------------------------------------------------------------------

@router.get("/graph/subgraph/{repo_name}")
@_limiter.limit("15/minute")
async def get_repo_subgraph(
    request: Request,
    repo_name: str,
    neighbours: int = Query(default=8, ge=1, le=30,
                            description="Max neighbours per hop"),
    min_similarity: float = Query(default=0.5, ge=0.0, le=1.0,
                                  description="Minimum cosine similarity threshold"),
    db: AsyncSession = Depends(get_db),
):
    """
    Returns the 2-hop neighbourhood of a specific repo for focused
    exploration (e.g. "show me repos related to LangChain").

    Hop 1: direct embedding neighbours of the target repo.
    Hop 2: neighbours of each hop-1 repo.

    Redis cached with 30-min TTL.
    """
    cache_key = f"graph_subgraph:{repo_name}:{neighbours}:{min_similarity}"
    cached = await redis_cache.get(cache_key)
    if cached is not None:
        return cached

    # Verify the repo exists and has an embedding
    check = await db.execute(text("""
        SELECT r.id, r.name, r.owner, r.description, r.primary_category,
               r.stargazers_count, r.quality_signals
        FROM repos r
        JOIN repo_embeddings re ON re.repo_id = r.id
        WHERE r.name = :name AND r.is_private = false
          AND re.embedding_vec IS NOT NULL
        LIMIT 1
    """), {"name": repo_name})
    seed_row = check.fetchone()

    if seed_row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Repo '{repo_name}' not found or has no embedding",
        )

    # 2-hop neighbourhood via CTEs
    sql = text("""
        WITH seed AS (
            SELECT re.repo_id, re.embedding_vec
            FROM repo_embeddings re
            JOIN repos r ON r.id = re.repo_id
            WHERE r.name = :name AND r.is_private = false
            LIMIT 1
        ),
        -- Hop 1: direct neighbours of seed
        hop1 AS (
            SELECT
                s.repo_id      AS source_id,
                e2.repo_id     AS target_id,
                1 - (s.embedding_vec <=> e2.embedding_vec) AS similarity
            FROM seed s
            CROSS JOIN LATERAL (
                SELECT e2_inner.repo_id, e2_inner.embedding_vec
                FROM repo_embeddings e2_inner
                WHERE e2_inner.repo_id != s.repo_id
                ORDER BY s.embedding_vec <=> e2_inner.embedding_vec
                LIMIT :neighbours
            ) e2
            WHERE 1 - (s.embedding_vec <=> e2.embedding_vec) >= :min_sim
        ),
        -- Hop 2: neighbours of hop-1 repos
        hop2 AS (
            SELECT
                h1.target_id   AS source_id,
                e3.repo_id     AS target_id,
                1 - (e1.embedding_vec <=> e3.embedding_vec) AS similarity
            FROM hop1 h1
            JOIN repo_embeddings e1 ON e1.repo_id = h1.target_id
            CROSS JOIN LATERAL (
                SELECT e3_inner.repo_id, e3_inner.embedding_vec
                FROM repo_embeddings e3_inner
                WHERE e3_inner.repo_id != h1.target_id
                  AND e3_inner.repo_id != (SELECT repo_id FROM seed)
                ORDER BY e1.embedding_vec <=> e3_inner.embedding_vec
                LIMIT :neighbours
            ) e3
            WHERE 1 - (e1.embedding_vec <=> e3.embedding_vec) >= :min_sim
        ),
        all_edges AS (
            SELECT source_id, target_id, similarity FROM hop1
            UNION
            SELECT source_id, target_id, similarity FROM hop2
        ),
        deduped AS (
            SELECT DISTINCT ON (LEAST(source_id, target_id), GREATEST(source_id, target_id))
                source_id, target_id, similarity
            FROM all_edges
            ORDER BY LEAST(source_id, target_id), GREATEST(source_id, target_id),
                     similarity DESC
        )
        SELECT
            d.similarity,
            r1.name               AS source_name,
            r1.description        AS source_description,
            r1.primary_category   AS source_category,
            r1.owner              AS source_owner,
            r1.stargazers_count   AS source_stars,
            r1.quality_signals    AS source_quality_signals,
            r2.name               AS target_name,
            r2.description        AS target_description,
            r2.primary_category   AS target_category,
            r2.owner              AS target_owner,
            r2.stargazers_count   AS target_stars,
            r2.quality_signals    AS target_quality_signals
        FROM deduped d
        JOIN repos r1 ON r1.id = d.source_id AND r1.is_private = false
        JOIN repos r2 ON r2.id = d.target_id AND r2.is_private = false
        ORDER BY d.similarity DESC
    """)

    result = await db.execute(sql, {
        "name": repo_name,
        "neighbours": neighbours,
        "min_sim": min_similarity,
    })
    rows = result.fetchall()
    edges = _rows_to_edges(rows)
    nodes = _rows_to_nodes(rows)

    # Ensure the seed node is in the nodes list
    seed_in_nodes = any(n["name"] == repo_name for n in nodes)
    if not seed_in_nodes:
        nodes.insert(0, {
            "name": seed_row.name,
            "owner": seed_row.owner,
            "description": seed_row.description,
            "primary_category": seed_row.primary_category,
            "stars": seed_row.stargazers_count or 0,
            "stars_log": _log_scale_stars(seed_row.stargazers_count),
            "quality": _extract_quality(seed_row.quality_signals),
        })

    payload = {
        "repo_name": repo_name,
        "total_edges": len(edges),
        "total_nodes": len(nodes),
        "edgeTypes": ["SIMILAR_TO"],
        "nodes": nodes,
        "edges": edges,
    }

    await redis_cache.set(cache_key, payload, ttl=CACHE_TTL_GRAPH_SUBGRAPH)
    return payload


# ---------------------------------------------------------------------------
# GET /graph/clusters — repos grouped by primary_category
# ---------------------------------------------------------------------------

@router.get("/graph/clusters")
@_limiter.limit("10/minute")
async def get_graph_clusters(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Returns repos grouped by primary_category with cluster-level stats:
    category name, repo count, average stars, top 5 repos by stars, and
    inter-cluster edge counts.

    Redis cached with 1-hr TTL.
    """
    cache_key = "graph_clusters"
    cached = await redis_cache.get(cache_key)
    if cached is not None:
        return cached

    # 1. Cluster stats + top repos per category
    stats_sql = text("""
        WITH cat_stats AS (
            SELECT
                primary_category,
                COUNT(*)                        AS repo_count,
                ROUND(AVG(COALESCE(stargazers_count, 0))::numeric, 1) AS avg_stars
            FROM repos
            WHERE is_private = false AND primary_category IS NOT NULL
            GROUP BY primary_category
        ),
        top_repos AS (
            SELECT
                r.primary_category,
                r.name,
                r.owner,
                COALESCE(r.stargazers_count, 0) AS stars,
                ROW_NUMBER() OVER (
                    PARTITION BY r.primary_category
                    ORDER BY COALESCE(r.stargazers_count, 0) DESC
                ) AS rn
            FROM repos r
            WHERE r.is_private = false AND r.primary_category IS NOT NULL
        )
        SELECT
            cs.primary_category,
            cs.repo_count,
            cs.avg_stars,
            tr.name     AS repo_name,
            tr.owner    AS repo_owner,
            tr.stars    AS repo_stars
        FROM cat_stats cs
        LEFT JOIN top_repos tr
            ON tr.primary_category = cs.primary_category AND tr.rn <= 5
        ORDER BY cs.repo_count DESC, cs.primary_category, tr.rn
    """)
    stats_result = await db.execute(stats_sql)
    stats_rows = stats_result.fetchall()

    # Build cluster map
    clusters: dict[str, dict] = {}
    for row in stats_rows:
        cat = row.primary_category
        if cat not in clusters:
            clusters[cat] = {
                "category": cat,
                "repo_count": row.repo_count,
                "avg_stars": float(row.avg_stars) if row.avg_stars else 0.0,
                "top_repos": [],
                "inter_cluster_edges": {},
            }
        if row.repo_name:
            clusters[cat]["top_repos"].append({
                "name": row.repo_name,
                "owner": row.repo_owner,
                "stars": row.repo_stars,
            })

    # 2. Inter-cluster edge counts (how connected categories are)
    # Uses a sampled approach: top-3 neighbours per repo, count cross-category edges
    inter_sql = text("""
        WITH edges AS (
            SELECT
                r1.primary_category AS cat1,
                r2.primary_category AS cat2
            FROM repo_embeddings e1
            CROSS JOIN LATERAL (
                SELECT e2_inner.repo_id
                FROM repo_embeddings e2_inner
                WHERE e2_inner.repo_id != e1.repo_id
                ORDER BY e1.embedding_vec <=> e2_inner.embedding_vec
                LIMIT 3
            ) e2
            JOIN repos r1 ON r1.id = e1.repo_id AND r1.is_private = false
            JOIN repos r2 ON r2.id = e2.repo_id AND r2.is_private = false
            WHERE r1.primary_category IS NOT NULL
              AND r2.primary_category IS NOT NULL
              AND r1.primary_category != r2.primary_category
        )
        SELECT cat1, cat2, COUNT(*) AS edge_count
        FROM edges
        GROUP BY cat1, cat2
        ORDER BY edge_count DESC
    """)
    inter_result = await db.execute(inter_sql)
    inter_rows = inter_result.fetchall()

    for row in inter_rows:
        if row.cat1 in clusters:
            clusters[row.cat1]["inter_cluster_edges"][row.cat2] = row.edge_count
        if row.cat2 in clusters:
            clusters[row.cat2]["inter_cluster_edges"][row.cat1] = row.edge_count

    cluster_list = list(clusters.values())

    payload = {
        "total_clusters": len(cluster_list),
        "clusters": cluster_list,
    }

    await redis_cache.set(cache_key, payload, ttl=CACHE_TTL_GRAPH_CLUSTERS)
    return payload


# ---------------------------------------------------------------------------
# GET /metrics/embeddings — embedding coverage diagnostics
# ---------------------------------------------------------------------------

@router.get("/metrics/embeddings")
@_limiter.limit("30/minute")
async def get_embedding_metrics(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Returns embedding coverage metrics: how many public repos have
    embeddings vs total, the model name, and vector dimension.
    """
    counts = await db.execute(text("""
        SELECT
            (SELECT COUNT(*) FROM repos WHERE is_private = false) AS total_public,
            (SELECT COUNT(DISTINCT re.repo_id)
             FROM repo_embeddings re
             JOIN repos r ON r.id = re.repo_id
             WHERE r.is_private = false
               AND re.embedding_vec IS NOT NULL) AS with_embeddings
    """))
    row = counts.fetchone()
    total = row.total_public if row else 0
    with_emb = row.with_embeddings if row else 0
    coverage = round((with_emb / total * 100) if total > 0 else 0.0, 2)

    return {
        "total_public_repos": total,
        "repos_with_embeddings": with_emb,
        "coverage_percent": coverage,
        "model": _EMBEDDING_MODEL_NAME,
        "dimension": _EMBEDDING_DIMENSION,
    }
