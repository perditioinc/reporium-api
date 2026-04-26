"""Platform-level endpoints consumed by sibling repos and dashboards."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, Request, Response
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy import distinct, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import require_ingest_key, require_metrics_access, verify_api_key
from app.config import settings
from app.database import get_db
from app.models.repo import Repo, RepoAIDevSkill, RepoCategory
from app.prometheus_metrics import render_latest_metrics
from app.rate_limit import rate_limit_storage
from app.slo_observer import slo_observer, token_observer

# Shared limiter - matches the pattern used in intelligence.py / nl_filter.py.
_limiter = Limiter(key_func=get_remote_address, storage_uri=rate_limit_storage)

# Latency targets documented in docs/SLOs.md. The /metrics/slo and
# /metrics/latency endpoints use the same targets so the JSON view stays
# consistent with the Prometheus/Grafana view.
_LATENCY_TARGETS: dict[str, dict] = {
    "/health": {"p95_ms": 500, "p99_ms": 1000, "max_error_rate": 0.001},
    "/stats": {"p95_ms": 200, "p99_ms": 500, "max_error_rate": 0.01},
    "/library": {"p95_ms": 750, "p99_ms": 1500, "max_error_rate": 0.01},
    "/library/full": {"p95_ms": 2000, "p99_ms": 4000, "max_error_rate": 0.01},
    "/graph/edges": {"p95_ms": 200, "p99_ms": 500, "max_error_rate": 0.01},
    "/graph/edges/search": {"p95_ms": 1500, "p99_ms": 3000, "max_error_rate": 0.01},
    "/intelligence/ask": {"p95_ms": 15000, "p99_ms": 25000, "max_error_rate": 0.01},
    "/intelligence/nl-filter": {"p95_ms": 3000, "p99_ms": 5000, "max_error_rate": 0.01},
}

router = APIRouter(tags=["Platform"])


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)


def _normalize_repo_name(value: str | None) -> str:
    return (value or "").lower().replace("-", "").replace("_", "").strip()


def _build_latency_routes(snapshot: dict[str, dict], targets: dict[str, dict]) -> dict[str, dict]:
    routes: dict[str, dict] = {}
    for route, target in targets.items():
        observed = snapshot.get(route, {})
        p95 = observed.get("p95_ms")
        p99 = observed.get("p99_ms")
        err = observed.get("error_rate")

        breaches: list[str] = []
        if p95 is not None and "p95_ms" in target and p95 > target["p95_ms"]:
            breaches.append(f"p95 {p95}ms > target {target['p95_ms']}ms")
        if p99 is not None and "p99_ms" in target and p99 > target["p99_ms"]:
            breaches.append(f"p99 {p99}ms > target {target['p99_ms']}ms")
        if err is not None and err > target["max_error_rate"]:
            breaches.append(f"error_rate {err} > target {target['max_error_rate']}")

        routes[route] = {
            "target": target,
            "observed": observed,
            "status": "breach" if breaches else ("ok" if observed.get("count") else "no_data"),
            "breaches": breaches,
        }
    return routes


def _latency_payload() -> dict[str, Any]:
    return {
        "window_seconds": 24 * 60 * 60,
        "source": "in_memory_histogram",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "routes": _build_latency_routes(slo_observer.snapshot(), _LATENCY_TARGETS),
    }


def _spend_status(total_usd: float, budget_usd: float) -> str:
    """
    Map total spend against the soft daily budget:
      < 80%   -> ok
      80-100% -> warning
      >= 100% -> breach
    """
    if budget_usd <= 0:
        return "ok"
    ratio = total_usd / budget_usd
    if ratio >= 1.0:
        return "breach"
    if ratio >= 0.8:
        return "warning"
    return "ok"


async def _table_exists(db: AsyncSession, table_name: str) -> bool:
    result = await db.execute(
        text("SELECT to_regclass(:table_name) IS NOT NULL AS table_exists"),
        {"table_name": table_name},
    )
    row = result.fetchone()
    return bool(row.table_exists) if row is not None else False


async def _column_exists(db: AsyncSession, table_name: str, column_name: str) -> bool:
    result = await db.execute(
        text(
            """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_name = :table_name
                  AND column_name = :column_name
            ) AS column_exists
            """
        ),
        {"table_name": table_name, "column_name": column_name},
    )
    row = result.fetchone()
    return bool(row.column_exists) if row is not None else False


async def _backfill_snapshot(db: AsyncSession) -> dict[str, Any]:
    if not await _table_exists(db, "repo_dependencies"):
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": "postgres_live",
            "available": False,
            "reason": "repo_dependencies table not found",
        }

    result = await db.execute(
        text(
            """
            WITH per_repo AS (
                SELECT
                    repo_id,
                    COUNT(*) FILTER (
                        WHERE package_name != '__none__' AND is_direct = true
                    ) AS dep_count,
                    BOOL_OR(package_name = '__none__' OR is_direct = false) AS marked_no_deps,
                    MAX(fetched_at) AS last_fetched_at
                FROM repo_dependencies
                GROUP BY repo_id
            )
            SELECT
                (SELECT COUNT(*) FROM repos WHERE is_private = false) AS total_public_repos,
                COALESCE((SELECT COUNT(*) FROM per_repo), 0) AS repos_scanned,
                COALESCE((SELECT COUNT(*) FROM per_repo WHERE dep_count > 0), 0) AS repos_with_dependencies,
                COALESCE((SELECT COUNT(*) FROM per_repo WHERE marked_no_deps), 0) AS repos_marked_no_dependencies,
                COALESCE((
                    SELECT COUNT(*)
                    FROM repo_dependencies
                    WHERE package_name != '__none__' AND is_direct = true
                ), 0) AS dependency_rows,
                COALESCE((
                    SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY dep_count)
                    FROM per_repo
                ), 0) AS p50_deps_per_scanned_repo,
                COALESCE((
                    SELECT percentile_cont(0.95) WITHIN GROUP (ORDER BY dep_count)
                    FROM per_repo
                ), 0) AS p95_deps_per_scanned_repo,
                COALESCE((
                    SELECT COUNT(*)
                    FROM per_repo
                    WHERE last_fetched_at >= NOW() - INTERVAL '1 hour'
                ), 0) AS repos_scanned_last_hour,
                COALESCE((
                    SELECT COUNT(*)
                    FROM per_repo
                    WHERE last_fetched_at >= NOW() - INTERVAL '24 hours'
                ), 0) AS repos_scanned_last_24h
            """
        )
    )
    row = result.fetchone()

    total_repos = int(row.total_public_repos or 0)
    repos_scanned = int(row.repos_scanned or 0)
    repos_remaining = max(total_repos - repos_scanned, 0)
    repos_last_hour = int(row.repos_scanned_last_hour or 0)
    repos_last_24h = int(row.repos_scanned_last_24h or 0)

    rate_per_hour = 0.0
    if repos_last_hour > 0:
        rate_per_hour = float(repos_last_hour)
    elif repos_last_24h > 0:
        rate_per_hour = float(repos_last_24h) / 24.0

    estimated_hours_remaining = None
    if rate_per_hour > 0:
        estimated_hours_remaining = round(repos_remaining / rate_per_hour, 2)

    avg_deps = round((int(row.dependency_rows or 0) / repos_scanned), 2) if repos_scanned else 0.0

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "postgres_live",
        "available": True,
        "backfill_scope": "repo_dependencies coverage",
        "repos": {
            "total": total_repos,
            "scanned": repos_scanned,
            "remaining": repos_remaining,
            "percent_complete": round((repos_scanned / total_repos * 100) if total_repos else 0.0, 2),
            "with_dependencies": int(row.repos_with_dependencies or 0),
            "marked_no_dependencies": int(row.repos_marked_no_dependencies or 0),
        },
        "dependencies": {
            "rows": int(row.dependency_rows or 0),
            "avg_per_scanned_repo": avg_deps,
            "p50_per_scanned_repo": round(float(row.p50_deps_per_scanned_repo or 0.0), 2),
            "p95_per_scanned_repo": round(float(row.p95_deps_per_scanned_repo or 0.0), 2),
        },
        "throughput": {
            "repos_scanned_last_hour": repos_last_hour,
            "repos_scanned_last_24h": repos_last_24h,
            "estimated_hours_remaining": estimated_hours_remaining,
        },
        "notes": [
            "Coverage is exact when repos with zero dependencies write sentinel rows.",
            "ETA falls back from 1h throughput to a 24h average when recent scan volume is low.",
        ],
    }


def _edge_participants(edges: set[tuple[str, str]]) -> set[str]:
    participants: set[str] = set()
    for source_repo_id, target_repo_id in edges:
        participants.add(source_repo_id)
        participants.add(target_repo_id)
    return participants


async def _graph_quality_snapshot(db: AsyncSession) -> dict[str, Any]:
    missing_tables = [
        table_name
        for table_name in ("repo_edges", "repo_dependencies", "repo_categories")
        if not await _table_exists(db, table_name)
    ]
    if missing_tables:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": "postgres_live",
            "available": False,
            "reason": f"Missing tables: {', '.join(missing_tables)}",
        }

    has_integration_tags = await _column_exists(db, "repos", "integration_tags")
    repo_select = """
        SELECT
            id::text AS repo_id,
            owner,
            name,
            forked_from,
            is_fork,
            {integration_tags} AS integration_tags
        FROM repos
        WHERE is_private = false
    """.format(
        integration_tags=(
            "COALESCE(integration_tags, '[]'::jsonb)"
            if has_integration_tags
            else "'[]'::jsonb"
        )
    )
    repo_rows = (await db.execute(text(repo_select))).mappings().all()
    category_rows = (
        await db.execute(
            text(
                """
                SELECT repo_id::text AS repo_id, category_id
                FROM repo_categories
                WHERE is_primary = true
                """
            )
        )
    ).mappings().all()
    dependency_rows = (
        await db.execute(
            text(
                """
                SELECT repo_id::text AS repo_id, package_name
                FROM repo_dependencies
                WHERE is_direct = true
                  AND package_name != '__none__'
                """
            )
        )
    ).mappings().all()
    edge_rows = (
        await db.execute(
            text(
                """
                SELECT
                    edge_type,
                    source_repo_id::text AS source_repo_id,
                    target_repo_id::text AS target_repo_id
                FROM repo_edges
                """
            )
        )
    ).mappings().all()

    repo_by_id = {str(row["repo_id"]): row for row in repo_rows}
    repo_name_index: dict[str, str] = {}
    full_name_index: dict[str, str] = {}
    for row in repo_rows:
        repo_id = str(row["repo_id"])
        upstream_or_name = row["forked_from"] or row["name"]
        normalized_name = _normalize_repo_name(upstream_or_name.split("/")[-1])
        if normalized_name:
            repo_name_index[normalized_name] = repo_id
        full_name_index[f"{row['owner']}/{row['name']}"] = repo_id

    primary_category_by_repo = {
        str(row["repo_id"]): row["category_id"]
        for row in category_rows
    }

    live_edges_by_type: dict[str, set[tuple[str, str]]] = {}
    for row in edge_rows:
        live_edges_by_type.setdefault(row["edge_type"], set()).add(
            (str(row["source_repo_id"]), str(row["target_repo_id"]))
        )

    # DEPENDS_ON exact validation - same normalization as build_knowledge_graph.py.
    candidate_depends_on: set[tuple[str, str]] = set()
    for row in dependency_rows:
        target_repo_id = repo_name_index.get(_normalize_repo_name(row["package_name"]))
        source_repo_id = str(row["repo_id"])
        if target_repo_id and source_repo_id != target_repo_id:
            candidate_depends_on.add((source_repo_id, target_repo_id))

    live_depends_on = live_edges_by_type.get("DEPENDS_ON", set())
    matched_depends_on = live_depends_on & candidate_depends_on
    eligible_dep_repos = _edge_participants(candidate_depends_on)
    observed_dep_repos = _edge_participants(live_depends_on)

    depends_on = {
        "live_edges": len(live_depends_on),
        "candidate_edges": len(candidate_depends_on),
        "matched_edges": len(matched_depends_on),
        "precision": _ratio(len(matched_depends_on), len(live_depends_on)),
        "recall": _ratio(len(matched_depends_on), len(candidate_depends_on)),
        "source_coverage_recall": _ratio(len(observed_dep_repos), len(eligible_dep_repos)),
        "missing_live_edges": len(candidate_depends_on - live_depends_on),
        "unexpected_live_edges": len(live_depends_on - candidate_depends_on),
    }

    # ALTERNATIVE_TO proxy - valid when both repos still share the same primary category.
    live_alternative = live_edges_by_type.get("ALTERNATIVE_TO", set())
    valid_alternative = {
        edge for edge in live_alternative
        if primary_category_by_repo.get(edge[0]) and primary_category_by_repo.get(edge[0]) == primary_category_by_repo.get(edge[1])
    }
    eligible_alternative_repos = set(primary_category_by_repo.keys())

    # COMPATIBLE_WITH proxy - valid when both repos still share >=2 integration tags.
    live_compatible = live_edges_by_type.get("COMPATIBLE_WITH", set())
    valid_compatible: set[tuple[str, str]] = set()
    eligible_compatible_repos = {
        repo_id
        for repo_id, row in repo_by_id.items()
        if row["integration_tags"]
    }
    for edge in live_compatible:
        source_row = repo_by_id.get(edge[0])
        target_row = repo_by_id.get(edge[1])
        if not source_row or not target_row:
            continue
        source_tags = {str(tag).lower() for tag in (source_row["integration_tags"] or []) if tag}
        target_tags = {str(tag).lower() for tag in (target_row["integration_tags"] or []) if tag}
        if len(source_tags & target_tags) >= 2:
            valid_compatible.add(edge)

    # EXTENDS proxy - valid when the source fork still points at the target repo full name.
    live_extends = live_edges_by_type.get("EXTENDS", set())
    valid_extends: set[tuple[str, str]] = set()
    eligible_extends_repos: set[str] = set()
    for repo_id, row in repo_by_id.items():
        forked_from = row["forked_from"]
        if not row["is_fork"] or not forked_from:
            continue
        target_repo_id = full_name_index.get(forked_from)
        if target_repo_id:
            eligible_extends_repos.add(repo_id)
            if (repo_id, target_repo_id) in live_extends:
                valid_extends.add((repo_id, target_repo_id))

    edge_types = {
        "DEPENDS_ON": depends_on,
        "ALTERNATIVE_TO": {
            "live_edges": len(live_alternative),
            "eligible_repos": len(eligible_alternative_repos),
            "observed_repos": len(_edge_participants(live_alternative)),
            "precision_proxy": _ratio(len(valid_alternative), len(live_alternative)),
            "recall_proxy": _ratio(
                len(_edge_participants(live_alternative)),
                len(eligible_alternative_repos),
            ),
            "invalid_live_edges": len(live_alternative - valid_alternative),
        },
        "COMPATIBLE_WITH": {
            "live_edges": len(live_compatible),
            "eligible_repos": len(eligible_compatible_repos),
            "observed_repos": len(_edge_participants(live_compatible)),
            "precision_proxy": _ratio(len(valid_compatible), len(live_compatible)),
            "recall_proxy": _ratio(
                len(_edge_participants(live_compatible)),
                len(eligible_compatible_repos),
            ),
            "invalid_live_edges": len(live_compatible - valid_compatible),
        },
        "EXTENDS": {
            "live_edges": len(live_extends),
            "eligible_repos": len(eligible_extends_repos),
            "observed_repos": len(_edge_participants(live_extends)),
            "precision_proxy": _ratio(len(valid_extends), len(live_extends)),
            "recall_proxy": _ratio(
                len(_edge_participants(live_extends)),
                len(eligible_extends_repos),
            ),
            "invalid_live_edges": len(live_extends - valid_extends),
        },
    }

    total_edges = sum(len(edges) for edges in live_edges_by_type.values())
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "postgres_live",
        "available": True,
        "summary": {
            "total_edges": total_edges,
            "edge_types_present": sorted(live_edges_by_type.keys()),
            "repo_count": len(repo_by_id),
        },
        "edge_types": edge_types,
        "notes": [
            "DEPENDS_ON precision/recall is exact against the current repo_dependencies corpus.",
            "ALTERNATIVE_TO, COMPATIBLE_WITH, and EXTENDS use operational proxies rather than human-labeled ground truth.",
        ],
    }


@router.get("/metrics/latest", response_model=dict)
async def metrics_latest(
    db: AsyncSession = Depends(get_db),
    _gate: None = Depends(require_metrics_access),
) -> dict:
    """Platform metrics for reporium-metrics to consume."""
    total = (await db.execute(select(func.count(Repo.id)))).scalar_one()

    repos_with_skills = (
        await db.execute(
            select(func.count(distinct(RepoAIDevSkill.repo_id)))
        )
    ).scalar_one()

    repos_with_categories = (
        await db.execute(
            select(func.count(func.distinct(RepoCategory.repo_id)))
        )
    ).scalar_one()

    lang_count = (
        await db.execute(
            select(func.count(func.distinct(Repo.primary_language)))
            .where(Repo.primary_language.is_not(None))
        )
    ).scalar_one()

    last_updated = (
        await db.execute(select(func.max(Repo.updated_at)))
    ).scalar_one()

    # KAN-122: additive aliases expected by the frontend dashboard.
    # total_public_repos / repos_with_embeddings exclude private repos.
    counts_row = (
        await db.execute(
            text(
                """
                SELECT
                    (SELECT COUNT(*) FROM repos WHERE is_private = false) AS total_public,
                    (SELECT COUNT(DISTINCT re.repo_id)
                     FROM repo_embeddings re
                     JOIN repos r ON r.id = re.repo_id
                     WHERE r.is_private = false
                       AND re.embedding_vec IS NOT NULL) AS with_embeddings
                """
            )
        )
    ).fetchone()
    total_public = int(counts_row.total_public or 0) if counts_row else 0
    with_embeddings = int(counts_row.with_embeddings or 0) if counts_row else 0

    # snapshot_generated_at: read from the in-memory snapshot cache so this
    # endpoint needs no extra GCS round-trip.
    from app.graph_snapshot import _snapshot_cache  # local import; module already loaded
    snapshot_generated_at = (
        _snapshot_cache.get("generated_at") if _snapshot_cache else None
    )
    # Total typed+similarity edges from the snapshot stats block (if available).
    _snap_stats = (_snapshot_cache or {}).get("stats", {})
    total_edges = (
        int(_snap_stats.get("total_similarity_edges") or 0)
        + int(_snap_stats.get("total_typed_edges") or 0)
    ) or None

    enriched_pct = (
        round(repos_with_categories / total * 100, 1) if total > 0 else None
    )

    return {
        "repos_tracked": total,
        "repos_with_ai_skills": repos_with_skills,
        "repos_with_categories": repos_with_categories,
        "languages": lang_count,
        "last_sync": last_updated.isoformat() if last_updated else None,
        "api_version": os.getenv("APP_VERSION", os.getenv("GITHUB_SHA", "unknown")[:7]),
        "build_number": os.getenv("BUILD_NUMBER", "0"),
        # Frontend-expected aliases (additive — do NOT remove the keys above)
        "total_public_repos": total_public,
        "repos_with_embeddings": with_embeddings,
        "snapshot_generated_at": snapshot_generated_at,
        # Workato connector aliases (additive — KAN-162 fix)
        "total_repos": total,
        "total_edges": total_edges,
        "enriched_pct": enriched_pct,
        "graph_source": "snapshot" if _snapshot_cache else None,
    }


@router.get("/platform/metrics", response_model=dict, include_in_schema=False)
async def platform_metrics_alias(
    db: AsyncSession = Depends(get_db),
    _gate: None = Depends(require_metrics_access),
) -> dict:
    """Legacy alias for /metrics/latest — kept for Workato connector compatibility."""
    return await metrics_latest(db=db, _gate=_gate)


@router.get("/audit/status", response_model=dict)
async def audit_status(
    db: AsyncSession = Depends(get_db),
    _gate: None = Depends(require_metrics_access),
) -> dict:
    """Platform health for reporium-roadmap to consume."""
    db_ok = False
    try:
        await db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        pass

    total = 0
    last_sync = None
    if db_ok:
        total = (await db.execute(select(func.count(Repo.id)))).scalar_one()
        last_updated = (await db.execute(select(func.max(Repo.updated_at)))).scalar_one()
        last_sync = last_updated.isoformat() if last_updated else None

    return {
        "api": "ok" if db_ok else "degraded",
        "database": "ok" if db_ok else "error",
        "repos_tracked": total,
        "last_reporium_db_sync": last_sync,
        "last_forksync_run": None,
        "ingestion_status": "not_running",
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/metrics/slo", response_model=dict)
async def metrics_slo(
    _gate: None = Depends(require_metrics_access),
) -> dict:
    """
    Live 24h SLO snapshot for the routes documented in docs/SLOs.md.

    Values come from an in-memory rolling histogram populated by the request
    logging middleware. This remains a single-instance smoke view; Prometheus
    lives at /metrics/prometheus for multi-instance Grafana dashboards.
    """
    payload = _latency_payload()

    spend_snapshot = token_observer.get_spend_snapshot()
    total_usd = spend_snapshot["total"]["usd"]
    payload["spend_summary"] = {
        "usd_24h": total_usd,
        "cache_hit_rate": spend_snapshot["total"]["cache_hit_rate"],
        "status": _spend_status(total_usd, settings.spend_daily_budget_usd),
    }
    return payload


@router.get("/metrics/latency", response_model=dict)
@_limiter.limit("30/minute")
async def metrics_latency(
    request: Request,
    _gate: None = Depends(require_metrics_access),
) -> dict:
    """
    Route latency snapshot with p50/p95/p99 by endpoint.

    Keeps the payload Grafana-friendly and focused on read-heavy paths such as
    /graph/edges without requiring callers to unpack the broader /metrics/export
    envelope.
    """
    _ = request
    payload = _latency_payload()
    payload["prometheus_endpoint"] = "/metrics/prometheus"
    return payload


@router.get("/metrics/backfill", response_model=dict)
@_limiter.limit("30/minute")
async def metrics_backfill(
    request: Request,
    db: AsyncSession = Depends(get_db),
    _gate: None = Depends(require_metrics_access),
) -> dict:
    """Dependency backfill coverage, throughput, and ETA proxies."""
    _ = request
    return await _backfill_snapshot(db)


@router.get("/metrics/graph-quality", response_model=dict)
@_limiter.limit("30/minute")
async def metrics_graph_quality(
    request: Request,
    db: AsyncSession = Depends(get_db),
    _gate: None = Depends(require_metrics_access),
) -> dict:
    """Read-only graph quality metrics derived from current repo_edges state."""
    _ = request
    return await _graph_quality_snapshot(db)


@router.get("/metrics/data-quality", response_model=dict)
@_limiter.limit("30/minute")
async def metrics_data_quality(
    request: Request,
    db: AsyncSession = Depends(get_db),
    _gate: None = Depends(require_metrics_access),
) -> dict:
    """Aggregate counts the scheduled data-quality gate workflow reads over HTTPS.

    Replaces the workflow's previous direct psycopg2 connection to private-IP
    Cloud SQL, which cannot succeed from a GitHub-hosted runner.  All fields
    are public-filterable counts — no per-repo data is returned.
    """
    _ = request
    row = (
        await db.execute(
            text(
                """
                SELECT
                    (SELECT COUNT(*) FROM repos WHERE is_private = false) AS total_public,
                    (SELECT COUNT(*) FROM repos
                       WHERE is_private = false AND primary_category IS NOT NULL)
                      AS public_with_primary_category,
                    (SELECT COUNT(*) FROM repos
                       WHERE is_private = false
                         AND readme_summary IS NOT NULL
                         AND readme_summary <> '')
                      AS public_with_readme_summary,
                    (SELECT COUNT(DISTINCT re.repo_id)
                       FROM repo_embeddings re
                       JOIN repos r ON r.id = re.repo_id
                      WHERE r.is_private = false
                        AND re.embedding_vec IS NOT NULL)
                      AS public_with_embeddings,
                    (SELECT COUNT(*) FROM repos WHERE is_private IS NULL)
                      AS null_is_private_count
                """
            )
        )
    ).fetchone()
    total_public = int(row.total_public or 0) if row else 0

    # Surface the most recently-ingested public repos that are still missing a
    # primary_category. The data-quality gate is fed by the
    # reporium-ingestion enrichment job; when this gate fails the operator
    # needs to know *which* repos to chase, not just the percentage. Cap at
    # 10 names so the response stays compact.
    missing_sample_rows = (
        await db.execute(
            text(
                """
                SELECT owner || '/' || name AS full_name,
                       ingested_at
                FROM repos
                WHERE is_private = false
                  AND primary_category IS NULL
                ORDER BY ingested_at DESC NULLS LAST
                LIMIT 10
                """
            )
        )
    ).fetchall()
    missing_primary_category_sample = [
        {
            "name": r.full_name,
            "ingested_at": r.ingested_at.isoformat() if r.ingested_at else None,
        }
        for r in missing_sample_rows
    ]

    return {
        "total_public_repos": total_public,
        "public_with_primary_category": int(row.public_with_primary_category or 0) if row else 0,
        "public_with_readme_summary": int(row.public_with_readme_summary or 0) if row else 0,
        "public_with_embeddings": int(row.public_with_embeddings or 0) if row else 0,
        "null_is_private_count": int(row.null_is_private_count or 0) if row else 0,
        "missing_primary_category_sample": missing_primary_category_sample,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/metrics/prometheus", include_in_schema=False)
async def metrics_prometheus(
    _gate: None = Depends(require_metrics_access),
) -> Response:
    """Prometheus exposition endpoint for Grafana/Cloud Monitoring scraping."""
    payload, media_type = render_latest_metrics()
    return Response(content=payload, media_type=media_type)


@router.get("/metrics/spend", response_model=dict)
@_limiter.limit("30/minute")
async def metrics_spend(
    request: Request,
    _gate: None = Depends(require_metrics_access),
) -> dict:
    """
    Live 24h LLM token-spend snapshot for cost observability.

    Values come from an in-memory rolling accumulator populated by
    /intelligence/ask and /intelligence/nl-filter. Same caveat as /metrics/slo:
    single-process only, intended for dashboards and on-call debugging, not a
    replacement for billing.
    """
    _ = request
    snapshot = token_observer.get_spend_snapshot()
    budget = settings.spend_daily_budget_usd
    total = snapshot["total"]

    return {
        "window_seconds": 24 * 60 * 60,
        "source": "in_memory_accumulator",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "daily_budget_usd": budget,
        "total": total,
        "routes": snapshot["routes"],
        "status": _spend_status(total["usd"], budget),
    }


@router.get("/metrics/export", response_model=dict)
@_limiter.limit("30/minute")
async def metrics_export(
    request: Request,
    db: AsyncSession = Depends(get_db),
    _gate: None = Depends(require_metrics_access),
) -> dict:
    """
    Unified metrics export for dashboards and smoke tests.

    Returns latency/SLOs, token spend, embeddings, revision info, plus the
    read-only dependency backfill and graph quality snapshots.
    """
    _ = request
    slo = _latency_payload()

    spend_snapshot = token_observer.get_spend_snapshot()
    total_usd = spend_snapshot["total"]["usd"]
    spend = {
        "usd_24h": total_usd,
        "cache_hit_rate": spend_snapshot["total"]["cache_hit_rate"],
        "status": _spend_status(total_usd, settings.spend_daily_budget_usd),
        "daily_budget_usd": settings.spend_daily_budget_usd,
    }

    counts = await db.execute(
        text(
            """
            SELECT
                (SELECT COUNT(*) FROM repos WHERE is_private = false) AS total_public,
                (SELECT COUNT(DISTINCT re.repo_id)
                 FROM repo_embeddings re
                 JOIN repos r ON r.id = re.repo_id
                 WHERE r.is_private = false
                   AND re.embedding_vec IS NOT NULL) AS with_embeddings
            """
        )
    )
    row = counts.fetchone()
    total_pub = row.total_public if row else 0
    with_emb = row.with_embeddings if row else 0
    embeddings = {
        "total_public_repos": total_pub,
        "repos_with_embeddings": with_emb,
        "coverage_percent": round((with_emb / total_pub * 100) if total_pub > 0 else 0.0, 2),
    }

    revision = {
        "api_version": os.getenv("APP_VERSION", os.getenv("GITHUB_SHA", "unknown")[:7]),
        "build_number": os.getenv("BUILD_NUMBER", "0"),
    }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "slo": slo,
        "spend": spend,
        "embeddings": embeddings,
        "backfill": await _backfill_snapshot(db),
        "graph_quality": await _graph_quality_snapshot(db),
        "revision": revision,
    }


@router.post("/events/ingest", response_model=dict)
async def events_ingest(
    payload: dict,
    _api_key: str = Depends(verify_api_key),
    _ingest_key: None = Depends(require_ingest_key),
) -> dict:
    """Receive placeholder event pushes. Requires API and ingest keys in the current implementation."""
    _ = payload
    return {"status": "accepted", "message": "Event received (processing not yet implemented)"}
