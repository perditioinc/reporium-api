"""Platform-level endpoints consumed by sibling repos (reporium-metrics, reporium-roadmap)."""

import os
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import require_ingest_key, require_metrics_access, verify_api_key
from app.config import settings
from app.database import get_db
from app.models.repo import Repo, RepoAIDevSkill, RepoCategory
from app.rate_limit import rate_limit_storage
from app.slo_observer import slo_observer, token_observer

# Shared limiter — matches the pattern used in intelligence.py / nl_filter.py.
_limiter = Limiter(key_func=get_remote_address, storage_uri=rate_limit_storage)

# SLO targets documented in docs/SLOs.md. These are the thresholds the
# /metrics/slo endpoint compares live values against. Keeping the dict here
# (rather than in slo_observer) keeps the observer pure and testable.
_SLO_TARGETS: dict[str, dict] = {
    "/health": {"p95_ms": 500, "max_error_rate": 0.001},
    "/library/full": {"p95_ms": 2000, "max_error_rate": 0.01},
    "/intelligence/ask": {"p95_ms": 15000, "p99_ms": 25000, "max_error_rate": 0.01},
    "/intelligence/nl-filter": {"p95_ms": 3000, "max_error_rate": 0.01},
}

router = APIRouter(tags=["Platform"])


@router.get("/metrics/latest", response_model=dict)
async def metrics_latest(
    db: AsyncSession = Depends(get_db),
    _gate: None = Depends(require_metrics_access),
) -> dict:
    """Platform metrics for reporium-metrics to consume."""
    total = (await db.execute(select(func.count(Repo.id)))).scalar_one()

    repos_with_skills = (
        await db.execute(
            select(func.count(func.distinct(RepoAIDevSkill.repo_id)))
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

    return {
        "repos_tracked": total,
        "repos_with_ai_skills": repos_with_skills,
        "repos_with_categories": repos_with_categories,
        "languages": lang_count,
        "last_sync": last_updated.isoformat() if last_updated else None,
        "api_version": os.getenv("APP_VERSION", os.getenv("GITHUB_SHA", "unknown")[:7]),
        "build_number": os.getenv("BUILD_NUMBER", "0"),
    }


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
    logging middleware — single-process only, no Prometheus yet. This endpoint
    is intended for smoke-level dashboarding and on-call debugging; it is NOT
    a replacement for Cloud Monitoring.
    """
    snapshot = slo_observer.snapshot()
    routes: dict[str, dict] = {}
    for route, target in _SLO_TARGETS.items():
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

    # KAN-ask-spend: surface a compact cost summary alongside latency/error SLOs
    # so dashboards pulling /metrics/slo get token spend for free.
    spend_snapshot = token_observer.get_spend_snapshot()
    total_usd = spend_snapshot["total"]["usd"]
    spend_status = _spend_status(total_usd, settings.spend_daily_budget_usd)
    spend_summary = {
        "usd_24h": total_usd,
        "cache_hit_rate": spend_snapshot["total"]["cache_hit_rate"],
        "status": spend_status,
    }

    return {
        "window_seconds": 24 * 60 * 60,
        "source": "in_memory_histogram",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "routes": routes,
        "spend_summary": spend_summary,
    }


def _spend_status(total_usd: float, budget_usd: float) -> str:
    """
    Map total spend against the soft daily budget:
      < 80%  -> ok
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
    single-process only, intended for dashboards and on-call debugging, NOT a
    replacement for billing.

    The top-level ``status`` field maps the total 24h spend against the soft
    daily budget (``SPEND_DAILY_BUDGET_USD``, default $10):

      < 80%    -> ok
      80-100%  -> warning
      >= 100%  -> breach
    """
    snapshot = token_observer.get_spend_snapshot()
    budget = settings.spend_daily_budget_usd
    total = snapshot["total"]
    status = _spend_status(total["usd"], budget)

    return {
        "window_seconds": 24 * 60 * 60,
        "source": "in_memory_accumulator",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "daily_budget_usd": budget,
        "total": total,
        "routes": snapshot["routes"],
        "status": status,
    }


@router.post("/events/ingest", response_model=dict)
async def events_ingest(
    payload: dict,
    _api_key: str = Depends(verify_api_key),
    _ingest_key: None = Depends(require_ingest_key),
) -> dict:
    """Receive placeholder event pushes. Requires API and ingest keys in the current implementation."""
    # For now, acknowledge receipt without processing
    return {"status": "accepted", "message": "Event received (processing not yet implemented)"}
