"""Platform-level endpoints consumed by sibling repos (reporium-metrics, reporium-roadmap)."""

import os
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.repo import Repo, RepoAIDevSkill, RepoCategory

router = APIRouter()


@router.get("/metrics/latest")
async def metrics_latest(db: AsyncSession = Depends(get_db)) -> dict:
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


@router.get("/audit/status")
async def audit_status(db: AsyncSession = Depends(get_db)) -> dict:
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


@router.post("/events/ingest")
async def events_ingest(payload: dict) -> dict:
    """Receive Pub/Sub push events (placeholder for future reporium-events integration)."""
    # For now, acknowledge receipt without processing
    return {"status": "accepted", "message": "Event received (processing not yet implemented)"}
