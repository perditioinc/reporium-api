from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.cache import CACHE_TTL_GAPS, CACHE_TTL_STATS, CACHE_TTL_TRENDS, cache
from app.database import get_db
from app.models.repo import Repo, RepoCategory, RepoTag
from app.models.trend import GapAnalysis, IngestionLog, TrendSnapshot
from app.schemas.trend import GapAnalysisOut, IngestionLogOut, StatsResponse, TrendSnapshotOut

router = APIRouter()


@router.get("/trends", response_model=list[TrendSnapshotOut])
async def get_trends(db: AsyncSession = Depends(get_db)) -> list[TrendSnapshotOut]:
    cached = await cache.get("trends:latest")
    if cached:
        return [TrendSnapshotOut(**t) for t in cached]

    # Latest snapshot per tag
    latest_ts = (
        select(func.max(TrendSnapshot.snapshotted_at)).scalar_subquery()
    )
    stmt = (
        select(TrendSnapshot)
        .where(TrendSnapshot.snapshotted_at == latest_ts)
        .order_by(TrendSnapshot.commit_count_7d.desc())
    )
    result = await db.execute(stmt)
    rows = result.scalars().all()
    out = [TrendSnapshotOut.model_validate(r, from_attributes=True) for r in rows]
    await cache.set("trends:latest", [t.model_dump() for t in out], ttl=CACHE_TTL_TRENDS)
    return out


@router.get("/gaps", response_model=list[GapAnalysisOut])
async def get_gaps(db: AsyncSession = Depends(get_db)) -> list[GapAnalysisOut]:
    cached = await cache.get("gaps:latest")
    if cached:
        return [GapAnalysisOut(**g) for g in cached]

    latest_ts = (
        select(func.max(GapAnalysis.generated_at)).scalar_subquery()
    )
    stmt = (
        select(GapAnalysis)
        .where(GapAnalysis.generated_at == latest_ts)
        .order_by(GapAnalysis.severity.asc())
    )
    result = await db.execute(stmt)
    rows = result.scalars().all()
    out = [GapAnalysisOut.model_validate(r, from_attributes=True) for r in rows]
    await cache.set("gaps:latest", [g.model_dump() for g in out], ttl=CACHE_TTL_GAPS)
    return out


@router.get("/stats", response_model=StatsResponse)
async def get_stats(db: AsyncSession = Depends(get_db)) -> StatsResponse:
    cached = await cache.get("stats:overview")
    if cached:
        return StatsResponse(**cached)

    total = (await db.execute(select(func.count(Repo.id)))).scalar_one()
    total_forks = (
        await db.execute(select(func.count(Repo.id)).where(Repo.is_fork == True))  # noqa: E712
    ).scalar_one()

    # Language distribution
    lang_rows = (
        await db.execute(
            select(Repo.primary_language, func.count().label("cnt"))
            .where(Repo.primary_language.is_not(None))
            .group_by(Repo.primary_language)
            .order_by(func.count().desc())
        )
    ).all()
    languages = {row.primary_language: row.cnt for row in lang_rows}

    # Category distribution
    cat_rows = (
        await db.execute(
            select(RepoCategory.category_name, func.count().label("cnt"))
            .group_by(RepoCategory.category_name)
            .order_by(func.count().desc())
        )
    ).all()
    categories = {row.category_name: row.cnt for row in cat_rows}

    # Sync states
    sync_rows = (
        await db.execute(
            select(Repo.fork_sync_state, func.count().label("cnt"))
            .where(Repo.fork_sync_state.is_not(None))
            .group_by(Repo.fork_sync_state)
        )
    ).all()
    sync_states = {row.fork_sync_state: row.cnt for row in sync_rows}

    # Last ingestion
    last_log_stmt = (
        select(IngestionLog)
        .order_by(IngestionLog.started_at.desc())
        .limit(1)
    )
    last_log_row = (await db.execute(last_log_stmt)).scalar_one_or_none()
    last_ingestion = (
        IngestionLogOut.model_validate(last_log_row, from_attributes=True)
        if last_log_row
        else None
    )

    # Top tags by repo count — excludes system tags (Active, Forked, Built by Me)
    _SYSTEM_TAGS = {"Active", "Forked", "Built by Me", "Inactive", "Archived", "Popular"}
    tag_rows = (
        await db.execute(
            select(RepoTag.tag, func.count().label("cnt"))
            .where(RepoTag.tag.not_in(_SYSTEM_TAGS))
            .group_by(RepoTag.tag)
            .order_by(func.count().desc())
            .limit(20)
        )
    ).all()
    top_tags = [row.tag for row in tag_rows]

    response = StatsResponse(
        total_repos=total,
        total_forks=total_forks,
        total_non_forks=total - total_forks,
        languages=languages,
        categories=categories,
        top_tags=top_tags,
        sync_states=sync_states,
        last_ingestion=last_ingestion,
    )
    await cache.set("stats:overview", response.model_dump(), ttl=CACHE_TTL_STATS)
    return response
