from fastapi import APIRouter, Depends
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.cache import CACHE_TTL_GAPS, CACHE_TTL_STATS, CACHE_TTL_TRENDS, cache
from app.database import get_db
from app.models.repo import Repo, RepoCategory, RepoTag
from app.models.trend import GapAnalysis, IngestionLog, TrendSnapshot
from app.schemas.trend import GapAnalysisOut, IngestionLogOut, StatsResponse, TaxonomyGapItem, TrendSnapshotOut

router = APIRouter(tags=["Trends"])


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


@router.get("/gaps", response_model=list[GapAnalysisOut], tags=["Trends", "Taxonomy"])
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

    # Taxonomy dimension counts (distinct values per dimension)
    tax_dim_rows = (await db.execute(text(
        "SELECT dimension, COUNT(DISTINCT raw_value) AS cnt "
        "FROM repo_taxonomy GROUP BY dimension"
    ))).fetchall()
    taxonomy_dimension_counts = {row.dimension: row.cnt for row in tax_dim_rows}

    # has_tests / has_ci counts
    has_tests_count = (await db.execute(
        select(func.count(Repo.id)).where(Repo.has_tests == True)  # noqa: E712
    )).scalar_one()
    has_ci_count = (await db.execute(
        select(func.count(Repo.id)).where(Repo.has_ci == True)  # noqa: E712
    )).scalar_one()

    # Average overall_score from quality_signals JSONB
    quality_score_avg_row = (await db.execute(text(
        "SELECT AVG((quality_signals->>'overall_score')::float) "
        "FROM repos WHERE quality_signals IS NOT NULL "
        "AND quality_signals->>'overall_score' IS NOT NULL"
    ))).scalar_one()
    quality_score_avg = round(float(quality_score_avg_row), 2) if quality_score_avg_row is not None else None

    # Enriched repo count (readme_summary is not null)
    enriched_repo_count = (await db.execute(
        select(func.count(Repo.id)).where(Repo.readme_summary.is_not(None))
    )).scalar_one()

    response = StatsResponse(
        total_repos=total,
        total_forks=total_forks,
        total_non_forks=total - total_forks,
        languages=languages,
        categories=categories,
        top_tags=top_tags,
        sync_states=sync_states,
        last_ingestion=last_ingestion,
        taxonomy_dimension_counts=taxonomy_dimension_counts,
        has_tests_count=has_tests_count,
        has_ci_count=has_ci_count,
        quality_score_avg=quality_score_avg,
        enriched_repo_count=enriched_repo_count,
    )
    await cache.set("stats:overview", response.model_dump(), ttl=CACHE_TTL_STATS)
    return response


@router.get("/gaps/taxonomy", response_model=list[TaxonomyGapItem])
async def get_taxonomy_gaps(
    min_repos: int = 1,
    max_repos: int = 10,
    db: AsyncSession = Depends(get_db),
) -> list[TaxonomyGapItem]:
    """
    Compute gap analysis across all 8 taxonomy dimensions from repo_taxonomy.
    Returns values that are underrepresented relative to the rest of their dimension.

    Args:
        min_repos: Only include values with at least this many repos (filters noise).
        max_repos: Only include values with at most this many repos (these are the gaps).
    """
    cache_key = f"gaps:taxonomy:{min_repos}:{max_repos}"
    cached = await cache.get(cache_key)
    if cached:
        return [TaxonomyGapItem(**g) for g in cached]

    from sqlalchemy import text as _text

    # Get repo counts per (dimension, raw_value)
    rows = (await db.execute(_text("""
        SELECT dimension, raw_value, COUNT(DISTINCT repo_id) AS repo_count
        FROM repo_taxonomy
        GROUP BY dimension, raw_value
        HAVING COUNT(DISTINCT repo_id) BETWEEN :min_repos AND :max_repos
        ORDER BY dimension, repo_count ASC
    """), {"min_repos": min_repos, "max_repos": max_repos})).fetchall()

    if not rows:
        return []

    # Get max count per dimension for normalisation
    max_rows = (await db.execute(_text("""
        SELECT dimension, MAX(cnt) AS max_count FROM (
            SELECT dimension, COUNT(DISTINCT repo_id) AS cnt
            FROM repo_taxonomy
            GROUP BY dimension, raw_value
        ) sub
        GROUP BY dimension
    """))).fetchall()
    max_by_dim = {r.dimension: r.max_count for r in max_rows}

    gaps: list[TaxonomyGapItem] = []
    for row in rows:
        max_count = max_by_dim.get(row.dimension, 1) or 1
        gap_score = round(1.0 - (row.repo_count / max_count), 3)
        if gap_score < 0.3:
            severity = "low"
        elif gap_score < 0.7:
            severity = "medium"
        else:
            severity = "high"
        gaps.append(TaxonomyGapItem(
            dimension=row.dimension,
            name=row.raw_value,
            repo_count=row.repo_count,
            gap_score=gap_score,
            severity=severity,
        ))

    # Sort: highest gap first, then by dimension
    gaps.sort(key=lambda g: (-g.gap_score, g.dimension))

    await cache.set(cache_key, [g.model_dump() for g in gaps], ttl=CACHE_TTL_GAPS)
    return gaps
