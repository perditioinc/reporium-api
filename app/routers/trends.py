from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Response
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.cache import CACHE_TTL_GAPS, CACHE_TTL_STATS, CACHE_TTL_TRENDS, cache
from app.database import get_db
from app.models.repo import Repo, RepoCategory, RepoTag
from app.models.trend import GapAnalysis, IngestionLog, TrendSnapshot
from app.schemas.trend import (
    GapAnalysisOut,
    IngestionLogOut,
    StatsResponse,
    TaxonomyGapItem,
    TrendReportOut,
    TrendReportPeriodOut,
    TrendReportSignalOut,
    TrendSnapshotOut,
)

router = APIRouter(tags=["Trends"])


def _compute_change_percent(current: int, previous: int | None) -> float:
    if previous is None or previous <= 0:
        return 100.0 if current > 0 else 0.0
    return round(((current - previous) / previous) * 100.0, 2)


def _build_trend_report(
    latest_rows: list[dict],
    previous_counts: dict[str, int],
    earliest_snapshot: datetime | None,
    latest_snapshot: datetime | None,
    snapshot_count: int,
    first_seen_rows: list[dict],
) -> TrendReportOut:
    generated_at = latest_snapshot or datetime.now(timezone.utc)
    latest_by_tag = {row["tag"]: row for row in latest_rows}

    trending = [
        TrendReportSignalOut(
            name=row["tag"],
            changePercent=_compute_change_percent(row["commit_count_7d"], previous_counts.get(row["tag"])),
            repoCount=row["repo_count"],
        )
        for row in sorted(latest_rows, key=lambda item: item["commit_count_7d"], reverse=True)[:5]
    ]

    emerging_candidates: list[TrendReportSignalOut] = []
    for row in first_seen_rows:
        latest = latest_by_tag.get(row["tag"])
        if not latest or latest["commit_count_7d"] <= 0:
            continue
        emerging_candidates.append(
            TrendReportSignalOut(
                name=row["tag"],
                changePercent=_compute_change_percent(latest["commit_count_7d"], previous_counts.get(row["tag"])),
                repoCount=latest["repo_count"],
            )
        )
    emerging = sorted(emerging_candidates, key=lambda item: item.repoCount, reverse=True)[:5]

    insights: list[str] = []
    if trending:
        insights.append(
            f"{trending[0].name} is the strongest current trend, with {trending[0].repoCount} repos in the latest snapshot."
        )

    return TrendReportOut(
        generatedAt=generated_at,
        period=TrendReportPeriodOut(from_=earliest_snapshot, to=latest_snapshot, snapshots=snapshot_count),
        trending=trending,
        emerging=emerging,
        cooling=[],
        stable=[],
        newReleases=[],
        insights=insights,
    )


@router.get("/trends", response_model=list[TrendSnapshotOut])
async def get_trends(response: Response, db: AsyncSession = Depends(get_db)) -> list[TrendSnapshotOut]:
    response.headers["Cache-Control"] = "public, max-age=300, stale-while-revalidate=3600"
    cached = await cache.get("trends:latest")
    if cached:
        return [TrendSnapshotOut(**t) for t in cached]

    latest_ts = select(func.max(TrendSnapshot.snapshotted_at)).scalar_subquery()
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


@router.get("/trends/report", response_model=TrendReportOut)
async def get_trends_report(db: AsyncSession = Depends(get_db)) -> TrendReportOut:
    cached = await cache.get("trends:report")
    if cached:
        return TrendReportOut(**cached)

    latest_snapshot = (await db.execute(select(func.max(TrendSnapshot.snapshotted_at)))).scalar_one()
    earliest_snapshot = (await db.execute(select(func.min(TrendSnapshot.snapshotted_at)))).scalar_one()
    snapshot_count = (
        await db.execute(select(func.count(func.distinct(func.date(TrendSnapshot.snapshotted_at)))))
    ).scalar_one()

    if latest_snapshot is None:
        report = TrendReportOut(
            generatedAt=datetime.now(timezone.utc),
            period=TrendReportPeriodOut(from_=None, to=None, snapshots=0),
            trending=[],
            emerging=[],
            cooling=[],
            stable=[],
            newReleases=[],
            insights=[],
        )
        await cache.set("trends:report", report.model_dump(mode="json"), ttl=CACHE_TTL_TRENDS)
        return report

    latest_rows_result = await db.execute(
        select(
            TrendSnapshot.tag.label("tag"),
            TrendSnapshot.repo_count.label("repo_count"),
            TrendSnapshot.commit_count_7d.label("commit_count_7d"),
        )
        .where(TrendSnapshot.snapshotted_at == latest_snapshot)
        .order_by(TrendSnapshot.commit_count_7d.desc(), TrendSnapshot.repo_count.desc())
    )
    latest_rows = [dict(row) for row in latest_rows_result.mappings().all()]

    previous_snapshot = (
        await db.execute(
            select(func.max(TrendSnapshot.snapshotted_at)).where(TrendSnapshot.snapshotted_at < latest_snapshot)
        )
    ).scalar_one()
    previous_counts: dict[str, int] = {}
    if previous_snapshot is not None:
        previous_rows_result = await db.execute(
            select(TrendSnapshot.tag, TrendSnapshot.commit_count_7d).where(
                TrendSnapshot.snapshotted_at == previous_snapshot
            )
        )
        previous_counts = {row.tag: row.commit_count_7d for row in previous_rows_result}

    first_seen_cutoff = latest_snapshot - timedelta(days=30)
    first_seen_rows_result = await db.execute(
        select(
            TrendSnapshot.tag.label("tag"),
            func.min(TrendSnapshot.snapshotted_at).label("first_seen_at"),
        )
        .group_by(TrendSnapshot.tag)
        .having(func.min(TrendSnapshot.snapshotted_at) >= first_seen_cutoff)
        .order_by(func.min(TrendSnapshot.snapshotted_at).desc())
    )
    first_seen_rows = [dict(row) for row in first_seen_rows_result.mappings().all()]

    report = _build_trend_report(
        latest_rows=latest_rows,
        previous_counts=previous_counts,
        earliest_snapshot=earliest_snapshot,
        latest_snapshot=latest_snapshot,
        snapshot_count=snapshot_count or 0,
        first_seen_rows=first_seen_rows,
    )
    await cache.set("trends:report", report.model_dump(mode="json"), ttl=CACHE_TTL_TRENDS)
    return report


@router.get("/gaps", response_model=list[GapAnalysisOut], tags=["Trends", "Taxonomy"])
async def get_gaps(response: Response, db: AsyncSession = Depends(get_db)) -> list[GapAnalysisOut]:
    response.headers["Cache-Control"] = "public, max-age=300, stale-while-revalidate=3600"
    cached = await cache.get("gaps:latest")
    if cached:
        return [GapAnalysisOut(**g) for g in cached]

    latest_ts = select(func.max(GapAnalysis.generated_at)).scalar_subquery()
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

    lang_rows = (
        await db.execute(
            select(Repo.primary_language, func.count().label("cnt"))
            .where(Repo.primary_language.is_not(None))
            .group_by(Repo.primary_language)
            .order_by(func.count().desc())
        )
    ).all()
    languages = {row.primary_language: row.cnt for row in lang_rows}

    cat_rows = (
        await db.execute(
            select(RepoCategory.category_name, func.count().label("cnt"))
            .group_by(RepoCategory.category_name)
            .order_by(func.count().desc())
        )
    ).all()
    categories = {row.category_name: row.cnt for row in cat_rows}

    sync_rows = (
        await db.execute(
            select(Repo.fork_sync_state, func.count().label("cnt"))
            .where(Repo.fork_sync_state.is_not(None))
            .group_by(Repo.fork_sync_state)
        )
    ).all()
    sync_states = {row.fork_sync_state: row.cnt for row in sync_rows}

    last_log_stmt = select(IngestionLog).order_by(IngestionLog.started_at.desc()).limit(1)
    last_log_row = (await db.execute(last_log_stmt)).scalar_one_or_none()
    last_ingestion = (
        IngestionLogOut.model_validate(last_log_row, from_attributes=True)
        if last_log_row
        else None
    )

    system_tags = {"Active", "Forked", "Built by Me", "Inactive", "Archived", "Popular"}
    tag_rows = (
        await db.execute(
            select(RepoTag.tag, func.count().label("cnt"))
            .where(RepoTag.tag.not_in(system_tags))
            .group_by(RepoTag.tag)
            .order_by(func.count().desc())
            .limit(20)
        )
    ).all()
    top_tags = [row.tag for row in tag_rows]

    tax_dim_rows = (
        await db.execute(
            text(
                "SELECT dimension, COUNT(DISTINCT raw_value) AS cnt "
                "FROM repo_taxonomy GROUP BY dimension"
            )
        )
    ).fetchall()
    taxonomy_dimension_counts = {row.dimension: row.cnt for row in tax_dim_rows}

    has_tests_count = (
        await db.execute(select(func.count(Repo.id)).where(Repo.has_tests == True))  # noqa: E712
    ).scalar_one()
    has_ci_count = (
        await db.execute(select(func.count(Repo.id)).where(Repo.has_ci == True))  # noqa: E712
    ).scalar_one()

    quality_score_avg_row = (
        await db.execute(
            text(
                "SELECT AVG((quality_signals->>'overall_score')::float) "
                "FROM repos WHERE quality_signals IS NOT NULL "
                "AND quality_signals->>'overall_score' IS NOT NULL"
            )
        )
    ).scalar_one()
    quality_score_avg = round(float(quality_score_avg_row), 2) if quality_score_avg_row is not None else None

    enriched_repo_count = (
        await db.execute(select(func.count(Repo.id)).where(Repo.readme_summary.is_not(None)))
    ).scalar_one()

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
    cache_key = f"gaps:taxonomy:{min_repos}:{max_repos}"
    cached = await cache.get(cache_key)
    if cached:
        return [TaxonomyGapItem(**g) for g in cached]

    rows = (
        await db.execute(
            text(
                """
                SELECT dimension, raw_value, COUNT(DISTINCT repo_id) AS repo_count
                FROM repo_taxonomy
                GROUP BY dimension, raw_value
                HAVING COUNT(DISTINCT repo_id) BETWEEN :min_repos AND :max_repos
                ORDER BY dimension, repo_count ASC
                """
            ),
            {"min_repos": min_repos, "max_repos": max_repos},
        )
    ).fetchall()

    if not rows:
        return []

    max_rows = (
        await db.execute(
            text(
                """
                SELECT dimension, MAX(cnt) AS max_count FROM (
                    SELECT dimension, COUNT(DISTINCT repo_id) AS cnt
                    FROM repo_taxonomy
                    GROUP BY dimension, raw_value
                ) sub
                GROUP BY dimension
                """
            )
        )
    ).fetchall()
    max_by_dim = {row.dimension: row.max_count for row in max_rows}

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
        gaps.append(
            TaxonomyGapItem(
                dimension=row.dimension,
                name=row.raw_value,
                repo_count=row.repo_count,
                gap_score=gap_score,
                severity=severity,
            )
        )

    gaps.sort(key=lambda gap: (-gap.gap_score, gap.dimension))
    await cache.set(cache_key, [gap.model_dump() for gap in gaps], ttl=CACHE_TTL_GAPS)
    return gaps
