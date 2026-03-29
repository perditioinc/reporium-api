import hashlib
import json

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.cache import CACHE_TTL_LIBRARY, cache
from app.database import get_db
from app.models.repo import Repo, RepoCategory, RepoTag
from app.schemas.library import CategorySummary, LibraryResponse, LibraryStats, TagMetric
from app.schemas.repo import RepoSummary

router = APIRouter(tags=["Library"])


def _repo_to_summary(repo: Repo) -> RepoSummary:
    return RepoSummary(
        id=repo.id,
        name=repo.name,
        owner=repo.owner,
        description=repo.description,
        is_fork=repo.is_fork,
        forked_from=repo.forked_from,
        primary_language=repo.primary_language,
        github_url=repo.github_url,
        fork_sync_state=repo.fork_sync_state,
        behind_by=repo.behind_by,
        ahead_by=repo.ahead_by,
        upstream_created_at=repo.upstream_created_at,
        forked_at=repo.forked_at,
        your_last_push_at=repo.your_last_push_at,
        upstream_last_push_at=repo.upstream_last_push_at,
        parent_stars=repo.parent_stars,
        parent_forks=repo.parent_forks,
        parent_is_archived=repo.parent_is_archived,
        stargazers_count=repo.stargazers_count,
        open_issues_count=repo.open_issues_count,
        license_spdx=repo.license_spdx,
        commits_last_7_days=repo.commits_last_7_days,
        commits_last_30_days=repo.commits_last_30_days,
        commits_last_90_days=repo.commits_last_90_days,
        readme_summary=repo.readme_summary,
        activity_score=repo.activity_score,
        quality_signals=repo.quality_signals,
        problem_solved=repo.problem_solved,
        ingested_at=repo.ingested_at,
        updated_at=repo.updated_at,
        github_updated_at=repo.github_updated_at,
        tags=[t.tag for t in repo.tags],
        categories=[
            {"category_id": c.category_id, "category_name": c.category_name, "is_primary": c.is_primary}
            for c in repo.categories
        ],
        allCategories=[c.category_name for c in repo.categories],
        builders=[
            {
                "login": b.login,
                "display_name": b.display_name,
                "org_category": b.org_category,
                "is_known_org": b.is_known_org,
            }
            for b in repo.builders
        ],
        ai_dev_skills=[s.skill for s in repo.ai_dev_skills],
        pm_skills=[s.skill for s in repo.pm_skills],
        languages=[
            {"language": l.language, "bytes": l.bytes, "percentage": l.percentage}
            for l in repo.languages
        ],
        taxonomy=[
            {"dimension": t.dimension, "value": t.raw_value, "similarityScore": t.similarity_score, "assignedBy": t.assigned_by}
            for t in getattr(repo, "taxonomy", [])
        ],
        security_signals=getattr(repo, "security_signals", None),
    )


@router.get("/library", response_model=LibraryResponse)
async def get_library(
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=100, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
) -> LibraryResponse:
    cache_key = f"library:full:{page}:{limit}"
    cached = await cache.get(cache_key)
    if cached:
        return LibraryResponse(**cached)

    offset = (page - 1) * limit

    # Load repos with all relationships
    stmt = (
        select(Repo)
        .where(Repo.is_private == False)  # noqa: E712 — SECURITY: never expose private repos
        .options(
            selectinload(Repo.tags),
            selectinload(Repo.categories),
            selectinload(Repo.builders),
            selectinload(Repo.ai_dev_skills),
            selectinload(Repo.pm_skills),
            selectinload(Repo.languages),
            selectinload(Repo.taxonomy),
        )
        .order_by(Repo.updated_at.desc())
        .offset(offset)
        .limit(limit)
    )
    result = await db.execute(stmt)
    repos = result.scalars().all()

    total_stmt = select(func.count(Repo.id)).where(Repo.is_private == False)  # noqa: E712
    total_result = await db.execute(total_stmt)
    total = total_result.scalar_one()

    # Build per-page stats (language distribution uses the current page only)
    lang_counts: dict[str, int] = {}
    page_tag_commit_map: dict[str, dict] = {}
    for repo in repos:
        if repo.primary_language:
            lang_counts[repo.primary_language] = lang_counts.get(repo.primary_language, 0) + 1
        for t in repo.tags:
            if t.tag not in page_tag_commit_map:
                page_tag_commit_map[t.tag] = {"count": 0, "commits": 0}
            page_tag_commit_map[t.tag]["count"] += 1
            page_tag_commit_map[t.tag]["commits"] += repo.commits_last_7_days

    # Top tags — query across ALL repo_tags so the tag cloud reflects the
    # full corpus, not just the current page's 100 repos.
    # No LIMIT: return every unique tag; the frontend can truncate as desired.
    global_tags_stmt = (
        select(RepoTag.tag, func.count(RepoTag.repo_id).label("cnt"))
        .group_by(RepoTag.tag)
        .order_by(func.count(RepoTag.repo_id).desc())
    )
    global_tags_result = await db.execute(global_tags_stmt)
    global_top_tags = [row.tag for row in global_tags_result.all()]

    # Categories
    cat_stmt = (
        select(RepoCategory.category_id, RepoCategory.category_name, func.count().label("cnt"))
        .group_by(RepoCategory.category_id, RepoCategory.category_name)
        .order_by(func.count().desc())
    )
    cat_result = await db.execute(cat_stmt)
    categories = [
        CategorySummary(id=row.category_id, name=row.category_name, count=row.cnt)
        for row in cat_result.all()
    ]

    tag_metrics = [
        TagMetric(
            tag=tag,
            count=data["count"],
            commit_velocity=data["commits"] / max(data["count"], 1),
        )
        for tag, data in sorted(page_tag_commit_map.items(), key=lambda x: x[1]["count"], reverse=True)
    ]

    stats = LibraryStats(
        total_repos=total,
        total_forks=sum(1 for r in repos if r.is_fork),
        total_non_forks=sum(1 for r in repos if not r.is_fork),
        languages=lang_counts,
        top_tags=global_top_tags,
    )

    response = LibraryResponse(
        repos=[_repo_to_summary(r) for r in repos],
        stats=stats,
        categories=categories,
        tag_metrics=tag_metrics,
        total=total,
        page=page,
        limit=limit,
    )

    await cache.set(cache_key, response.model_dump(), ttl=CACHE_TTL_LIBRARY)
    return response
