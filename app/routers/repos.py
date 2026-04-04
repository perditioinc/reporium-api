import hashlib
import json
import math

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy import func, select, or_, cast, Text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.cache import CACHE_TTL_REPO_DETAIL, cache
from app.database import get_db
from app.models.repo import (
    Repo,
    RepoAIDevSkill,
    RepoBuilder,
    RepoCategory,
    RepoCommit,
    RepoLanguage,
    RepoTag,
    RepoTaxonomy,
)
from app.routers.library import _repo_to_summary
from app.schemas.repo import RepoDetail, RepoSummary

from app.rate_limit import rate_limit_storage

router = APIRouter(tags=["Repos"])
_limiter = Limiter(key_func=get_remote_address, storage_uri=rate_limit_storage)

VALID_SORT = {"stars", "updated", "behind", "name"}
VALID_SYNC = {"up-to-date", "behind", "ahead", "diverged"}


def _repo_to_detail(repo: Repo) -> RepoDetail:
    summary = _repo_to_summary(repo)
    return RepoDetail(
        **summary.model_dump(),
        commits=[
            {
                "sha": c.sha,
                "message": c.message,
                "author": c.author,
                "committed_at": c.committed_at,
                "url": c.url,
            }
            for c in sorted(repo.commits, key=lambda x: x.committed_at, reverse=True)[:20]
        ],
    )


@router.get("/repos", response_model=dict)
async def list_repos(
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=50, ge=1, le=200),
    category: str | None = None,
    tag: str | None = None,
    topic: str | None = None,
    builder: str | None = None,
    ai_dev_skill: str | None = None,
    language: str | None = None,
    license: str | None = None,
    min_stars: int | None = None,
    sync_status: str | None = None,
    sort: str = Query(default="updated"),
    q: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> dict:
    # Build cache key from params
    params = {
        "page": page, "limit": limit, "category": category, "tag": tag,
        "topic": topic, "builder": builder, "ai_dev_skill": ai_dev_skill,
        "language": language, "license": license, "min_stars": min_stars,
        "sync_status": sync_status, "sort": sort, "q": q,
    }
    param_hash = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()
    cache_key = f"repos:list:{param_hash}"
    cached = await cache.get(cache_key)
    if cached:
        return cached

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
    )

    if category:
        stmt = stmt.join(RepoCategory, RepoCategory.repo_id == Repo.id).where(
            RepoCategory.category_id == category
        )
    if tag:
        stmt = stmt.join(RepoTag, RepoTag.repo_id == Repo.id).where(RepoTag.tag == tag)
    if builder:
        stmt = stmt.join(RepoBuilder, RepoBuilder.repo_id == Repo.id).where(
            RepoBuilder.login == builder
        )
    if ai_dev_skill:
        stmt = stmt.join(RepoAIDevSkill, RepoAIDevSkill.repo_id == Repo.id).where(
            RepoAIDevSkill.skill == ai_dev_skill
        )
    if topic:
        stmt = stmt.join(RepoTag, RepoTag.repo_id == Repo.id).where(RepoTag.tag == topic)
    if language:
        stmt = stmt.where(Repo.primary_language == language)
    if license:
        stmt = stmt.where(Repo.license_spdx == license)
    if min_stars:
        stmt = stmt.where(Repo.parent_stars >= min_stars)
    if sync_status:
        if sync_status == "up-to-date":
            stmt = stmt.where(Repo.fork_sync_state == "up-to-date")
        elif sync_status == "behind":
            stmt = stmt.where(Repo.behind_by > 0)
        elif sync_status == "ahead":
            stmt = stmt.where(Repo.ahead_by > 0)
        elif sync_status == "diverged":
            stmt = stmt.where(Repo.ahead_by > 0, Repo.behind_by > 0)
    if q:
        search = f"%{q}%"
        stmt = stmt.where(
            Repo.name.ilike(search) | Repo.description.ilike(search)
        )

    # Sorting
    if sort == "stars":
        stmt = stmt.order_by(Repo.parent_stars.desc().nulls_last())
    elif sort == "pushed_at":
        stmt = stmt.order_by(Repo.github_updated_at.desc().nulls_last())
    elif sort == "behind":
        stmt = stmt.order_by(Repo.behind_by.desc())
    elif sort == "name":
        stmt = stmt.order_by(Repo.name.asc())
    else:
        stmt = stmt.order_by(Repo.updated_at.desc())

    # Total count (before pagination)
    count_result = await db.execute(
        select(func.count()).select_from(stmt.distinct().subquery())
    )
    total = count_result.scalar_one()

    offset = (page - 1) * limit
    stmt = stmt.distinct().offset(offset).limit(limit)

    result = await db.execute(stmt)
    repos = result.scalars().all()

    response = {
        "repos": [_repo_to_summary(r).model_dump() for r in repos],
        "total": total,
        "page": page,
        "limit": limit,
    }
    await cache.set(cache_key, response)
    return response


@router.get("/repos/discover/cross-category")
@_limiter.limit("60/minute")
async def cross_category_repos(
    request: Request,
    categories: str = Query(..., description="Comma-separated category slugs"),
    limit: int = Query(10, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
):
    """
    Discover repos that span multiple categories (Tier 3 Item 14).
    Returns repos whose primary or secondary categories overlap with at least 2
    of the provided slugs, sorted by star count.
    """
    slugs = [s.strip() for s in categories.split(",") if s.strip()]
    if len(slugs) < 2:
        raise HTTPException(
            status_code=400,
            detail="Provide at least 2 comma-separated category slugs.",
        )

    # Build a filter: repo.primary_category IN slugs
    #   OR any element in repo.secondary_categories (JSONB array) is in slugs
    # Then count how many of the requested slugs each repo matches.
    #
    # Strategy: for each slug, check if it matches primary_category or is
    # contained in secondary_categories. Sum the matches and keep repos with >= 2.

    # We build per-slug match expressions and sum them.
    from sqlalchemy import case, literal_column
    from sqlalchemy.dialects.postgresql import JSONB as PG_JSONB

    match_exprs = []
    for slug in slugs:
        # primary_category exact match OR slug is in the JSONB array
        match_exprs.append(
            case(
                (
                    or_(
                        Repo.primary_category == slug,
                        Repo.secondary_categories.op("@>")(
                            cast(json.dumps([slug]), PG_JSONB)
                        ),
                    ),
                    1,
                ),
                else_=0,
            )
        )

    match_count = sum(match_exprs).label("match_count")

    stmt = (
        select(Repo, match_count)
        .where(Repo.is_private == False)  # noqa: E712
        .having(match_count >= 2)
        .group_by(Repo.id)
        .order_by(Repo.parent_stars.desc().nulls_last())
        .limit(limit)
    )

    result = await db.execute(stmt)
    rows = result.all()

    return {
        "repos": [
            {
                "name": repo.name,
                "description": repo.description,
                "primary_category": repo.primary_category,
                "secondary_categories": repo.secondary_categories,
                "parent_stars": repo.parent_stars,
                "categories_matched": int(cnt),
            }
            for repo, cnt in rows
        ],
        "total": len(rows),
        "categories_requested": slugs,
    }


@router.get("/repos/{name}/health")
@_limiter.limit("60/minute")
async def repo_health(
    request: Request,
    name: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Composite health score for a repo based on activity, quality, and security signals.
    Pure SQL — $0 cost.
    """
    stmt = (
        select(Repo)
        .where(func.lower(Repo.name) == func.lower(name))
        .where(Repo.is_private == False)  # noqa: E712
    )
    result = await db.execute(stmt)
    repo = result.scalar_one_or_none()
    if not repo:
        raise HTTPException(status_code=404, detail=f"Repo '{name}' not found")

    # --- Activity (30%) ---
    activity_raw = repo.activity_score or 0  # already 0-100

    # --- Stars momentum (15%) ---
    stars = repo.parent_stars or 0
    if stars <= 0:
        stars_raw = 0
    else:
        # log scale, cap at 50k
        stars_raw = min(100, int(math.log(min(stars, 50_000) + 1) / math.log(50_001) * 100))

    # --- Code quality (25%) ---
    quality_raw = 0
    has_tests = bool(repo.has_tests)
    has_ci = bool(repo.has_ci)
    has_license = bool(repo.license_spdx)
    if has_tests:
        quality_raw += 10
    if has_ci:
        quality_raw += 10
    if has_license:
        quality_raw += 5
    # Normalize to 0-100: max raw is 25, scale up by 4
    quality_raw = min(100, quality_raw * 4)

    # --- Freshness (20%) ---
    commits_30d = repo.commits_last_30_days or 0
    if commits_30d == 0:
        freshness_raw = 0
    elif commits_30d <= 5:
        freshness_raw = 50
    elif commits_30d <= 20:
        freshness_raw = 75
    else:
        freshness_raw = 100

    # --- Documentation (10%) ---
    has_readme = bool(repo.readme_summary)
    has_desc = bool(repo.description)
    doc_raw = 0
    if has_readme:
        doc_raw += 5
    if has_desc:
        doc_raw += 5
    # Normalize to 0-100: max raw is 10, scale up by 10
    doc_raw = min(100, doc_raw * 10)

    # Weighted composite
    health_score = int(
        activity_raw * 0.30
        + stars_raw * 0.15
        + quality_raw * 0.25
        + freshness_raw * 0.20
        + doc_raw * 0.10
    )

    return {
        "name": repo.name,
        "health_score": health_score,
        "breakdown": {
            "activity": activity_raw,
            "stars_momentum": stars_raw,
            "code_quality": quality_raw,
            "freshness": freshness_raw,
            "documentation": doc_raw,
        },
        "signals": {
            "has_tests": has_tests,
            "has_ci": has_ci,
            "license": repo.license_spdx,
            "commits_30d": commits_30d,
            "activity_score": activity_raw,
        },
    }


@router.get("/repos/{name}", response_model=RepoDetail)
async def get_repo(name: str, db: AsyncSession = Depends(get_db)) -> RepoDetail:
    cache_key = f"repos:detail:{name}"
    cached = await cache.get(cache_key)
    if cached:
        return RepoDetail(**cached)

    stmt = (
        select(Repo)
        .where(Repo.name == name, Repo.is_private == False)  # noqa: E712
        .options(
            selectinload(Repo.tags),
            selectinload(Repo.categories),
            selectinload(Repo.builders),
            selectinload(Repo.ai_dev_skills),
            selectinload(Repo.pm_skills),
            selectinload(Repo.languages),
            selectinload(Repo.commits),
            selectinload(Repo.taxonomy),
        )
    )
    result = await db.execute(stmt)
    repo = result.scalar_one_or_none()
    if not repo:
        raise HTTPException(status_code=404, detail=f"Repo '{name}' not found")

    detail = _repo_to_detail(repo)
    await cache.set(cache_key, detail.model_dump(), ttl=CACHE_TTL_REPO_DETAIL)
    return detail


@router.get("/repos/{owner}/{repo}", response_model=RepoDetail)
async def get_repo_by_owner(owner: str, repo: str, db: AsyncSession = Depends(get_db)) -> RepoDetail:
    """Get a single repo by owner/name."""
    stmt = (
        select(Repo)
        .where(Repo.owner == owner, Repo.name == repo, Repo.is_private == False)  # noqa: E712
        .options(
            selectinload(Repo.tags),
            selectinload(Repo.categories),
            selectinload(Repo.builders),
            selectinload(Repo.ai_dev_skills),
            selectinload(Repo.pm_skills),
            selectinload(Repo.languages),
            selectinload(Repo.commits),
            selectinload(Repo.taxonomy),
        )
    )
    result = await db.execute(stmt)
    found = result.scalar_one_or_none()
    if not found:
        raise HTTPException(status_code=404, detail=f"Repo '{owner}/{repo}' not found")
    return _repo_to_detail(found)
