import hashlib
import json

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
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
)
from app.routers.library import _repo_to_summary
from app.schemas.repo import RepoDetail, RepoSummary

router = APIRouter()

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
    builder: str | None = None,
    ai_dev_skill: str | None = None,
    language: str | None = None,
    sync_status: str | None = None,
    sort: str = Query(default="updated"),
    q: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> dict:
    # Build cache key from params
    params = {
        "page": page, "limit": limit, "category": category, "tag": tag,
        "builder": builder, "ai_dev_skill": ai_dev_skill, "language": language,
        "sync_status": sync_status, "sort": sort, "q": q,
    }
    param_hash = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()
    cache_key = f"repos:list:{param_hash}"
    cached = await cache.get(cache_key)
    if cached:
        return cached

    stmt = (
        select(Repo)
        .options(
            selectinload(Repo.tags),
            selectinload(Repo.categories),
            selectinload(Repo.builders),
            selectinload(Repo.ai_dev_skills),
            selectinload(Repo.pm_skills),
            selectinload(Repo.languages),
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
    if language:
        stmt = stmt.where(Repo.primary_language == language)
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
    elif sort == "behind":
        stmt = stmt.order_by(Repo.behind_by.desc())
    elif sort == "name":
        stmt = stmt.order_by(Repo.name.asc())
    else:
        stmt = stmt.order_by(Repo.updated_at.desc())

    # Total count (without pagination)
    count_stmt = select(stmt.subquery().c.id)

    offset = (page - 1) * limit
    stmt = stmt.distinct().offset(offset).limit(limit)

    result = await db.execute(stmt)
    repos = result.scalars().all()

    response = {
        "repos": [_repo_to_summary(r).model_dump() for r in repos],
        "page": page,
        "limit": limit,
    }
    await cache.set(cache_key, response)
    return response


@router.get("/repos/{name}", response_model=RepoDetail)
async def get_repo(name: str, db: AsyncSession = Depends(get_db)) -> RepoDetail:
    cache_key = f"repos:detail:{name}"
    cached = await cache.get(cache_key)
    if cached:
        return RepoDetail(**cached)

    stmt = (
        select(Repo)
        .where(Repo.name == name)
        .options(
            selectinload(Repo.tags),
            selectinload(Repo.categories),
            selectinload(Repo.builders),
            selectinload(Repo.ai_dev_skills),
            selectinload(Repo.pm_skills),
            selectinload(Repo.languages),
            selectinload(Repo.commits),
        )
    )
    result = await db.execute(stmt)
    repo = result.scalar_one_or_none()
    if not repo:
        raise HTTPException(status_code=404, detail=f"Repo '{name}' not found")

    detail = _repo_to_detail(repo)
    await cache.set(cache_key, detail.model_dump(), ttl=CACHE_TTL_REPO_DETAIL)
    return detail
