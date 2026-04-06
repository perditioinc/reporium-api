"""
GET /intelligence/compare — Side-by-side comparison of 2-5 repos.

Returns structured comparison data including metrics matrix, shared/unique tags,
and HN mention counts. Cached in Redis for 30 minutes, rate limited to 10/min.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.auth import require_app_token
from app.cache import cache
from app.database import get_db
from app.models.mention import RepoMention
from app.models.repo import Repo, RepoTag
from app.rate_limit import rate_limit_storage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/intelligence", tags=["Intelligence"])
_limiter = Limiter(key_func=get_remote_address, storage_uri=rate_limit_storage)

CACHE_TTL_COMPARE = 1800  # 30 minutes


def _extract_quality(repo: Repo) -> str | None:
    qs = repo.quality_signals
    if isinstance(qs, dict):
        return qs.get("quality")
    return None


def _extract_maturity(repo: Repo) -> str | None:
    qs = repo.quality_signals
    if isinstance(qs, dict):
        return qs.get("maturity")
    return None


def _extract_pros_cons(repo: Repo) -> dict | None:
    qs = repo.quality_signals
    if isinstance(qs, dict) and "pros_cons" in qs:
        return qs["pros_cons"]
    return None


def _repo_to_compare_dict(
    repo: Repo,
    tags: list[str],
    hn_count: int,
    top_hn: dict | None,
) -> dict:
    return {
        "name": repo.name,
        "owner": repo.owner,
        "stars": repo.parent_stars or repo.stargazers_count or 0,
        "primary_category": repo.primary_category,
        "language": repo.primary_language,
        "description": repo.description,
        "quality": _extract_quality(repo),
        "maturity": _extract_maturity(repo),
        "has_tests": repo.has_tests,
        "has_ci": repo.has_ci,
        "contributors_count": repo.contributors_count,
        "issue_close_rate": repo.issue_close_rate,
        "pr_merge_rate": repo.pr_merge_rate,
        "community_health_pct": repo.community_health_pct,
        "release_count": repo.release_count,
        "activity_score": repo.activity_score,
        "enriched_tags": tags,
        "pros_cons": _extract_pros_cons(repo),
        "hn_mentions_count": hn_count,
        "top_hn_mention": top_hn,
    }


@router.get("/compare")
@_limiter.limit("10/minute")
async def compare_repos(
    request: Request,
    repos: str = Query(
        ...,
        description="Comma-separated repo names (2-5 required)",
    ),
    db: AsyncSession = Depends(get_db),
    _token: None = Depends(require_app_token),
):
    """
    Compare 2-5 repos side-by-side with metrics matrix, shared/unique tags,
    and HN mention data.
    """
    repo_names = [n.strip() for n in repos.split(",") if n.strip()]

    if len(repo_names) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 repo names are required for comparison.",
        )
    if len(repo_names) > 5:
        raise HTTPException(
            status_code=400,
            detail="At most 5 repo names can be compared at once.",
        )

    # Cache key: sorted names for deterministic key regardless of input order
    cache_key = "compare:" + ",".join(sorted(n.lower() for n in repo_names))
    cached = await cache.get(cache_key)
    if cached:
        return cached

    # Fetch repos (case-insensitive match on name)
    stmt = (
        select(Repo)
        .where(
            func.lower(Repo.name).in_([n.lower() for n in repo_names]),
            Repo.is_private == False,  # noqa: E712
        )
        .options(selectinload(Repo.tags))
    )
    result = await db.execute(stmt)
    found_repos = result.scalars().all()

    # Check all requested repos were found
    found_names_lower = {r.name.lower() for r in found_repos}
    missing = [n for n in repo_names if n.lower() not in found_names_lower]
    if missing:
        raise HTTPException(
            status_code=404,
            detail=f"Repositories not found: {', '.join(missing)}",
        )

    # Build repo_id -> repo mapping
    repo_map = {r.id: r for r in found_repos}
    repo_ids = list(repo_map.keys())

    # Fetch HN mentions: count + top mention per repo
    hn_stmt = (
        select(RepoMention)
        .where(
            RepoMention.repo_id.in_(repo_ids),
            RepoMention.source == "hackernews",
        )
        .order_by(RepoMention.score.desc().nullslast())
    )
    hn_result = await db.execute(hn_stmt)
    hn_mentions = hn_result.scalars().all()

    hn_counts: dict[str, int] = {}
    hn_top: dict[str, dict | None] = {}
    for m in hn_mentions:
        repo = repo_map[m.repo_id]
        name = repo.name
        hn_counts[name] = hn_counts.get(name, 0) + 1
        if name not in hn_top:
            hn_top[name] = {
                "title": m.title,
                "url": m.url,
                "score": m.score,
                "comment_count": m.comment_count,
            }

    # Build per-repo tags and comparison dicts
    all_tag_sets: dict[str, set[str]] = {}
    repo_dicts = []
    for repo in found_repos:
        tags = sorted({t.tag for t in repo.tags})
        all_tag_sets[repo.name] = set(tags)
        repo_dicts.append(
            _repo_to_compare_dict(
                repo,
                tags=tags,
                hn_count=hn_counts.get(repo.name, 0),
                top_hn=hn_top.get(repo.name),
            )
        )

    # Comparison matrix
    comparison_matrix = {
        "stars": {r["name"]: r["stars"] for r in repo_dicts},
        "contributors": {r["name"]: r["contributors_count"] for r in repo_dicts},
        "issue_close_rate": {r["name"]: r["issue_close_rate"] for r in repo_dicts},
        "quality": {r["name"]: r["quality"] for r in repo_dicts},
        "maturity": {r["name"]: r["maturity"] for r in repo_dicts},
    }

    # Shared and unique tags
    if all_tag_sets:
        tag_sets = list(all_tag_sets.values())
        shared = tag_sets[0]
        for s in tag_sets[1:]:
            shared = shared & s
        shared_tags = sorted(shared)
    else:
        shared_tags = []

    unique_tags = {}
    for name, tags in all_tag_sets.items():
        others = set()
        for other_name, other_tags in all_tag_sets.items():
            if other_name != name:
                others |= other_tags
        unique_tags[name] = sorted(tags - others)

    response = {
        "repos": repo_dicts,
        "comparison_matrix": comparison_matrix,
        "shared_tags": shared_tags,
        "unique_tags": unique_tags,
    }

    await cache.set(cache_key, response, ttl=CACHE_TTL_COMPARE)
    return response
