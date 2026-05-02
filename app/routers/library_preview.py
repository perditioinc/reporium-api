"""
GET /library/preview — lean projection of the library for above-the-fold home rendering.

Per KAN-151 design memo (.audit/2026-05-02/library-preview-endpoint-design.md):
home page weighs 6.9 MB largely because of the 4-page /library/full?page_size=500
ladder pulling the full corpus when only ~60 cards render above the fold. This
endpoint returns the minimal field set RepoCardMinimal needs (~1.5 KB/repo)
plus top-5 enriched tags, with no aggregates / categories / tagMetrics. Projected
home payload: 5.2 MB → ~0.4 MB once KAN-152 frontend migrates the caller.

Ships dead — no caller until KAN-152. Intentional rollout safety.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Query, Request, Response
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.cache_redis import redis_cache
from app.database import get_db
from app.rate_limit import rate_limit_storage

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Library"])
_limiter = Limiter(key_func=get_remote_address, storage_uri=rate_limit_storage)

CACHE_TTL = 300  # 5 minutes — same window as /library/full

# Tags-per-repo cap. RepoCardMinimal renders at most a handful, so we keep
# this tight to bound per-repo payload.
_TOP_TAGS_PER_REPO = 5

# SQL ORDER BY clauses, keyed by the validated `sort` query param.
# `stars` mirrors /library/full's existing sort (uses ix_repos_stars after the
# COALESCE filter — sub-ms cost at 1.8K rows; revisit at 50K+).
_SORT_CLAUSES: dict[str, str] = {
    "stars": "COALESCE(parent_stars, stargazers_count, 0) DESC",
    "updated": "github_updated_at DESC NULLS LAST",
    "activity": "activity_score DESC NULLS LAST",
}


# ---------------------------------------------------------------------------
# Pydantic response models — surfaced in /openapi.json
# ---------------------------------------------------------------------------


class PreviewRepo(BaseModel):
    """Minimal repo projection for above-the-fold home rendering.

    Strict subset of /library/full's EnrichedRepo. RepoCardMinimal consumers
    must not access fields outside this model on preview data.
    """
    id: str
    name: str
    fullName: str
    description: Optional[str] = None
    isFork: bool = False
    forkedFrom: Optional[str] = None
    language: Optional[str] = None
    stars: int = 0
    forks: int = 0
    lastUpdated: str = ""
    primaryCategory: Optional[str] = None
    dbCategory: Optional[str] = None
    enrichedTags: list[str] = Field(default_factory=list)
    isArchived: bool = False
    url: str = ""


class PreviewResponse(BaseModel):
    generatedAt: str
    totalRepos: int
    limit: int
    sort: str
    category: Optional[str] = None
    repos: list[PreviewRepo]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iso(val) -> str:
    if val is None:
        return ""
    if isinstance(val, datetime):
        return val.isoformat()
    return str(val)


def _build_preview_repo(row: dict, tags: list[str]) -> dict:
    """Project a DB row + top-N tags into the PreviewRepo shape."""
    name = row.get("name") or ""
    owner = row.get("owner") or "perditioinc"
    full_name = row.get("full_name") or f"{owner}/{name}"
    is_fork = bool(row.get("is_fork"))

    # Match /library/full's stars/forks resolution: forks show parent counts.
    if is_fork:
        stars = row.get("parent_stars") or 0
        forks = row.get("parent_forks") or 0
    else:
        stars = row.get("stargazers_count") or 0
        forks = row.get("fork_count") or 0

    return {
        "id": str(row.get("id") or ""),
        "name": name,
        "fullName": full_name,
        "description": row.get("description"),
        "isFork": is_fork,
        "forkedFrom": row.get("forked_from"),
        "language": row.get("primary_language"),
        "stars": stars,
        "forks": forks,
        "lastUpdated": _iso(row.get("github_updated_at") or row.get("updated_at")),
        "primaryCategory": row.get("primary_category"),
        "dbCategory": row.get("primary_category"),
        "enrichedTags": tags,
        "isArchived": bool(row.get("parent_is_archived")),
        "url": row.get("github_url") or f"https://github.com/{owner}/{name}",
    }


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.get("/library/preview", response_model=PreviewResponse)
# 120/minute: matches the cache shape — preview is cheap (single repo SELECT
# + one junction read) and Redis-cached for 5 min, so we can afford a higher
# limit than /library/full's 60/minute. Frontend (KAN-152) will hit at most
# once per page load.
@_limiter.limit("120/minute")
async def library_preview(
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
    limit: int = Query(default=300, ge=1, le=500, description="Number of repos (max 500)"),
    sort: str = Query(default="stars", pattern="^(stars|updated|activity)$",
                       description="Sort key: stars | updated | activity"),
    category: Optional[str] = Query(default=None, description="Optional primary_category filter"),
):
    """Lean projection of the library for above-the-fold home rendering.

    Returns up to `limit` repos with the minimal field set for `RepoCardMinimal`,
    skipping all aggregates (stats / categories / tagMetrics / builderStats /
    aiDevSkillStats / pmSkillStats). Approximately 1.5 KB / repo wire size.

    See `.audit/2026-05-02/library-preview-endpoint-design.md` for the full
    design (response shape, sort options, projected Lighthouse delta).
    """
    response.headers["Cache-Control"] = "public, max-age=300, stale-while-revalidate=3600"

    cat_key = category if category else "*"
    redis_key = f"library:preview:{sort}:{limit}:{cat_key}"

    cached = await redis_cache.get(redis_key)
    if cached is not None:
        logger.info(
            "Redis hit /library/preview sort=%s limit=%d category=%s", sort, limit, cat_key
        )
        return cached

    t0 = time.monotonic()
    order_by = _SORT_CLAUSES[sort]

    # Count for `totalRepos` — public corpus size, independent of limit/category.
    # This matches /library/full so the frontend can show "Showing N of M repos".
    total = (await db.execute(
        text("SELECT COUNT(*) FROM repos WHERE is_private = false")
    )).scalar() or 0

    # Main projection. is_private = false is the SQL invariant — same predicate
    # as /library/full and app.db_filters.PUBLIC_REPO_SQL_PREDICATE. Optional
    # primary_category equality narrows by frontend-canonical category name.
    sql_main = f"""
        SELECT id, name, owner, (owner || '/' || name) AS full_name, description,
               is_fork, forked_from, primary_language, github_url,
               parent_stars, parent_forks, parent_is_archived,
               stargazers_count, fork_count,
               github_updated_at, updated_at,
               primary_category
        FROM repos
        WHERE is_private = false
          {"AND primary_category = :cat" if category else ""}
        ORDER BY {order_by}
        LIMIT :lim
    """
    params: dict = {"lim": limit}
    if category:
        params["cat"] = category

    result = await db.execute(text(sql_main), params)
    rows = result.fetchall()
    columns = list(result.keys())

    if not rows:
        body = {
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "totalRepos": total,
            "limit": limit,
            "sort": sort,
            "category": category,
            "repos": [],
        }
        await redis_cache.set(redis_key, body, ttl=CACHE_TTL)
        return body

    repo_dicts = [dict(zip(columns, row)) for row in rows]
    page_ids = [str(r["id"]) for r in repo_dicts]

    # Single junction read for top-N tags per repo. Same ANY(CAST(:ids AS uuid[]))
    # pattern as `_fetch_page_repos` to avoid asyncpg's `::uuid[]` parser quirk.
    # We over-fetch tags and trim per-repo in Python (cheap; ~1.8K rows in the
    # worst case at limit=500). Avoids per-repo subqueries / window functions.
    tag_result = await db.execute(text(
        "SELECT repo_id, tag FROM repo_tags "
        "WHERE repo_id = ANY(CAST(:ids AS uuid[]))"
    ), {"ids": page_ids})
    tags_by_repo: dict[str, list[str]] = defaultdict(list)
    for tag_row in tag_result.fetchall():
        rid = str(tag_row.repo_id)
        if len(tags_by_repo[rid]) < _TOP_TAGS_PER_REPO:
            tags_by_repo[rid].append(tag_row.tag)

    repos = [
        _build_preview_repo(row, tags_by_repo.get(str(row["id"]), []))
        for row in repo_dicts
    ]

    body = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "totalRepos": total,
        "limit": limit,
        "sort": sort,
        "category": category,
        "repos": repos,
    }

    elapsed_ms = (time.monotonic() - t0) * 1000
    logger.info(
        "/library/preview built sort=%s limit=%d category=%s -> %d repos in %.1f ms",
        sort, limit, cat_key, len(repos), elapsed_ms,
    )

    await redis_cache.set(redis_key, body, ttl=CACHE_TTL)
    return body
