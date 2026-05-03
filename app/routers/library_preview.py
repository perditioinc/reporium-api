"""
GET /library/preview — lean projection of the library for above-the-fold home rendering.

Per KAN-151 design memo (.audit/2026-05-02/library-preview-endpoint-design.md):
home page weighs 6.9 MB largely because of the 4-page /library/full?page_size=500
ladder pulling the full corpus when only ~60 cards render above the fold. This
endpoint returns the minimal field set RepoCardMinimal needs (~1.5 KB/repo)
plus top-5 enriched tags, with no aggregates / categories / tagMetrics. Projected
home payload: 5.2 MB → ~0.4 MB once KAN-152 frontend migrates the caller.

KAN-179 extension: optional `?include=` query parameter accepts a comma-separated
list of tokens (`stats`, `parent`, `quality`) to opt into additional per-repo
fields without falling back to /library/full. Default behaviour (no `?include`)
is unchanged — KAN-151 contracts hold (15 fields per repo). Enables /insights/
+ /trends/ pages to drop /library/full (1.46 MB) calls.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
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

# KAN-179: known `?include=` tokens. Each token unlocks a specific group of
# optional projection fields. Unknown tokens 400 to keep the contract crisp
# (so a typo can't silently degrade to default-projection).
_VALID_INCLUDE_TOKENS: frozenset[str] = frozenset({"stats", "parent", "quality"})


def _parse_include(raw: Optional[str]) -> frozenset[str]:
    """Validate + normalise the `?include=` query param.

    Returns a frozenset of recognised tokens. Empty / None input → empty set
    (default projection). Unknown tokens raise HTTPException 400 — they are
    NOT silently dropped, because that would make `?include=quality,hacker`
    indistinguishable from `?include=quality`.
    """
    if not raw:
        return frozenset()
    tokens = {t.strip().lower() for t in raw.split(",") if t.strip()}
    unknown = tokens - _VALID_INCLUDE_TOKENS
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown include token(s): {sorted(unknown)}. "
                f"Valid tokens: {sorted(_VALID_INCLUDE_TOKENS)}"
            ),
        )
    return frozenset(tokens)


# ---------------------------------------------------------------------------
# Pydantic response models — surfaced in /openapi.json
# ---------------------------------------------------------------------------


class ParentStats(BaseModel):
    """Subset of /library/full's parentStats — sufficient for /insights/ + /trends/."""
    owner: str = ""
    repo: str = ""
    stars: int = 0
    forks: int = 0
    isArchived: bool = False
    lastCommitDate: Optional[str] = None
    description: Optional[str] = None
    url: Optional[str] = None


class CommitStats(BaseModel):
    """Lean commit aggregate — counts only, no per-commit detail.

    /library/full also computes `today` from binned `recentCommits`; preview
    intentionally omits the per-commit list (the whole point of preview is
    avoiding the join), so `today` is sourced from the scalar columns alone
    and may be 0 if the DB scalar isn't populated. Consumers that need the
    daily granularity should still hit /library/full.
    """
    last7Days: int = 0
    last30Days: int = 0
    last90Days: int = 0


class PreviewRepo(BaseModel):
    """Minimal repo projection for above-the-fold home rendering.

    Strict subset of /library/full's EnrichedRepo. RepoCardMinimal consumers
    must not access fields outside this model on preview data.

    KAN-179: `commitStats`, `parentStats`, `upstreamCreatedAt`, `qualitySignals`
    are optional and only populated when the matching `?include=` token is
    supplied. Default-projection responses are unchanged.
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
    # KAN-179 extension fields — Optional, omitted from default response.
    commitStats: Optional[CommitStats] = None
    parentStats: Optional[ParentStats] = None
    upstreamCreatedAt: Optional[str] = None
    qualitySignals: Optional[dict] = None


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


def _build_preview_repo(row: dict, tags: list[str], include: frozenset[str]) -> dict:
    """Project a DB row + top-N tags into the PreviewRepo shape.

    `include` controls which optional KAN-179 extension fields are populated.
    Fields outside the include set are simply omitted from the dict (Pydantic
    serialises Optional=None, but we keep the wire compact by leaving them
    out entirely so default responses match KAN-151 byte-for-byte).
    """
    name = row.get("name") or ""
    owner = row.get("owner") or "perditioinc"
    full_name = row.get("full_name") or f"{owner}/{name}"
    is_fork = bool(row.get("is_fork"))

    # Match /library/full's stars/forks resolution: forks show parent counts;
    # built repos always show 0 forks (we don't track inbound forks of our own
    # repos — same convention as /library/full's _build_enriched_repo).
    if is_fork:
        stars = row.get("parent_stars") or 0
        forks = row.get("parent_forks") or 0
    else:
        stars = row.get("stargazers_count") or 0
        forks = 0

    out: dict = {
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

    # KAN-179 extension projections — keys ONLY present when token requested.
    if "stats" in include:
        out["commitStats"] = {
            "last7Days": row.get("commits_last_7_days") or 0,
            "last30Days": row.get("commits_last_30_days") or 0,
            "last90Days": row.get("commits_last_90_days") or 0,
        }

    if "parent" in include:
        forked_from = row.get("forked_from") or ""
        if forked_from:
            parts = forked_from.split("/", 1)
            parent_owner = parts[0] if len(parts) == 2 else ""
            parent_repo = parts[1] if len(parts) == 2 else forked_from
            out["parentStats"] = {
                "owner": parent_owner,
                "repo": parent_repo,
                "stars": row.get("parent_stars") or 0,
                "forks": row.get("parent_forks") or 0,
                "isArchived": bool(row.get("parent_is_archived")),
                "lastCommitDate": _iso(row.get("upstream_last_push_at")) or None,
                "description": row.get("description"),
                "url": f"https://github.com/{forked_from}",
            }
        else:
            out["parentStats"] = None
        out["upstreamCreatedAt"] = _iso(row.get("upstream_created_at")) or None

    if "quality" in include:
        out["qualitySignals"] = row.get("quality_signals")

    return out


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.get("/library/preview", response_model=PreviewResponse, response_model_exclude_none=True)
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
    include: Optional[str] = Query(
        default=None,
        description=(
            "Comma-separated extension tokens (KAN-179): `stats` (commitStats), "
            "`parent` (parentStats + upstreamCreatedAt), `quality` (qualitySignals). "
            "Default omits all extension fields."
        ),
    ),
):
    """Lean projection of the library for above-the-fold home rendering.

    Returns up to `limit` repos with the minimal field set for `RepoCardMinimal`,
    skipping all aggregates (stats / categories / tagMetrics / builderStats /
    aiDevSkillStats / pmSkillStats). Approximately 1.5 KB / repo wire size.

    See `.audit/2026-05-02/library-preview-endpoint-design.md` for the full
    design (response shape, sort options, projected Lighthouse delta).
    """
    # KAN-179: validate include first so a malformed `?include=` 400s before
    # we touch Redis or DB. _parse_include itself raises HTTPException(400).
    include_tokens = _parse_include(include)

    # KAN-170: switch to s-maxage (shared/CDN-only) so any future CDN/Cloud LB in
    # front of Cloud Run can edge-cache; browsers + non-CDN clients still hit
    # origin. stale-while-revalidate=60 keeps the total stale window (s-maxage +
    # swr = 6 min) within the 5-min Redis TTL band so ingestion-driven
    # invalidate_library_cache() flushes Redis before the CDN window expires.
    response.headers["Cache-Control"] = "public, s-maxage=300, stale-while-revalidate=60"

    cat_key = category if category else "*"
    # KAN-179: cache key includes a sorted-comma-joined include set so two
    # requests with different `?include=` values don't collide. `none` sentinel
    # for the empty set keeps the key visually distinct from a categories miss.
    include_key = ",".join(sorted(include_tokens)) if include_tokens else "none"
    redis_key = f"library:preview:{sort}:{limit}:{cat_key}:{include_key}"

    cached = await redis_cache.get(redis_key)
    if cached is not None:
        logger.info(
            "Redis hit /library/preview sort=%s limit=%d category=%s include=%s",
            sort, limit, cat_key, include_key,
        )
        return cached

    t0 = time.monotonic()
    order_by = _SORT_CLAUSES[sort]

    # Count for `totalRepos` — public corpus size, independent of limit/category.
    # This matches /library/full so the frontend can show "Showing N of M repos".
    total = (await db.execute(
        text("SELECT COUNT(*) FROM repos WHERE is_private = false")
    )).scalar() or 0

    # KAN-179: build the column projection list dynamically. Default columns
    # mirror the original KAN-151 SELECT exactly; extension columns appended
    # only when the matching token is in the include set. All columns are on
    # the `repos` table (no new joins) — verified against migration 036's
    # repos schema.
    base_columns = [
        "id", "name", "owner", "(owner || '/' || name) AS full_name", "description",
        "is_fork", "forked_from", "primary_language", "github_url",
        "parent_stars", "parent_forks", "parent_is_archived",
        "stargazers_count",
        "github_updated_at", "updated_at",
        "primary_category",
    ]
    extra_columns: list[str] = []
    if "stats" in include_tokens:
        extra_columns += [
            "commits_last_7_days", "commits_last_30_days", "commits_last_90_days",
        ]
    if "parent" in include_tokens:
        # parent_stars / parent_forks / parent_is_archived / forked_from / description
        # are already in base_columns (they're shared with the default projection).
        # Only `upstream_*` are net-new.
        extra_columns += ["upstream_last_push_at", "upstream_created_at"]
    if "quality" in include_tokens:
        extra_columns += ["quality_signals"]

    select_list = ", ".join(base_columns + extra_columns)

    # Main projection. is_private = false is the SQL invariant — same predicate
    # as /library/full and app.db_filters.PUBLIC_REPO_SQL_PREDICATE. Optional
    # primary_category equality narrows by frontend-canonical category name.
    sql_main = f"""
        SELECT {select_list}
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
        _build_preview_repo(row, tags_by_repo.get(str(row["id"]), []), include_tokens)
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
        "/library/preview built sort=%s limit=%d category=%s include=%s -> %d repos in %.1f ms",
        sort, limit, cat_key, include_key, len(repos), elapsed_ms,
    )

    await redis_cache.set(redis_key, body, ttl=CACHE_TTL)
    return body
