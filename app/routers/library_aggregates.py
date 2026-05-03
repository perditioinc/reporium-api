"""
GET /library/aggregates — KAN-188.

Lean aggregate-only sibling of /library/full. Returns the aggregate fields
(stats, gapAnalysis, tagMetrics, categories, builderStats, aiDevSkillStats,
pmSkillStats) WITHOUT the per-repo array.

Per the 4h perf audit P2: /library/full ships 1.46 MB warm; the aggregates
constitute ~30-40% of that. Lean callers (StatsBar, MetricsSidebar,
LibraryInsightsWidget, etc.) that don't need the per-repo array can hit this
endpoint instead, dropping ~50-300 KiB-per-request to ~1.46 MB-per-request.

Backwards-compat: /library/full's response shape is UNCHANGED. This endpoint
is purely additive. Frontend migration of aggregate consumers is a separate
follow-up ticket.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Request, Response
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.cache_redis import redis_cache
from app.database import get_db
from app.rate_limit import rate_limit_storage
from app.routers.library_aggregates_helpers import build_gap_analysis
from app.routers.library_full import _fetch_aggregates

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Library"])
_limiter = Limiter(key_func=get_remote_address, storage_uri=rate_limit_storage)

# Cache TTL matches /library/full and /library/preview — 5 minutes.
# This aligns the staleness window across all three endpoints so a single
# invalidate_library_cache() call drops them in lockstep.
CACHE_TTL = 300

# Independent Redis key from /library/full so cold-loading aggregates doesn't
# warm /library/full's per-page envelopes (and vice versa). Both keys live
# under the `library:` prefix so the existing prefix-sweep in
# invalidate_library_cache() catches both on backfill.
_REDIS_KEY = "library:aggregates:v1"


@router.get("/library/aggregates", response_model=dict)
# 60/minute: aggregates are computed across the FULL corpus (one query per
# page, ~4 pages at 500 repos each), so we keep the limit conservative even
# though Redis serves most calls. This matches /library/full's 60/minute and
# is lower than /library/preview (120/minute) which is a single-page projection.
@_limiter.limit("60/minute")
async def library_aggregates(
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
):
    """Aggregate-only response — no per-repo array.

    Returns:
        - generatedAt: ISO timestamp
        - totalRepos: count of public repos (mirrors /library/full)
        - stats, gapAnalysis, tagMetrics, categories, builderStats,
          aiDevSkillStats, pmSkillStats: same shapes as /library/full

    Cache hierarchy (mirrors /library/full):
      1. Redis under `library:aggregates:v1` (5-min TTL, cluster-shared)
      2. _fetch_aggregates() in-memory cache (5-min TTL, per-instance)
      3. Cold path: load all public repos via _fetch_page_repos pagination,
         then run the aggregate builders.

    Cache-Control: `public, s-maxage=300, stale-while-revalidate=60` — same
    KAN-170 pattern as /library/preview so a future CDN/Cloud LB in front of
    Cloud Run can edge-cache.
    """
    response.headers["Cache-Control"] = "public, s-maxage=300, stale-while-revalidate=60"

    cached = await redis_cache.get(_REDIS_KEY)
    if cached is not None:
        logger.info("Redis hit /library/aggregates")
        return cached

    t0 = time.monotonic()

    # totalRepos mirrors the count /library/full exposes — public corpus only.
    # The same WHERE clause is enforced by app.db_filters.PUBLIC_REPO_SQL_PREDICATE.
    total = (await db.execute(
        text("SELECT COUNT(*) FROM repos WHERE is_private = false")
    )).scalar() or 0

    # _fetch_aggregates loads all public repos through the same paginated path
    # /library/full uses, then runs the aggregate builders. Result is cached
    # in-memory so back-to-back calls (e.g. /library/full + /library/aggregates
    # within the same TTL window) share the work.
    aggregates = await _fetch_aggregates(db)

    # Build the gap analysis from the same enriched-repo population. Currently
    # a stub — empty `gaps` array — but the helper signature lets the future
    # gap-analysis ticket land without re-touching this endpoint.
    # Note: _fetch_aggregates discards the per-repo list; gapAnalysis only
    # needs the count + timestamp shape today, so passing an empty list is
    # equivalent to today's behaviour. When gapAnalysis goes live, this will
    # need to plumb the enriched repo list through.
    body = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "totalRepos": total,
        "gapAnalysis": build_gap_analysis([]),
        **aggregates,
    }

    await redis_cache.set(_REDIS_KEY, body, ttl=CACHE_TTL)

    elapsed_ms = (time.monotonic() - t0) * 1000
    logger.info(
        "/library/aggregates built totalRepos=%d in %.1f ms", total, elapsed_ms,
    )

    return body
