"""Per-key rate limiting and budget enforcement for the API gateway.

Uses Redis for state when available (REDIS_URL set); falls back to in-memory
dicts for local development.  All public functions are async and safe to call
even when Redis is down -- they default to *allowing* the request so a Redis
outage never blocks legitimate traffic.

Environment variables:
    PER_KEY_RATE_LIMIT  -- max requests per minute per (api_key, route) pair.
                           Default: 30.
    PER_KEY_BUDGET_USD  -- daily spend cap per key in USD.  Default: 5.0.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import date

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# In-memory fallback stores (used when Redis is unavailable)
# --------------------------------------------------------------------------- #

_mem_rate: dict[str, list[float]] = {}   # key -> list of timestamps
_mem_budget: dict[str, float] = {}       # key -> accumulated spend today


def _per_key_rate_limit() -> int:
    return int(os.environ.get("PER_KEY_RATE_LIMIT", "30"))


def _per_key_budget_usd() -> float:
    return float(os.environ.get("PER_KEY_BUDGET_USD", "5.0"))


# --------------------------------------------------------------------------- #
# Rate limiting
# --------------------------------------------------------------------------- #

async def check_rate_limit(api_key: str, route: str) -> bool:
    """Check per-key rate limit.  Returns True if the request is allowed.

    Uses a sliding-window counter backed by Redis INCR + TTL.  If Redis is
    unavailable the function falls back to an in-memory list and still
    enforces the limit within the current process.
    """
    limit = _per_key_rate_limit()
    if limit <= 0:
        return True  # disabled

    redis_key = f"ratelimit:{api_key}:{route}"

    # --- Try Redis first ---
    try:
        from app.cache_redis import redis_cache
        client = await redis_cache._get_client()
        if client is not None:
            pipe = client.pipeline(transaction=True)
            pipe.incr(redis_key)
            pipe.expire(redis_key, 60)  # 60-second sliding window
            results = await pipe.execute()
            current: int = results[0]
            return current <= limit
    except Exception:
        logger.debug("check_rate_limit: Redis unavailable, using in-memory fallback")

    # --- In-memory fallback ---
    now = time.monotonic()
    window = _mem_rate.setdefault(redis_key, [])
    # Purge entries older than 60s
    window[:] = [t for t in window if now - t < 60]
    if len(window) >= limit:
        return False
    window.append(now)
    return True


# --------------------------------------------------------------------------- #
# Budget enforcement
# --------------------------------------------------------------------------- #

async def check_budget(api_key: str) -> tuple[bool, float]:
    """Check if *api_key* has remaining daily budget.

    Returns ``(allowed, remaining_usd)``.  ``allowed`` is False when the key
    has exhausted its daily budget.
    """
    budget = _per_key_budget_usd()
    if budget <= 0:
        return True, 0.0  # budgeting disabled

    today = date.today().isoformat()
    redis_key = f"budget:{api_key}:{today}"

    # --- Try Redis first ---
    try:
        from app.cache_redis import redis_cache
        client = await redis_cache._get_client()
        if client is not None:
            raw = await client.get(redis_key)
            spent = float(raw) if raw else 0.0
            remaining = max(budget - spent, 0.0)
            return spent < budget, remaining
    except Exception:
        logger.debug("check_budget: Redis unavailable, using in-memory fallback")

    # --- In-memory fallback ---
    spent = _mem_budget.get(redis_key, 0.0)
    remaining = max(budget - spent, 0.0)
    return spent < budget, remaining


async def record_spend(api_key: str, cost_usd: float) -> None:
    """Record token spend against the key's daily budget."""
    if cost_usd <= 0:
        return

    today = date.today().isoformat()
    redis_key = f"budget:{api_key}:{today}"

    # --- Try Redis first ---
    try:
        from app.cache_redis import redis_cache
        client = await redis_cache._get_client()
        if client is not None:
            pipe = client.pipeline(transaction=True)
            pipe.incrbyfloat(redis_key, cost_usd)
            pipe.expire(redis_key, 86_400)  # auto-expire after 24h
            await pipe.execute()
            return
    except Exception:
        logger.debug("record_spend: Redis unavailable, using in-memory fallback")

    # --- In-memory fallback ---
    _mem_budget[redis_key] = _mem_budget.get(redis_key, 0.0) + cost_usd
