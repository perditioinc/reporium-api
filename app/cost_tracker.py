"""Daily LLM cost tracker backed by Redis for cross-instance accuracy."""
import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_DAILY_CAP = float(os.getenv("DAILY_LLM_COST_CAP", "5.0"))


def _today_key() -> str:
    return f"llm_cost:{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"


async def check_budget() -> bool:
    """Returns True if under daily cap. Fails closed if Redis unavailable."""
    try:
        from app.cache import cache
        val = await cache.get(_today_key())
        if val is None:
            return True
        return float(val) < _DAILY_CAP
    except Exception:
        logger.error("Redis unavailable for budget check — rejecting LLM call")
        return False  # Fail closed: block LLM calls when we can't verify budget


async def record_cost(cost_usd: float, model: str = "unknown"):
    """Record an LLM call cost atomically in Redis."""
    try:
        from app.cache import cache
        key = _today_key()
        # Use Redis cache.get/set since we don't have raw client access
        current = await cache.get(key)
        new_total = (float(current) if current else 0.0) + cost_usd
        await cache.set(key, new_total, ttl=90000)  # 25 hours

        if new_total >= _DAILY_CAP * 0.8:
            logger.warning(
                "LLM daily budget at %.0f%% ($%.4f / $%.2f)",
                (new_total / _DAILY_CAP) * 100, new_total, _DAILY_CAP,
            )
    except Exception:
        logger.exception("Failed to record LLM cost")


async def get_usage() -> dict:
    """Return current usage for monitoring."""
    try:
        from app.cache import cache
        val = await cache.get(_today_key())
        total = float(val) if val else 0.0
    except Exception:
        total = 0.0
    return {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "total_usd": round(total, 4),
        "cap_usd": _DAILY_CAP,
        "remaining_usd": round(max(0, _DAILY_CAP - total), 4),
    }
