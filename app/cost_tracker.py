"""In-memory daily LLM cost tracker with configurable cap."""
import os
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_DAILY_CAP = float(os.getenv("DAILY_LLM_COST_CAP", "5.0"))
_state = {"date": "", "total_usd": 0.0, "calls": 0}


def check_budget() -> bool:
    """Returns True if under daily cap."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if _state["date"] != today:
        _state["date"] = today
        _state["total_usd"] = 0.0
        _state["calls"] = 0
    return _state["total_usd"] < _DAILY_CAP


def record_cost(cost_usd: float, model: str = "unknown"):
    """Record an LLM call cost."""
    check_budget()
    _state["total_usd"] += cost_usd
    _state["calls"] += 1
    if _state["total_usd"] >= _DAILY_CAP * 0.8:
        logger.warning(
            "LLM daily budget at %.0f%% ($%.4f / $%.2f) after %d calls",
            (_state["total_usd"] / _DAILY_CAP) * 100,
            _state["total_usd"],
            _DAILY_CAP,
            _state["calls"],
        )


def get_usage() -> dict:
    """Return current usage for monitoring."""
    check_budget()
    return {
        "date": _state["date"],
        "total_usd": round(_state["total_usd"], 4),
        "calls": _state["calls"],
        "cap_usd": _DAILY_CAP,
        "remaining_usd": round(max(0, _DAILY_CAP - _state["total_usd"]), 4),
    }
