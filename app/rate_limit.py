"""Shared rate-limiter storage configuration.

All routers should create their Limiter with ``storage_uri=rate_limit_storage``
so that counters are shared across Cloud Run instances when Redis is available.

Safety: if REDIS_URL is configured but the host is unreachable at process start,
we probe with a 2s connect timeout and fall back to in-memory storage. This
prevents SlowAPI's Redis backend from hanging every rate-limited request for
the full Cloud Run request timeout — a failure mode that took prod down on
2026-04-05 when a stale Memorystore IP was left in REDIS_URL.
"""
import logging
import socket
from urllib.parse import urlparse

from app.config import settings

logger = logging.getLogger(__name__)

_REDIS_CONNECT_TIMEOUT_S = 2.0


def _redis_reachable(url: str) -> bool:
    """TCP-probe the Redis host with a short timeout. True iff the port is open."""
    try:
        parsed = urlparse(url if "://" in url else f"redis://{url}")
        host = parsed.hostname
        port = parsed.port or 6379
        if not host:
            return False
        with socket.create_connection((host, port), timeout=_REDIS_CONNECT_TIMEOUT_S):
            return True
    except OSError as exc:
        logger.warning(
            "Rate-limiter Redis probe failed (%s); falling back to in-memory storage. "
            "Cross-instance counters will be per-pod until Redis is reachable.",
            exc,
        )
        return False


def _resolve_storage() -> str:
    raw = (settings.redis_url or "").strip()
    if not raw:
        return "memory://"
    url = raw if raw.startswith("redis") else f"redis://{raw}"
    if _redis_reachable(url):
        return url
    return "memory://"


rate_limit_storage: str = _resolve_storage()
