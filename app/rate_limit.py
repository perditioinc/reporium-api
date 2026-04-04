"""Shared rate-limiter storage configuration.

All routers should create their Limiter with ``storage_uri=rate_limit_storage``
so that counters are shared across Cloud Run instances when Redis is available.
"""
from app.config import settings

_redis_url = settings.redis_url or ""
rate_limit_storage: str = (
    _redis_url if _redis_url.startswith("redis")
    else f"redis://{_redis_url}" if _redis_url
    else "memory://"
)
