import json
import logging
from typing import Any

import redis.asyncio as redis

from app.config import settings

logger = logging.getLogger(__name__)

CACHE_TTL_LIBRARY = 300       # 5 min
CACHE_TTL_REPO_DETAIL = 3600  # 1 hr
CACHE_TTL_TRENDS = 3600       # 1 hr
CACHE_TTL_GAPS = 3600         # 1 hr
CACHE_TTL_STATS = 300         # 5 min


class CacheManager:
    """
    Redis cache with graceful fallback.
    If Redis is not configured, all operations are no-ops.
    Never crashes the API if cache is unavailable.
    """

    def __init__(self) -> None:
        self._client: redis.Redis | None = None
        self._available: bool = False

    async def connect(self) -> None:
        if not settings.redis_url:
            logger.info("Redis not configured — cache disabled")
            return
        try:
            self._client = redis.from_url(settings.redis_url, decode_responses=True)
            await self._client.ping()
            self._available = True
            logger.info("Redis connected")
        except Exception as e:
            logger.warning(f"Redis unavailable: {e} — running without cache")
            self._client = None
            self._available = False

    async def disconnect(self) -> None:
        if self._client:
            await self._client.aclose()
            self._available = False

    async def get(self, key: str) -> dict | list | None:
        if not self._available or not self._client:
            return None
        try:
            value = await self._client.get(key)
            return json.loads(value) if value else None
        except Exception as e:
            logger.warning(f"Cache get error for '{key}': {e}")
            return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        if not self._available or not self._client:
            return
        try:
            effective_ttl = ttl if ttl is not None else settings.cache_ttl_seconds
            await self._client.setex(key, effective_ttl, json.dumps(value, default=str))
        except Exception as e:
            logger.warning(f"Cache set error for '{key}': {e}")

    async def invalidate(self, pattern: str) -> None:
        if not self._available or not self._client:
            return
        try:
            keys = await self._client.keys(pattern)
            if keys:
                await self._client.delete(*keys)
        except Exception as e:
            logger.warning(f"Cache invalidate error for '{pattern}': {e}")

    async def is_available(self) -> bool:
        return self._available


cache = CacheManager()
