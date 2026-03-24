"""
Redis cache layer for reporium-api.

Reads REDIS_URL from the environment.  If the variable is absent (local dev,
preview deploys, etc.) every method is a no-op so the app continues to work
with only the in-memory cache — no crash, no special config needed.

Usage:
    from app.cache_redis import redis_cache

    value = await redis_cache.get("my:key")
    await redis_cache.set("my:key", {"foo": 1}, ttl=300)
    await redis_cache.delete("my:key")
    await redis_cache.clear_prefix("library:")
"""

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class RedisCache:
    """Async Redis wrapper with graceful degradation when REDIS_URL is unset."""

    def __init__(self) -> None:
        self._url: str | None = os.environ.get("REDIS_URL")
        self._client = None  # lazy-initialised on first use

    async def _get_client(self):
        """Return a connected async Redis client, or None if REDIS_URL is unset."""
        if self._url is None:
            return None
        if self._client is None:
            try:
                import redis.asyncio as aioredis

                self._client = aioredis.from_url(
                    self._url,
                    encoding="utf-8",
                    decode_responses=True,
                )
            except Exception:
                logger.exception("RedisCache: failed to create client — falling back to no-op")
                self._client = None
        return self._client

    async def get(self, key: str) -> Any | None:
        """
        Return the cached value for *key*, or None on miss / error.

        Values are stored as JSON strings; this method deserialises them
        automatically.  Returns None if Redis is unavailable.
        """
        client = await self._get_client()
        if client is None:
            return None
        try:
            raw = await client.get(key)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception:
            logger.warning("RedisCache.get(%s) failed — treating as cache miss", key, exc_info=True)
            return None

    async def set(self, key: str, value: Any, ttl: int = 300) -> None:
        """
        Serialise *value* as JSON and store it under *key* with the given TTL
        (seconds).  Silently swallows errors so a Redis outage never breaks a
        request.
        """
        client = await self._get_client()
        if client is None:
            return
        try:
            await client.set(key, json.dumps(value), ex=ttl)
        except Exception:
            logger.warning("RedisCache.set(%s) failed — skipping cache write", key, exc_info=True)

    async def delete(self, key: str) -> None:
        """Delete a single key.  No-op when Redis is unavailable."""
        client = await self._get_client()
        if client is None:
            return
        try:
            await client.delete(key)
        except Exception:
            logger.warning("RedisCache.delete(%s) failed", key, exc_info=True)

    async def clear_prefix(self, prefix: str) -> None:
        """
        Delete every key that starts with *prefix*.

        Uses SCAN to avoid blocking the server on large key-spaces.  No-op
        when Redis is unavailable.
        """
        client = await self._get_client()
        if client is None:
            return
        try:
            cursor = 0
            pattern = f"{prefix}*"
            while True:
                cursor, keys = await client.scan(cursor, match=pattern, count=100)
                if keys:
                    await client.delete(*keys)
                if cursor == 0:
                    break
            logger.debug("RedisCache.clear_prefix(%s) completed", prefix)
        except Exception:
            logger.warning("RedisCache.clear_prefix(%s) failed", prefix, exc_info=True)


# Module-level singleton — import and use directly.
redis_cache = RedisCache()
