"""Tests for rate-limit storage resolution — ensures we never hang on unreachable Redis.

Regression test for the 2026-04-05 production outage where a stale REDIS_URL
pointing to a decommissioned Memorystore IP caused every rate-limited request
to 504 because SlowAPI's Redis backend had no connect timeout.
"""
import time
from unittest.mock import patch

import app.rate_limit as rl


def _settings(url: str):
    return type("S", (), {"redis_url": url})()


def test_memory_fallback_when_redis_url_empty():
    """Empty REDIS_URL → memory://, no network call."""
    with patch.object(rl, "settings", _settings("")):
        assert rl._resolve_storage() == "memory://"


def test_memory_fallback_when_redis_unreachable():
    """Unreachable REDIS_URL → memory:// (probe fails fast, no hang)."""
    # 10.255.255.1 is a bogon — TCP connect should fail or timeout quickly.
    with patch.object(rl, "settings", _settings("redis://10.255.255.1:6379")):
        start = time.monotonic()
        result = rl._resolve_storage()
        elapsed = time.monotonic() - start
    assert result == "memory://", f"Expected memory fallback, got {result!r}"
    # 2s timeout + small slack. Must NOT hang for 60s like the prod outage.
    assert elapsed < 5.0, f"Probe took {elapsed:.1f}s — should be <5s"


def test_redis_url_used_when_reachable():
    """Reachable Redis → returns the redis:// URL unchanged."""
    with patch.object(rl, "settings", _settings("redis://reachable.local:6379")), \
         patch.object(rl, "_redis_reachable", return_value=True):
        assert rl._resolve_storage() == "redis://reachable.local:6379"


def test_bare_host_coerced_to_redis_scheme():
    """A bare host:port without scheme is treated as redis://host:port."""
    with patch.object(rl, "settings", _settings("host.local:6379")), \
         patch.object(rl, "_redis_reachable", return_value=True):
        assert rl._resolve_storage() == "redis://host.local:6379"
