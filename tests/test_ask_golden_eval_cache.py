"""
KAN-176: Tests for the Redis-backed evaluator cache layer used by
`tests/test_ask_golden_numeric.py`.

The Ask Quality Gate runs on every push + PR (per KAN-146). Each fixture
entry triggers ~3-4 Anthropic calls inside `/intelligence/ask` (smart-route
probe, semantic-cache embedding, main answer, optional judge). Caching the
*evaluator response* by `(model, entry-yaml, question)` SHA in Redis with a
24h TTL means identical re-runs (same fixture set) skip the entire Anthropic
chain.

Cache surface tested:
  - Key shape: stable hash of `(model, entry, prompt)`.
  - Hit path: pre-populated Redis returns the cached payload without calling
    the underlying HTTP client.
  - Miss path: empty Redis triggers a real POST and writes the response back
    with TTL=86400.
  - Sensitivity: changing one character of the entry's question changes the
    cache key (no false positives across slightly-edited fixtures).
"""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.test_ask_golden_numeric import (
    _ASK_GATE_CACHE_PREFIX,
    _ASK_GATE_CACHE_TTL_SECONDS,
    _ask_with_cache,
    _cache_model_id,
    _CachedResponse,
    _evaluator_cache_key,
)


# ---------------------------------------------------------------------------
# Cache-key shape & sensitivity
# ---------------------------------------------------------------------------

def test_cache_key_has_expected_prefix_and_length():
    key = _evaluator_cache_key("router", {"q": "x"}, "hello")
    assert key.startswith(_ASK_GATE_CACHE_PREFIX), (
        f"Key must use the {_ASK_GATE_CACHE_PREFIX!r} namespace so "
        "`clear_prefix` can flush only eval-cache entries."
    )
    suffix = key[len(_ASK_GATE_CACHE_PREFIX):]
    assert len(suffix) == 16, "Truncated SHA-256 must be 16 hex chars"
    assert all(c in "0123456789abcdef" for c in suffix)


def test_cache_key_stable_across_calls():
    """Same inputs => same key (otherwise no cache would ever hit)."""
    entry = {"name": "demo", "owner": "perditioinc", "question": "what?"}
    k1 = _evaluator_cache_key("router", entry, "question text")
    k2 = _evaluator_cache_key("router", entry, "question text")
    assert k1 == k2


def test_cache_key_changes_on_entry_modification():
    """One-char change to the entry's question must change the cache key.

    Anti-stale-cache safety: if a fixture is edited the next run MUST re-issue
    the Anthropic call, otherwise the gate would silently grade the new
    fixture against the old answer.
    """
    base = {"name": "demo", "question": "what does redis cache?"}
    edited = {"name": "demo", "question": "what does redis cache!"}  # ? -> !
    assert _evaluator_cache_key("router", base, "p") != _evaluator_cache_key(
        "router", edited, "p"
    )


def test_cache_key_changes_on_prompt_modification():
    entry = {"name": "demo"}
    assert _evaluator_cache_key("router", entry, "alpha") != _evaluator_cache_key(
        "router", entry, "beta"
    )


def test_cache_key_changes_on_model_modification():
    """Bumping `ASK_GATE_CACHE_MODEL` must invalidate the cache."""
    entry = {"name": "demo"}
    assert _evaluator_cache_key("haiku-4-5", entry, "p") != _evaluator_cache_key(
        "haiku-4-6", entry, "p"
    )


def test_cache_key_dict_order_insensitive():
    """JSON serialisation must use sort_keys so dict-iteration order doesn't
    flake the cache key across Python versions."""
    a = {"x": 1, "y": 2}
    b = {"y": 2, "x": 1}
    assert _evaluator_cache_key("m", a, "p") == _evaluator_cache_key("m", b, "p")


def test_cache_model_id_default():
    """Default model id is 'router' (since the API picks Haiku/Sonnet itself)."""
    with patch.dict("os.environ", {}, clear=False):
        import os
        os.environ.pop("ASK_GATE_CACHE_MODEL", None)
        assert _cache_model_id() == "router"


def test_cache_model_id_override():
    with patch.dict("os.environ", {"ASK_GATE_CACHE_MODEL": "claude-haiku-4-6"}):
        assert _cache_model_id() == "claude-haiku-4-6"


def test_cache_model_id_blank_falls_back_to_router():
    """Empty / whitespace override must not produce an empty model component."""
    with patch.dict("os.environ", {"ASK_GATE_CACHE_MODEL": "   "}):
        assert _cache_model_id() == "router"


# ---------------------------------------------------------------------------
# _CachedResponse adapter
# ---------------------------------------------------------------------------

def test_cached_response_mimics_httpx_surface():
    """Caller reads `.status_code`, `.json()`, `.text`. Verify all three."""
    payload = {"answer": "42", "sources": [], "tokens_used": {"total": 7}}
    r = _CachedResponse(payload)
    assert r.status_code == 200
    assert r.json() == payload
    # `.text` is consulted only on the error path; it's still populated for
    # safety against future code paths.
    assert "42" in r.text


# ---------------------------------------------------------------------------
# Hit / miss behaviour
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_evaluator_cache_hit_skips_anthropic():
    """Pre-populated Redis must return the cached payload without POSTing."""
    cached_payload = {
        "answer": "cached",
        "sources": [],
        "tokens_used": {"total": 0},
    }
    fake_client = MagicMock()
    fake_client.post = AsyncMock(
        side_effect=AssertionError(
            "client.post must NOT be called on cache hit — the whole point "
            "of the cache is to skip the Anthropic chain."
        )
    )

    fake_redis = MagicMock()
    fake_redis.get = AsyncMock(return_value=cached_payload)
    fake_redis.set = AsyncMock()

    with patch("app.cache_redis.redis_cache", fake_redis):
        response = await _ask_with_cache(
            fake_client,
            "router",
            {"name": "fixture-a"},
            "ping?",
        )

    assert response.status_code == 200
    assert response.json() == cached_payload
    fake_client.post.assert_not_called()
    fake_redis.set.assert_not_called()  # No write on hit (TTL not refreshed)


@pytest.mark.asyncio
async def test_evaluator_cache_miss_calls_anthropic_then_writes():
    """Empty Redis: real POST happens, response is written back with 24h TTL."""
    api_payload = {"answer": "live", "sources": [], "tokens_used": {"total": 42}}
    real_response = MagicMock()
    real_response.status_code = 200
    real_response.json = MagicMock(return_value=api_payload)

    fake_client = MagicMock()
    fake_client.post = AsyncMock(return_value=real_response)

    fake_redis = MagicMock()
    fake_redis.get = AsyncMock(return_value=None)  # MISS
    fake_redis.set = AsyncMock()

    with patch("app.cache_redis.redis_cache", fake_redis):
        response = await _ask_with_cache(
            fake_client,
            "router",
            {"name": "fixture-b"},
            "ping?",
        )

    assert response is real_response
    fake_client.post.assert_awaited_once_with(
        "/intelligence/ask", json={"question": "ping?"}
    )
    fake_redis.set.assert_awaited_once()
    args, kwargs = fake_redis.set.call_args
    assert args[0].startswith(_ASK_GATE_CACHE_PREFIX)
    assert args[1] == api_payload
    assert kwargs.get("ttl") == _ASK_GATE_CACHE_TTL_SECONDS == 86400


@pytest.mark.asyncio
async def test_evaluator_cache_does_not_write_on_non_200():
    """5xx / 4xx responses must NOT be cached — would pin transient failures."""
    bad_response = MagicMock()
    bad_response.status_code = 502
    bad_response.json = MagicMock(return_value={"detail": "bad gateway"})

    fake_client = MagicMock()
    fake_client.post = AsyncMock(return_value=bad_response)

    fake_redis = MagicMock()
    fake_redis.get = AsyncMock(return_value=None)
    fake_redis.set = AsyncMock()

    with patch("app.cache_redis.redis_cache", fake_redis):
        response = await _ask_with_cache(
            fake_client,
            "router",
            {"name": "fixture-c"},
            "ping?",
        )

    assert response.status_code == 502
    fake_redis.set.assert_not_called()


@pytest.mark.asyncio
async def test_evaluator_cache_redis_unavailable_falls_through():
    """Redis import-fail / connection error must degrade to plain POST.

    This is the path taken by the slim CI workflow today (REDIS_URL="").
    Behaviour must be byte-identical to the pre-KAN-176 code: POST happens,
    response is returned, no exception leaks.
    """
    api_payload = {"answer": "no-redis", "sources": []}
    real_response = MagicMock()
    real_response.status_code = 200
    real_response.json = MagicMock(return_value=api_payload)

    fake_client = MagicMock()
    fake_client.post = AsyncMock(return_value=real_response)

    # Simulate `redis_cache.get` raising — wrapper must swallow and POST.
    fake_redis = MagicMock()
    fake_redis.get = AsyncMock(side_effect=ConnectionError("no redis here"))
    fake_redis.set = AsyncMock()

    with patch("app.cache_redis.redis_cache", fake_redis):
        response = await _ask_with_cache(
            fake_client,
            "router",
            {"name": "fixture-d"},
            "ping?",
        )

    assert response is real_response
    fake_client.post.assert_awaited_once()


@pytest.mark.asyncio
async def test_cache_write_failure_does_not_break_response():
    """A failed cache.set must not propagate — gate must keep running."""
    api_payload = {"answer": "ok", "sources": []}
    real_response = MagicMock()
    real_response.status_code = 200
    real_response.json = MagicMock(return_value=api_payload)

    fake_client = MagicMock()
    fake_client.post = AsyncMock(return_value=real_response)

    fake_redis = MagicMock()
    fake_redis.get = AsyncMock(return_value=None)
    fake_redis.set = AsyncMock(side_effect=RuntimeError("write blew up"))

    with patch("app.cache_redis.redis_cache", fake_redis):
        # Must not raise.
        response = await _ask_with_cache(
            fake_client,
            "router",
            {"name": "fixture-e"},
            "ping?",
        )

    assert response.status_code == 200
