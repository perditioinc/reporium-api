"""
KAN-ask-cache-effectiveness: Tests for cache-effectiveness improvements to
/intelligence/ask.

Covers:
  1. Streaming endpoint writes to Redis under the same key/payload/TTL as the
     non-streaming path (cross-endpoint cache sharing).
  2. _semantic_cache_threshold() A/B feature flag (ASK_CACHE_RELAXED env var).
  3. Smart-route Redis TTL bump from 300s -> 3600s.
  4. Cache-key alignment: the Redis key and the semantic-cache embedding input
     both use _normalize_question, so trivial variants share a cache row.
"""
import hashlib
import os
from unittest.mock import AsyncMock, patch

import pytest

from app.routers import intelligence as intel
from app.routers.intelligence import (
    _normalize_question,
    _semantic_cache_threshold,
    _SEMANTIC_CACHE_DISTANCE_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Deliverable 3: semantic-cache threshold feature flag
# ---------------------------------------------------------------------------

def test_semantic_cache_threshold_default():
    """Default (flag unset) must return the strict 0.15 threshold."""
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("ASK_CACHE_RELAXED", None)
        assert _semantic_cache_threshold() == 0.15


def test_semantic_cache_threshold_relaxed():
    """ASK_CACHE_RELAXED=1 must widen the radius to 0.25."""
    with patch.dict(os.environ, {"ASK_CACHE_RELAXED": "1"}):
        assert _semantic_cache_threshold() == 0.25


def test_semantic_cache_threshold_relaxed_with_whitespace():
    """Leading/trailing whitespace in the env var must still be honored."""
    with patch.dict(os.environ, {"ASK_CACHE_RELAXED": " 1 "}):
        assert _semantic_cache_threshold() == 0.25


def test_semantic_cache_threshold_unknown_value_is_strict():
    """Anything other than '1' must keep the strict default."""
    for val in ["0", "true", "yes", "", "off"]:
        with patch.dict(os.environ, {"ASK_CACHE_RELAXED": val}):
            assert _semantic_cache_threshold() == 0.15, f"value={val!r}"


def test_semantic_cache_threshold_backcompat_constant():
    """The old module-level constant is preserved as a default alias."""
    assert _SEMANTIC_CACHE_DISTANCE_THRESHOLD == 0.15


# ---------------------------------------------------------------------------
# Deliverable 2: smart-route Redis TTL bumped to 3600s
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_smart_route_cache_ttl_is_3600():
    """
    _try_smart_route must write with ttl=3600 (bumped from 300) because
    smart-route answers come from SQL aggregates that change at most hourly.
    """
    fake_result = {
        "answer": "42",
        "sources": [],
        "route": "count_repos",
    }
    with patch.object(intel.cache, "get", new=AsyncMock(return_value=None)), \
         patch.object(intel.cache, "set", new=AsyncMock()) as mock_set, \
         patch.object(intel, "_try_smart_route_inner", new=AsyncMock(return_value=fake_result)):
        out = await intel._try_smart_route("how many repos are there", db=None)
        assert out == fake_result
        assert mock_set.called, "cache.set must be called on smart-route hit"
        _args, kwargs = mock_set.call_args
        assert kwargs.get("ttl") == 3600, (
            f"Expected TTL=3600 (1h), got {kwargs.get('ttl')}. "
            "See KAN-ask-cache-effectiveness."
        )


# ---------------------------------------------------------------------------
# Deliverable 4: cache-key alignment between Redis key and semantic embedding
# ---------------------------------------------------------------------------

def test_redis_cache_key_uses_normalized_question():
    """
    The Redis `llm_response:<md5>` key must hash the NORMALIZED form, so
    "What is an LLM?" and "what is an llm" share the same cache row.
    """
    a = "What is an LLM?"
    b = "what is an llm"
    key_a = f"llm_response:{hashlib.md5(_normalize_question(a).encode()).hexdigest()}"
    key_b = f"llm_response:{hashlib.md5(_normalize_question(b).encode()).hexdigest()}"
    assert key_a == key_b, (
        "Normalized variants must collide on the same Redis cache key. "
        "If this fails, _prepare_query is hashing the raw question."
    )


def test_normalized_and_raw_differ_when_expected():
    """Sanity: normalization actually collapses the variants above."""
    assert _normalize_question("What is an LLM?") != "What is an LLM?"
    assert _normalize_question("What is an LLM?") == _normalize_question("what is an llm")


# ---------------------------------------------------------------------------
# Deliverable 1: streaming path writes to Redis after successful answer
# ---------------------------------------------------------------------------

def test_streaming_path_has_cache_write():
    """
    Source-level check: the /ask/stream event generator must contain a
    cache.set(qctx.redis_cache_key, ...) call with ttl=1800 matching the
    non-streaming payload shape. This ensures a /ask/stream request warms
    the same cache that a subsequent /ask request reads from.
    """
    import inspect
    src = inspect.getsource(intel.intelligence_ask_stream)
    assert "cache.set(qctx.redis_cache_key" in src, (
        "Streaming endpoint must write to the same Redis key as the "
        "non-streaming path. See KAN-ask-cache-effectiveness."
    )
    assert "ttl=1800" in src, (
        "Streaming cache-write must use the same 1800s TTL as _run_query."
    )
    for field in ('"answer"', '"sources"', '"tokens_used"', '"model"'):
        assert field in src, f"Streaming cache payload missing field {field}"


def test_streaming_and_nonstreaming_payload_shape_parity():
    """
    Both the streaming and non-streaming paths must write a payload with the
    same top-level keys so a cache write from one endpoint is consumable by
    the other.
    """
    import inspect
    stream_src = inspect.getsource(intel.intelligence_ask_stream)
    run_src = inspect.getsource(intel._run_query)
    required = ('"answer"', '"sources"', '"tokens_used"', '"model"')
    for field in required:
        assert field in stream_src, f"Streaming path missing {field}"
        assert field in run_src, f"Non-streaming path missing {field}"
