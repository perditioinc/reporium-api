"""
KAN-ask-timeout: Tests for embedding timeout guard and streaming timeout
alignment in /intelligence/ask.

Covers:
  1. Embedding timeout: when embed_model.encode() hangs, _prepare_query returns
     a graceful early-exit response instead of blocking indefinitely.
  2. Streaming/non-streaming Claude call timeout alignment: both paths use the
     same _CLAUDE_TIMEOUT_S constant.
"""
import asyncio
import inspect
import re
import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from app.routers import intelligence as intel
from app.routers.intelligence import _CLAUDE_TIMEOUT_S


# ---------------------------------------------------------------------------
# 1. Embedding timeout guard
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_embedding_timeout_returns_graceful_response():
    """When embed_model.encode() takes longer than 5s, _prepare_query must
    return a QueryContext with model='embedding-timeout' and a user-friendly
    answer instead of raising or hanging."""

    def _slow_encode(text):
        """Simulate a hanging embedding model."""
        time.sleep(10)
        return np.zeros(384, dtype=np.float32)

    mock_model = MagicMock()
    mock_model.encode = _slow_encode

    mock_db = AsyncMock()

    with (
        patch.object(intel, "get_embedding_model", return_value=mock_model),
        patch.object(intel, "_try_smart_route", new_callable=AsyncMock, return_value=None),
        patch.object(intel, "cache", new_callable=MagicMock) as mock_cache,
    ):
        mock_cache.get = AsyncMock(return_value=None)

        start = time.monotonic()
        ctx = await intel._prepare_query(
            question="What is the best testing framework?",
            session_id=None,
            top_k=5,
            db=mock_db,
        )
        elapsed = time.monotonic() - start

        # Must complete well under 10s (the sleep duration) — proves timeout fired
        assert elapsed < 8, f"Took {elapsed:.1f}s, expected < 8s (timeout should fire at 5s)"

        # Must return a graceful early-exit, not raise
        assert ctx.model == "embedding-timeout"
        assert ctx.cache_result is not None
        assert "sorry" in ctx.cache_result["answer"].lower() or "try again" in ctx.cache_result["answer"].lower()
        assert ctx.query_embedding is None
        assert ctx.sources == []


@pytest.mark.asyncio
async def test_embedding_success_returns_normal_context():
    """When embedding completes quickly, _prepare_query should proceed normally
    (no early exit)."""

    fake_embedding = np.random.rand(384).astype(np.float32)

    mock_model = MagicMock()
    mock_model.encode = MagicMock(return_value=fake_embedding)

    mock_db = AsyncMock()

    with (
        patch.object(intel, "get_embedding_model", return_value=mock_model),
        patch.object(intel, "_try_smart_route", new_callable=AsyncMock, return_value=None),
        patch.object(intel, "cache", new_callable=MagicMock) as mock_cache,
        patch.object(intel, "_find_semantic_cache_hit", new_callable=AsyncMock, return_value=("cached answer", [], "test-model")),
    ):
        mock_cache.get = AsyncMock(return_value=None)

        ctx = await intel._prepare_query(
            question="What repos use Python?",
            session_id=None,
            top_k=5,
            db=mock_db,
        )

        # Should NOT be the timeout path
        assert ctx.model != "embedding-timeout"
        # Should have hit the semantic cache
        assert ctx.cache_result is not None
        assert ctx.cache_result["cache_source"] == "semantic"


# ---------------------------------------------------------------------------
# 2. Streaming / non-streaming timeout alignment
# ---------------------------------------------------------------------------

def test_streaming_queue_timeout_uses_claude_timeout_constant():
    """The streaming path's token_queue.get(timeout=...) must use the same
    _CLAUDE_TIMEOUT_S constant as the non-streaming path, not a hard-coded
    value."""
    source = inspect.getsource(intel)

    # Find the non-streaming timeout — should reference _CLAUDE_TIMEOUT_S
    assert "timeout=_CLAUDE_TIMEOUT_S" in source, (
        "Non-streaming path should use _CLAUDE_TIMEOUT_S"
    )

    # The streaming queue.get call should also use _CLAUDE_TIMEOUT_S, not a
    # hard-coded 35 (the old value)
    assert "token_queue.get(timeout=35)" not in source, (
        "Streaming path should not use hard-coded 35s timeout"
    )
    assert "token_queue.get(timeout=_CLAUDE_TIMEOUT_S)" in source, (
        "Streaming path should reference _CLAUDE_TIMEOUT_S for consistency"
    )


def test_claude_timeout_value_is_30():
    """Both paths should use 30s to match Cloud Run headroom budget."""
    assert _CLAUDE_TIMEOUT_S == 30, (
        f"_CLAUDE_TIMEOUT_S should be 30, got {_CLAUDE_TIMEOUT_S}"
    )
