"""
KAN-ask-output-caps: unit tests for cost-saving improvements to
POST /intelligence/ask.

Covers:
- ``_max_tokens_for`` returns per-model caps (512 / 768) and falls back to
  768 for unknown models.
- Low-similarity early-exit guard in ``_run_query``: when retrieval returns
  only sources with similarity < 0.40, the LLM is not called and the
  response is the deterministic "insufficient info" answer with model
  ``early-exit`` and zero token usage.
- Negative caching: when Claude returns a short refusal answer, the Redis
  cache.set is called with ``ttl=60`` and the payload carries
  ``"negative": True``.

Mocking pattern mirrors ``tests/test_intelligence.py`` and
``tests/test_ask_memory.py`` — we patch ``_prepare_query`` directly so the
tests don't depend on the embedding model or DB.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from app.routers.intelligence import (
    QueryContext,
    QueryRequest,
    _EARLY_EXIT_ANSWER,
    _MODEL_HAIKU,
    _MODEL_SONNET,
    _is_low_quality_answer,
    _max_tokens_for,
    _run_query,
)


# ---------------------------------------------------------------------------
# _max_tokens_for
# ---------------------------------------------------------------------------

def test_max_tokens_for_haiku():
    assert _max_tokens_for(_MODEL_HAIKU) == 512


def test_max_tokens_for_sonnet():
    assert _max_tokens_for(_MODEL_SONNET) == 768


def test_max_tokens_for_unknown_model_falls_back_to_768():
    assert _max_tokens_for("some-future-model-2099") == 768
    assert _max_tokens_for("") == 768


# ---------------------------------------------------------------------------
# _is_low_quality_answer (helper under test via negative-cache path)
# ---------------------------------------------------------------------------

def test_is_low_quality_answer_detects_short():
    assert _is_low_quality_answer("too short") is True


def test_is_low_quality_answer_detects_refusal_phrases():
    assert _is_low_quality_answer("I don't know the answer to that question.") is True
    assert _is_low_quality_answer("Sorry, I don't have enough information available to answer.") is True
    assert _is_low_quality_answer("I cannot answer this based on the retrieved repos.") is True


def test_is_low_quality_answer_accepts_substantive():
    good = (
        "LangChain is a Python framework for building LLM applications. It "
        "provides chains, agents, and integrations with dozens of vector stores."
    )
    assert _is_low_quality_answer(good) is False


# ---------------------------------------------------------------------------
# Early-exit guard: all retrieved sources below 0.40 similarity
# ---------------------------------------------------------------------------

def _make_ctx(sources: list[dict]) -> QueryContext:
    return QueryContext(
        sources=sources,
        context_text="<sources>\n(low-relevance)\n</sources>",
        model=_MODEL_HAIKU,
        session_history=[],
        cache_result=None,
        query_embedding=[0.0] * 384,
        route_label=None,
        embedding_candidates=len(sources),
        redis_cache_key="llm_response:test-early-exit",
    )


@pytest.mark.asyncio
async def test_run_query_early_exit_skips_claude_when_all_sources_below_threshold():
    """No source >= 0.40 similarity ⇒ return canned answer, never call Claude."""
    low_similarity_sources = [
        {
            "name": "unrelated-a",
            "owner": "someone",
            "forked_from": None,
            "description": "a repo about completely unrelated things",
            "stars": 10,
            "similarity": 0.31,
            "problem_solved": None,
            "integration_tags": [],
        },
        {
            "name": "unrelated-b",
            "owner": "someone",
            "forked_from": None,
            "description": "another unrelated repo",
            "stars": 5,
            "similarity": 0.22,
            "problem_solved": None,
            "integration_tags": [],
        },
    ]
    ctx = _make_ctx(low_similarity_sources)

    db = AsyncMock()
    mock_cache_set = AsyncMock()
    mock_log_query = AsyncMock()

    # _get_client must never be invoked — if it is, the test fails.
    with patch("app.routers.intelligence._prepare_query", new=AsyncMock(return_value=ctx)), \
         patch("app.routers.intelligence._get_client") as get_client, \
         patch("app.routers.intelligence.cache.set", new=mock_cache_set), \
         patch("app.routers.intelligence._log_query", new=mock_log_query), \
         patch("app.routers.intelligence.check_budget", new=AsyncMock(return_value=True)):
        response = await _run_query(
            QueryRequest(question="tell me about an unrelated thing please"),
            db,
            client_ip="203.0.113.99",
        )
        # give fire-and-forget tasks a chance to run
        import asyncio as _asyncio
        await _asyncio.sleep(0)

    assert response.model == "early-exit"
    assert response.tokens_used == {"input": 0, "output": 0, "total": 0}
    assert response.cache_hit is False
    assert response.answer == _EARLY_EXIT_ANSWER
    assert response.sources == []

    # The Claude client must never have been constructed.
    get_client.assert_not_called()

    # Redis cache should receive a short-TTL negative entry.
    assert mock_cache_set.await_count >= 1
    args, kwargs = mock_cache_set.await_args
    # cache.set(key, payload, ttl=...)
    assert args[0] == "llm_response:test-early-exit"
    assert kwargs.get("ttl") == 300 or (len(args) >= 3 and args[2] == 300)
    payload = args[1]
    assert payload["model"] == "early-exit"
    assert payload.get("negative") is True


# ---------------------------------------------------------------------------
# Negative caching: Claude returns a short refusal, expect ttl=60 + negative flag
# ---------------------------------------------------------------------------

def _make_ctx_with_relevant_sources() -> QueryContext:
    return QueryContext(
        sources=[
            {
                "name": "langchain",
                "owner": "langchain-ai",
                "forked_from": None,
                "description": "LLM orchestration framework",
                "stars": 80000,
                "similarity": 0.87,
                "problem_solved": "chain LLM calls",
                "integration_tags": ["rag"],
            }
        ],
        context_text="<sources>\nlangchain stuff\n</sources>",
        model=_MODEL_HAIKU,
        session_history=[],
        cache_result=None,
        query_embedding=[0.01] * 384,
        route_label=None,
        embedding_candidates=1,
        redis_cache_key="llm_response:test-negative-cache",
    )


@pytest.mark.asyncio
async def test_run_query_negative_caches_short_refusal_answer():
    """Claude returns 'I don't know' → cache.set called with ttl=60, negative flag."""
    ctx = _make_ctx_with_relevant_sources()

    # Build a minimal fake Anthropic client. _run_query calls it via an
    # executor, so a sync MagicMock is fine.
    fake_message = MagicMock()
    fake_message.content = [MagicMock(text="I don't know.")]  # short + refusal phrase
    fake_message.usage = MagicMock(input_tokens=120, output_tokens=5)
    mock_client = MagicMock()
    mock_client.messages.create.return_value = fake_message

    db = AsyncMock()
    mock_cache_set = AsyncMock()
    mock_log_query = AsyncMock()

    with patch("app.routers.intelligence._prepare_query", new=AsyncMock(return_value=ctx)), \
         patch("app.routers.intelligence._get_client", return_value=mock_client), \
         patch("app.routers.intelligence.cache.set", new=mock_cache_set), \
         patch("app.routers.intelligence._log_query", new=mock_log_query), \
         patch("app.routers.intelligence.check_budget", new=AsyncMock(return_value=True)), \
         patch("app.routers.intelligence.record_cost", new=AsyncMock()):
        response = await _run_query(
            QueryRequest(question="What is the meaning of life?"),
            db,
            client_ip="203.0.113.1",
        )
        import asyncio as _asyncio
        await _asyncio.sleep(0)

    assert response.answer == "I don't know."
    # Claude was invoked exactly once
    mock_client.messages.create.assert_called_once()

    # The create call should carry the new max_tokens + stop_sequences args.
    create_kwargs = mock_client.messages.create.call_args.kwargs
    assert create_kwargs["max_tokens"] == 512  # Haiku cap
    assert "</answer>" in create_kwargs["stop_sequences"]

    # cache.set must have been called with ttl=60 and negative=True.
    assert mock_cache_set.await_count >= 1
    # Find the call targeting the LLM-response key
    matching = [
        c for c in mock_cache_set.await_args_list
        if c.args and c.args[0] == "llm_response:test-negative-cache"
    ]
    assert matching, "expected a cache.set call for the redis_cache_key"
    args, kwargs = matching[-1]
    payload = args[1]
    ttl = kwargs.get("ttl") if "ttl" in kwargs else (args[2] if len(args) >= 3 else None)
    assert ttl == 60
    assert payload.get("negative") is True
    assert payload["answer"] == "I don't know."
