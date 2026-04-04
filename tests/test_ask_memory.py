"""
KAN-158: Tests for conversational memory in POST /intelligence/ask.

Validates:
- session_id is optional (backward compatible — no session_id = stateless)
- History turns are loaded and prepended to Claude's messages
- Turn is saved after a successful answer
- History load failure is non-fatal (still answers)
- Turn save failure is non-fatal
- _load_session_turns returns oldest-first order
- _save_session_turn increments turn_number correctly

Uses AsyncMock / MagicMock patterns from test_intelligence_quality.py.
"""
import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch, call

import numpy as np
import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.main import app
from app.routers.intelligence import (
    _load_session_turns,
    _save_session_turn,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_SESSION_ID = str(uuid.uuid4())


def _make_claude_response(text: str = "Here is my answer.") -> MagicMock:
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    msg.usage = MagicMock(input_tokens=400, output_tokens=200)
    return msg


def _make_db_override_for_ask(rows=None, session_rows=None):
    """
    Yields a mock DB for /ask tests where _find_semantic_cache_hit is patched.

    With the semantic cache patched, actual DB execute() calls in _run_query are:
    1. Vector search → fetchall() = [] (empty — no embeddings in test DB)
    2. (edge query is skipped when top_for_answer is empty)
    3. _load_session_turns → fetchall() = session_rows (if session_id given)
    4. _save_session_turn max query → scalar() = -1 (if session_id given)
    5. _save_session_turn insert → (ignored)
    """
    mock_db = AsyncMock()

    empty_fetchall = MagicMock()
    empty_fetchall.fetchall.return_value = []

    if session_rows is not None:
        session_result = MagicMock()
        session_result.fetchall.return_value = session_rows
    else:
        session_result = MagicMock()
        session_result.fetchall.return_value = []

    # For _save_session_turn: scalar returns -1 (no prior turns) → next_turn=0
    scalar_result = MagicMock()
    scalar_result.scalar.return_value = -1

    call_idx = 0
    results = [empty_fetchall, session_result, scalar_result, MagicMock()]

    async def _execute(*args, **kwargs):
        nonlocal call_idx
        res = results[min(call_idx, len(results) - 1)]
        call_idx += 1
        return res

    mock_db.execute = _execute
    mock_db.commit = AsyncMock()

    async def _override():
        yield mock_db

    return mock_db, _override


# ---------------------------------------------------------------------------
# Unit tests for session helpers
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_load_session_turns_returns_oldest_first():
    """Rows fetched newest-first (DESC) are reversed to oldest-first for messages."""
    # Simulate 2 rows returned newest-first
    row1 = MagicMock()
    row1.question = "What is langchain?"
    row1.answer = "Langchain is an LLM framework."
    row2 = MagicMock()
    row2.question = "How do I use it?"
    row2.answer = "Import langchain and create a chain."

    mock_db = AsyncMock()
    result = MagicMock()
    result.fetchall.return_value = [row1, row2]  # newest-first from DB
    mock_db.execute = AsyncMock(return_value=result)

    turns = await _load_session_turns(_SESSION_ID, mock_db)

    # Should be reversed: row2 (oldest) first, then row1
    assert len(turns) == 4  # 2 turns × 2 messages each
    assert turns[0]["role"] == "user"
    assert turns[0]["content"] == "How do I use it?"   # row2 is oldest → first
    assert turns[1]["role"] == "assistant"
    assert turns[2]["role"] == "user"
    assert turns[2]["content"] == "What is langchain?"


@pytest.mark.asyncio
async def test_load_session_turns_returns_empty_on_db_error():
    """DB error during load is non-fatal — returns empty list."""
    mock_db = AsyncMock()
    mock_db.execute = AsyncMock(side_effect=Exception("DB unreachable"))

    turns = await _load_session_turns(_SESSION_ID, mock_db)
    assert turns == []


@pytest.mark.asyncio
async def test_save_session_turn_does_not_raise_on_error():
    """DB error during save is non-fatal — _save_session_turn uses its own session."""
    mock_db = AsyncMock()
    mock_db.execute = AsyncMock(side_effect=Exception("DB write failed"))
    mock_db.__aenter__ = AsyncMock(return_value=mock_db)
    mock_db.__aexit__ = AsyncMock(return_value=False)

    with patch("app.routers.intelligence.async_session_factory", return_value=mock_db):
        # Should not raise
        await _save_session_turn(_SESSION_ID, "question", "answer")


@pytest.mark.asyncio
async def test_save_session_turn_increments_turn_number():
    """Turn number is MAX(turn_number) + 1; SQL returns the raw max value."""
    mock_db = AsyncMock()

    # SQL: SELECT COALESCE(MAX(turn_number), -1) — returns 2 (existing max turn = 2)
    scalar_result = MagicMock()
    scalar_result.scalar.return_value = 2  # existing max turn = 2
    insert_result = MagicMock()
    mock_db.execute = AsyncMock(side_effect=[scalar_result, insert_result])
    mock_db.commit = AsyncMock()
    mock_db.__aenter__ = AsyncMock(return_value=mock_db)
    mock_db.__aexit__ = AsyncMock(return_value=False)

    with patch("app.routers.intelligence.async_session_factory", return_value=mock_db):
        await _save_session_turn(_SESSION_ID, "q", "a")

    # Python computes next_turn = max_turn + 1 = 2 + 1 = 3
    calls = mock_db.execute.call_args_list
    insert_call = calls[1]
    params = insert_call[0][1]  # positional arg dict
    assert params["turn"] == 3


# ---------------------------------------------------------------------------
# Integration tests via HTTP client
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ask_without_session_id_is_backward_compatible(client: AsyncClient):
    """Omitting session_id still works — no history loaded, no session saved."""
    mock_db, override = _make_db_override_for_ask()
    claude_resp = _make_claude_response()

    app.dependency_overrides[get_db] = override
    try:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = claude_resp
        with patch("app.routers.intelligence._get_client", return_value=mock_client), \
             patch("app.routers.intelligence.get_embedding_model") as mock_model, \
             patch("app.routers.intelligence._find_semantic_cache_hit", new=AsyncMock(return_value=None)), \
             patch("app.routers.intelligence._log_query", new=AsyncMock()), \
             patch("app.routers.intelligence._try_smart_route", new=AsyncMock(return_value=None)):
            mock_model.return_value.encode.return_value = np.zeros(384)
            resp = await client.post(
                "/intelligence/ask",
                json={"question": "What repos use LangChain?"},
            )
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "Here is my answer."
    assert data["cache_hit"] is False


@pytest.mark.asyncio
async def test_ask_with_session_id_prepends_history_to_claude(client: AsyncClient):
    """When session_id is provided, history is passed to Claude's messages array."""
    # Set up DB to return 1 prior turn
    prior_row = MagicMock()
    prior_row.question = "What is LangChain?"
    prior_row.answer = "LangChain is an LLM orchestration framework."

    mock_db, override = _make_db_override_for_ask(session_rows=[prior_row])
    claude_resp = _make_claude_response("Here is a follow-up answer.")

    captured_messages = []

    def _fake_create(**kwargs):
        captured_messages.extend(kwargs.get("messages", []))
        return claude_resp

    app.dependency_overrides[get_db] = override
    try:
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = _fake_create
        with patch("app.routers.intelligence._get_client", return_value=mock_client), \
             patch("app.routers.intelligence.get_embedding_model") as mock_model, \
             patch("app.routers.intelligence._find_semantic_cache_hit", new=AsyncMock(return_value=None)), \
             patch("app.routers.intelligence._log_query", new=AsyncMock()), \
             patch("app.routers.intelligence._save_session_turn", new=AsyncMock()):
            mock_model.return_value.encode.return_value = np.zeros(384)
            resp = await client.post(
                "/intelligence/ask",
                json={
                    "question": "How do I use it for RAG?",
                    "session_id": _SESSION_ID,
                },
            )
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 200

    # The messages array should have: prior user, prior assistant, current user
    # (prior rows are reversed: newest-first from DB → oldest-first in messages)
    # With 1 row returned (the prior turn), messages = [user, assistant, current_user]
    assert len(captured_messages) >= 3
    assert captured_messages[0]["role"] == "user"
    assert captured_messages[0]["content"] == "What is LangChain?"
    assert captured_messages[1]["role"] == "assistant"
    assert "LangChain" in captured_messages[1]["content"]
    assert captured_messages[-1]["role"] == "user"
    assert "RAG" in captured_messages[-1]["content"]


@pytest.mark.asyncio
async def test_ask_session_id_is_optional_field():
    """QueryRequest model accepts and defaults session_id to None."""
    from app.routers.intelligence import QueryRequest
    req = QueryRequest(question="What is the best RAG framework?")
    assert req.session_id is None

    req_with_session = QueryRequest(
        question="What is the best RAG framework?",
        session_id=_SESSION_ID,
    )
    assert req_with_session.session_id == _SESSION_ID
