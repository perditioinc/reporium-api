"""Tests for KAN-165: GET /admin/asks and PATCH /admin/asks/{ask_id}.

These are integration tests that run against the in-memory test DB (SQLite
via conftest.py's _setup_db fixture which creates tables from ORM metadata).
"""

import pytest
from httpx import AsyncClient
from sqlalchemy import text

from app.database import async_session_factory
from app.models.query_log import QueryLog
from tests.conftest import AUTH_HEADERS, TEST_API_KEY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _insert_query_log(question: str = "test question", **kwargs) -> int:
    """Insert a QueryLog row directly and return its id."""
    async with async_session_factory() as session:
        row = QueryLog(question=question, **kwargs)
        session.add(row)
        await session.commit()
        await session.refresh(row)
        return row.id


# ---------------------------------------------------------------------------
# Auth guards
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_asks_requires_api_key(client: AsyncClient):
    response = await client.get("/admin/asks")
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_patch_ask_requires_api_key(client: AsyncClient):
    response = await client.patch("/admin/asks/1", json={"jira_status": "open"})
    assert response.status_code == 403


# ---------------------------------------------------------------------------
# GET /admin/asks — basic shape
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_asks_returns_correct_shape(client: AsyncClient):
    response = await client.get("/admin/asks", headers=AUTH_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "asks" in data
    assert isinstance(data["total"], int)
    assert isinstance(data["asks"], list)


@pytest.mark.asyncio
async def test_list_asks_row_contains_expected_fields(client: AsyncClient):
    ask_id = await _insert_query_log(
        question="what is rag?",
        model="haiku",
        tokens_prompt=100,
        tokens_completion=200,
        cost_cents=1,
        latency_ms=350,
    )
    response = await client.get("/admin/asks", headers=AUTH_HEADERS)
    assert response.status_code == 200
    data = response.json()
    # Find our row
    matching = [a for a in data["asks"] if a["id"] == ask_id]
    assert len(matching) == 1
    row = matching[0]
    expected_keys = {
        "id", "created_at", "query", "model",
        "input_tokens", "output_tokens", "cost_cents",
        "jira_ticket_key", "jira_status", "action_taken",
        "sentiment", "latency_ms", "user_ip_hash",
    }
    assert expected_keys == set(row.keys())
    assert row["query"] == "what is rag?"
    assert row["model"] == "haiku"
    assert row["input_tokens"] == 100
    assert row["output_tokens"] == 200
    assert row["cost_cents"] == 1
    assert row["latency_ms"] == 350
    assert row["jira_ticket_key"] is None
    assert row["jira_status"] is None


@pytest.mark.asyncio
async def test_list_asks_limit_enforced(client: AsyncClient):
    # Insert 3 rows
    for i in range(3):
        await _insert_query_log(question=f"limit test question {i}")

    response = await client.get("/admin/asks?limit=2", headers=AUTH_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert len(data["asks"]) <= 2


@pytest.mark.asyncio
async def test_list_asks_limit_max_500(client: AsyncClient):
    response = await client.get("/admin/asks?limit=501", headers=AUTH_HEADERS)
    assert response.status_code == 422  # FastAPI validation error


@pytest.mark.asyncio
async def test_list_asks_filter_jira_status(client: AsyncClient):
    await _insert_query_log(question="open ticket q", jira_ticket_key="KAN-900", jira_status="open")
    await _insert_query_log(question="done ticket q", jira_ticket_key="KAN-901", jira_status="done")

    response = await client.get("/admin/asks?jira_status=open", headers=AUTH_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert all(a["jira_status"] == "open" for a in data["asks"])


@pytest.mark.asyncio
async def test_list_asks_filter_has_ticket_true(client: AsyncClient):
    await _insert_query_log(question="has ticket q", jira_ticket_key="KAN-902", jira_status="open")
    await _insert_query_log(question="no ticket q")

    response = await client.get("/admin/asks?has_ticket=true", headers=AUTH_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert all(a["jira_ticket_key"] is not None for a in data["asks"])


@pytest.mark.asyncio
async def test_list_asks_filter_has_ticket_false(client: AsyncClient):
    response = await client.get("/admin/asks?has_ticket=false", headers=AUTH_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert all(a["jira_ticket_key"] is None for a in data["asks"])


# ---------------------------------------------------------------------------
# PATCH /admin/asks/{ask_id}
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_patch_ask_404_when_not_found(client: AsyncClient):
    response = await client.patch(
        "/admin/asks/999999999",
        json={"jira_status": "open"},
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_patch_ask_sets_jira_ticket_key(client: AsyncClient):
    ask_id = await _insert_query_log(question="admin patch test")

    response = await client.patch(
        f"/admin/asks/{ask_id}",
        json={"jira_ticket_key": "KAN-165", "jira_status": "open"},
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["jira_ticket_key"] == "KAN-165"
    assert data["jira_status"] == "open"
    assert data["id"] == ask_id


@pytest.mark.asyncio
async def test_patch_ask_partial_update(client: AsyncClient):
    ask_id = await _insert_query_log(
        question="partial patch test",
        jira_ticket_key="KAN-200",
        jira_status="open",
    )

    # Only update jira_status — other fields should be unchanged
    response = await client.patch(
        f"/admin/asks/{ask_id}",
        json={"jira_status": "done"},
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["jira_status"] == "done"
    assert data["jira_ticket_key"] == "KAN-200"  # unchanged


@pytest.mark.asyncio
async def test_patch_ask_sets_action_taken_and_sentiment(client: AsyncClient):
    ask_id = await _insert_query_log(question="sentiment test")

    response = await client.patch(
        f"/admin/asks/{ask_id}",
        json={"action_taken": "Promoted to KAN-999", "sentiment": "positive"},
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["action_taken"] == "Promoted to KAN-999"
    assert data["sentiment"] == "positive"


@pytest.mark.asyncio
async def test_patch_ask_empty_body_is_noop(client: AsyncClient):
    ask_id = await _insert_query_log(
        question="noop patch test",
        jira_ticket_key="KAN-123",
        jira_status="in_progress",
    )

    response = await client.patch(
        f"/admin/asks/{ask_id}",
        json={},
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["jira_ticket_key"] == "KAN-123"
    assert data["jira_status"] == "in_progress"
