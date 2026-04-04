"""
KAN-155: Tests for POST /intelligence/nl-filter.

Validates request contract, caching behaviour, Haiku response parsing,
field validation/normalisation, and query_params construction — without
requiring a real Anthropic API key or DB connection.
"""
import json
from unittest.mock import MagicMock, patch

import pytest
from httpx import AsyncClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_haiku_response(data: dict) -> MagicMock:
    """Build a fake anthropic.messages.create response containing JSON."""
    msg = MagicMock()
    msg.content = [MagicMock(text=json.dumps(data))]
    return msg


_GOOD_PARSE = {
    "language": "python",
    "category": "rag-retrieval",
    "min_stars": 1000,
    "max_stars": None,
    "sort": "stars",
    "tags": ["rag", "langchain"],
    "quality": "high",
    "maturity": "production",
    "exclude_archived": True,
    "interpretation": "Python · RAG & Retrieval · 1,000+ stars",
}


# ---------------------------------------------------------------------------
# Contract: request validation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_nl_filter_requires_query(client: AsyncClient):
    resp = await client.post("/intelligence/nl-filter", json={})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_nl_filter_query_too_short(client: AsyncClient):
    resp = await client.post("/intelligence/nl-filter", json={"query": "ab"})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_nl_filter_query_too_long(client: AsyncClient):
    resp = await client.post("/intelligence/nl-filter", json={"query": "x" * 301})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Happy path: Haiku returns valid JSON
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_nl_filter_happy_path(client: AsyncClient):
    mock_resp = _mock_haiku_response(_GOOD_PARSE)
    breaker_mock = MagicMock(call=lambda fn: fn())

    with patch("app.routers.nl_filter.get_anthropic_key", return_value="sk-test"), \
         patch("app.routers.nl_filter.anthropic_breaker", breaker_mock), \
         patch("anthropic.Anthropic") as MockAnthropic, \
         patch("app.routers.nl_filter.cache.get", return_value=None), \
         patch("app.routers.nl_filter.cache.set"):
        MockAnthropic.return_value.messages.create.return_value = mock_resp
        resp = await client.post("/intelligence/nl-filter", json={"query": "Python RAG repos with 1000 stars"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["language"] == "python"
    assert data["category"] == "rag-retrieval"
    assert data["min_stars"] == 1000
    assert data["sort"] == "stars"
    assert data["exclude_archived"] is True
    assert "Python" in data["interpretation"]
    # query_params should be a usable URL fragment
    assert "language=python" in data["query_params"]
    assert "min_stars=1000" in data["query_params"]


# ---------------------------------------------------------------------------
# Normalisation: invalid category / sort / quality are nulled out
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_nl_filter_invalid_category_nulled(client: AsyncClient):
    bad = {**_GOOD_PARSE, "category": "not-a-real-category"}
    mock_resp = _mock_haiku_response(bad)
    breaker_mock = MagicMock(call=lambda fn: fn())

    with patch("app.routers.nl_filter.get_anthropic_key", return_value="sk-test"), \
         patch("app.routers.nl_filter.anthropic_breaker", breaker_mock), \
         patch("anthropic.Anthropic") as MockAnthropic, \
         patch("app.routers.nl_filter.cache.get", return_value=None), \
         patch("app.routers.nl_filter.cache.set"):
        MockAnthropic.return_value.messages.create.return_value = mock_resp
        resp = await client.post("/intelligence/nl-filter", json={"query": "Python RAG repos"})

    assert resp.status_code == 200
    assert resp.json()["category"] is None


@pytest.mark.asyncio
async def test_nl_filter_tags_truncated_to_five(client: AsyncClient):
    many_tags = {**_GOOD_PARSE, "tags": ["a", "b", "c", "d", "e", "f", "g"]}
    mock_resp = _mock_haiku_response(many_tags)
    breaker_mock = MagicMock(call=lambda fn: fn())

    with patch("app.routers.nl_filter.get_anthropic_key", return_value="sk-test"), \
         patch("app.routers.nl_filter.anthropic_breaker", breaker_mock), \
         patch("anthropic.Anthropic") as MockAnthropic, \
         patch("app.routers.nl_filter.cache.get", return_value=None), \
         patch("app.routers.nl_filter.cache.set"):
        MockAnthropic.return_value.messages.create.return_value = mock_resp
        resp = await client.post("/intelligence/nl-filter", json={"query": "repos with lots of tags"})

    assert resp.status_code == 200
    assert len(resp.json()["tags"]) <= 5


# ---------------------------------------------------------------------------
# Cache hit: Haiku is never called
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_nl_filter_cache_hit_skips_haiku(client: AsyncClient):
    cached = {**_GOOD_PARSE, "query_params": "language=python&min_stars=1000"}
    with patch("app.routers.nl_filter.cache.get", return_value=cached):
        with patch("app.routers.nl_filter.get_anthropic_key") as mock_key:
            resp = await client.post("/intelligence/nl-filter", json={"query": "Python RAG repos"})
            mock_key.assert_not_called()

    assert resp.status_code == 200
    assert resp.json()["language"] == "python"


# ---------------------------------------------------------------------------
# Error handling: Haiku returns unparseable JSON
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_nl_filter_json_parse_error_returns_502(client: AsyncClient):
    bad_msg = MagicMock()
    bad_msg.content = [MagicMock(text="this is not json at all")]
    breaker_mock = MagicMock(call=lambda fn: fn())

    with patch("app.routers.nl_filter.get_anthropic_key", return_value="sk-test"), \
         patch("app.routers.nl_filter.anthropic_breaker", breaker_mock), \
         patch("anthropic.Anthropic") as MockAnthropic, \
         patch("app.routers.nl_filter.cache.get", return_value=None):
        MockAnthropic.return_value.messages.create.return_value = bad_msg
        resp = await client.post("/intelligence/nl-filter", json={"query": "active Python repos"})

    assert resp.status_code == 502
