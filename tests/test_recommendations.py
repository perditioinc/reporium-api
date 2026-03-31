"""
KAN-156: Tests for GET /intelligence/similar/{name} and GET /intelligence/recommended.

Uses app.dependency_overrides[get_db] for DB injection and AsyncMock for
async cache methods — same pattern as test_intelligence_quality.py.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient

from app.database import get_db
from app.main import app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_row(name="langchain", similarity=0.85):
    row = MagicMock()
    row.name = name
    row.owner = "perditioinc"
    row.description = f"The {name} framework"
    row.primary_language = "Python"
    row.primary_category = "rag-retrieval"
    row.stars = 1000
    row.readme_summary = f"A helpful {name} summary."
    row.similarity = similarity
    return row


def _override_db_with_rows(rows):
    """Yield a mock db session whose execute() returns rows via fetchall()."""
    mock_db = AsyncMock()
    result = MagicMock()
    result.fetchall.return_value = rows
    mock_db.execute = AsyncMock(return_value=result)

    async def _override():
        yield mock_db

    return mock_db, _override


def _override_db_multi(call_results: list):
    """Successive execute() calls return successive row lists."""
    call_idx = 0

    async def _execute(*args, **kwargs):
        nonlocal call_idx
        res = MagicMock()
        res.fetchall.return_value = call_results[min(call_idx, len(call_results) - 1)]
        call_idx += 1
        return res

    mock_db = AsyncMock()
    mock_db.execute = _execute

    async def _override():
        yield mock_db

    return mock_db, _override


# ---------------------------------------------------------------------------
# GET /intelligence/similar/{name}
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_similar_repos_returns_results(client: AsyncClient):
    rows = [_make_row("llama-index", 0.91), _make_row("haystack", 0.82)]
    _, override = _override_db_with_rows(rows)

    app.dependency_overrides[get_db] = override
    try:
        with patch("app.routers.recommendations.cache.get", new=AsyncMock(return_value=None)), \
             patch("app.routers.recommendations.cache.set", new=AsyncMock()):
            resp = await client.get("/intelligence/similar/langchain")
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 200
    data = resp.json()
    assert data["source_repo"] == "langchain"
    assert len(data["similar"]) == 2
    assert data["similar"][0]["similarity"] == 0.91
    assert data["total"] == 2


@pytest.mark.asyncio
async def test_similar_repos_empty_when_no_embedding(client: AsyncClient):
    """Repo exists but has no embedding — returns empty list, not 404."""
    empty_result = MagicMock()
    empty_result.fetchall.return_value = []
    exists_result = MagicMock()
    exists_result.fetchone.return_value = MagicMock()  # repo found

    mock_db = AsyncMock()
    mock_db.execute = AsyncMock(side_effect=[empty_result, exists_result])

    async def _override():
        yield mock_db

    app.dependency_overrides[get_db] = _override
    try:
        with patch("app.routers.recommendations.cache.get", new=AsyncMock(return_value=None)), \
             patch("app.routers.recommendations.cache.set", new=AsyncMock()):
            resp = await client.get("/intelligence/similar/langchain")
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 200
    assert resp.json()["total"] == 0
    assert resp.json()["similar"] == []


@pytest.mark.asyncio
async def test_similar_repos_404_for_unknown(client: AsyncClient):
    empty_result = MagicMock()
    empty_result.fetchall.return_value = []
    no_repo = MagicMock()
    no_repo.fetchone.return_value = None

    mock_db = AsyncMock()
    mock_db.execute = AsyncMock(side_effect=[empty_result, no_repo])

    async def _override():
        yield mock_db

    app.dependency_overrides[get_db] = _override
    try:
        with patch("app.routers.recommendations.cache.get", new=AsyncMock(return_value=None)):
            resp = await client.get("/intelligence/similar/definitely-does-not-exist")
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_similar_repos_cache_hit_skips_db(client: AsyncClient):
    cached = {
        "source_repo": "langchain",
        "similar": [{"name": "llama-index", "owner": "o", "description": None,
                     "primary_language": "Python", "primary_category": "rag-retrieval",
                     "stars": 500, "similarity": 0.88, "readme_summary": None}],
        "total": 1,
    }
    mock_db = AsyncMock()

    async def _override():
        yield mock_db

    app.dependency_overrides[get_db] = _override
    try:
        with patch("app.routers.recommendations.cache.get", new=AsyncMock(return_value=cached)):
            resp = await client.get("/intelligence/similar/langchain")
        mock_db.execute.assert_not_called()
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 200
    assert resp.json()["total"] == 1


# ---------------------------------------------------------------------------
# GET /intelligence/recommended
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_recommended_requires_seeds(client: AsyncClient):
    resp = await client.get("/intelligence/recommended")
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_recommended_deduplicates_keeps_best_score(client: AsyncClient):
    """Same repo from two seeds — keep only the higher similarity score."""
    rows_seed1 = [_make_row("shared-repo", 0.90)]
    rows_seed2 = [_make_row("shared-repo", 0.75), _make_row("unique-repo", 0.80)]
    _, override = _override_db_multi([rows_seed1, rows_seed2])

    app.dependency_overrides[get_db] = override
    try:
        with patch("app.routers.recommendations.cache.get", new=AsyncMock(return_value=None)), \
             patch("app.routers.recommendations.cache.set", new=AsyncMock()):
            resp = await client.get("/intelligence/recommended?seeds=langchain,llama-index")
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 200
    recs = resp.json()["recommended"]
    names = [r["name"] for r in recs]
    assert names.count("shared-repo") == 1
    shared = next(r for r in recs if r["name"] == "shared-repo")
    assert shared["similarity"] == 0.90  # best score kept, not 0.75


@pytest.mark.asyncio
async def test_recommended_excludes_seed_repos(client: AsyncClient):
    """A seed repo should never appear in its own recommendations."""
    rows = [_make_row("langchain", 0.99), _make_row("haystack", 0.82)]
    _, override = _override_db_with_rows(rows)

    app.dependency_overrides[get_db] = override
    try:
        with patch("app.routers.recommendations.cache.get", new=AsyncMock(return_value=None)), \
             patch("app.routers.recommendations.cache.set", new=AsyncMock()):
            resp = await client.get("/intelligence/recommended?seeds=langchain")
    finally:
        app.dependency_overrides.pop(get_db, None)

    names = [r["name"] for r in resp.json()["recommended"]]
    assert "langchain" not in names
    assert "haystack" in names


@pytest.mark.asyncio
async def test_recommended_max_5_seeds(client: AsyncClient):
    """More than 5 comma-separated seeds are silently truncated to 5."""
    _, override = _override_db_with_rows([])

    app.dependency_overrides[get_db] = override
    try:
        with patch("app.routers.recommendations.cache.get", new=AsyncMock(return_value=None)), \
             patch("app.routers.recommendations.cache.set", new=AsyncMock()):
            resp = await client.get("/intelligence/recommended?seeds=a,b,c,d,e,f,g,h")
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 200
    assert len(resp.json()["seeds"]) == 5
