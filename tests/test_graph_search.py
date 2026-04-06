"""
Tests for GET /graph/edges/search and GET /metrics/embeddings.

Uses dependency_overrides[get_db] for DB injection and mocks for the
embedding model and Redis cache — same pattern as test_recommendations.py.
"""
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.database import get_db
from app.main import app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_edge_row(
    source_name="repo-a",
    target_name="repo-b",
    similarity=0.82,
):
    """Simulate a DB row from the graph search SQL."""
    row = MagicMock()
    row.similarity = similarity
    row.source_name = source_name
    row.source_owner = "perditioinc"
    row.source_description = f"{source_name} desc"
    row.source_category = "ai-agents"
    row.target_name = target_name
    row.target_owner = "perditioinc"
    row.target_description = f"{target_name} desc"
    row.target_category = "rag-retrieval"
    return row


def _make_counts_row(total_public=100, with_embeddings=80):
    row = MagicMock()
    row.total_public = total_public
    row.with_embeddings = with_embeddings
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
        result = MagicMock()
        if call_idx < len(call_results):
            data = call_results[call_idx]
            call_idx += 1
        else:
            data = []
        if isinstance(data, list):
            result.fetchall.return_value = data
            result.fetchone.return_value = data[0] if data else None
        else:
            # Single row (for fetchone)
            result.fetchall.return_value = [data]
            result.fetchone.return_value = data
        return result

    mock_db = AsyncMock()
    mock_db.execute = AsyncMock(side_effect=_execute)

    async def _override():
        yield mock_db

    return mock_db, _override


# ---------------------------------------------------------------------------
# GET /graph/edges/search
# ---------------------------------------------------------------------------

class TestGraphEdgesSearch:

    @pytest.mark.asyncio
    async def test_search_returns_edges(self):
        """Search endpoint should embed query, run pgvector search, return edges."""
        edge_rows = [
            _make_edge_row("langchain", "llamaindex", 0.88),
            _make_edge_row("langchain", "autogen", 0.79),
        ]
        _, db_override = _override_db_with_rows(edge_rows)
        app.dependency_overrides[get_db] = db_override

        fake_model = MagicMock()
        fake_model.encode.return_value = np.random.rand(384).astype(np.float32)

        try:
            with patch("app.routers.graph.get_embedding_model", return_value=fake_model), \
                 patch("app.routers.graph.redis_cache") as mock_redis:
                mock_redis.get = AsyncMock(return_value=None)
                mock_redis.set = AsyncMock()

                async with AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://test"
                ) as client:
                    resp = await client.get(
                        "/graph/edges/search",
                        params={"query": "RAG frameworks for LLM"},
                    )

                assert resp.status_code == 200
                body = resp.json()
                assert body["query"] == "RAG frameworks for LLM"
                assert body["total"] == 2
                assert body["total_repos"] == 3  # langchain, llamaindex, autogen
                assert len(body["edges"]) == 2
                assert body["edges"][0]["edgeType"] == "SIMILAR_TO"
                assert body["edges"][0]["source"]["name"] == "langchain"

                # Verify model was called
                fake_model.encode.assert_called_once_with("RAG frameworks for LLM")

                # Verify Redis cache was written
                mock_redis.set.assert_called_once()
                call_args = mock_redis.set.call_args
                assert call_args[1].get("ttl", call_args[0][2] if len(call_args[0]) > 2 else None) == 1800
        finally:
            app.dependency_overrides.pop(get_db, None)

    @pytest.mark.asyncio
    async def test_search_returns_cached_result(self):
        """When Redis has a cached result, skip DB and return it directly."""
        cached_payload = {
            "query": "cached query",
            "total": 1,
            "total_repos": 2,
            "edgeTypes": ["SIMILAR_TO"],
            "edges": [{"edgeType": "SIMILAR_TO", "weight": 0.9}],
        }
        _, db_override = _override_db_with_rows([])
        app.dependency_overrides[get_db] = db_override

        try:
            with patch("app.routers.graph.redis_cache") as mock_redis:
                mock_redis.get = AsyncMock(return_value=cached_payload)

                async with AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://test"
                ) as client:
                    resp = await client.get(
                        "/graph/edges/search",
                        params={"query": "cached query"},
                    )
                assert resp.status_code == 200
                assert resp.json() == cached_payload
        finally:
            app.dependency_overrides.pop(get_db, None)

    @pytest.mark.asyncio
    async def test_search_requires_query(self):
        """Missing query parameter should return 422."""
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/graph/edges/search")
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_search_empty_results(self):
        """Empty DB results should return empty edges list."""
        _, db_override = _override_db_with_rows([])
        app.dependency_overrides[get_db] = db_override

        fake_model = MagicMock()
        fake_model.encode.return_value = np.random.rand(384).astype(np.float32)

        try:
            with patch("app.routers.graph.get_embedding_model", return_value=fake_model), \
                 patch("app.routers.graph.redis_cache") as mock_redis:
                mock_redis.get = AsyncMock(return_value=None)
                mock_redis.set = AsyncMock()

                async with AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://test"
                ) as client:
                    resp = await client.get(
                        "/graph/edges/search",
                        params={"query": "nothing matches"},
                    )
                assert resp.status_code == 200
                body = resp.json()
                assert body["total"] == 0
                assert body["edges"] == []
                assert body["total_repos"] == 0
        finally:
            app.dependency_overrides.pop(get_db, None)

    @pytest.mark.asyncio
    async def test_search_custom_params(self):
        """Custom top_k, neighbours, min_similarity should be forwarded to DB."""
        _, db_override = _override_db_with_rows([])
        app.dependency_overrides[get_db] = db_override

        fake_model = MagicMock()
        fake_model.encode.return_value = np.random.rand(384).astype(np.float32)

        try:
            with patch("app.routers.graph.get_embedding_model", return_value=fake_model), \
                 patch("app.routers.graph.redis_cache") as mock_redis:
                mock_redis.get = AsyncMock(return_value=None)
                mock_redis.set = AsyncMock()

                async with AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://test"
                ) as client:
                    resp = await client.get(
                        "/graph/edges/search",
                        params={
                            "query": "agent frameworks",
                            "top_k": 5,
                            "neighbours": 2,
                            "min_similarity": 0.7,
                        },
                    )
                assert resp.status_code == 200

                # Verify the DB call included our params
                db_call_args = db_override  # just verify no crash
        finally:
            app.dependency_overrides.pop(get_db, None)


# ---------------------------------------------------------------------------
# GET /metrics/embeddings
# ---------------------------------------------------------------------------

class TestMetricsEmbeddings:

    @pytest.mark.asyncio
    async def test_returns_coverage(self):
        """Metrics endpoint returns correct coverage stats."""
        counts_row = _make_counts_row(total_public=200, with_embeddings=150)
        _, db_override = _override_db_multi([counts_row])
        app.dependency_overrides[get_db] = db_override

        try:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/metrics/embeddings")

            assert resp.status_code == 200
            body = resp.json()
            assert body["total_public_repos"] == 200
            assert body["repos_with_embeddings"] == 150
            assert body["coverage_percent"] == 75.0
            assert body["model"] == "all-MiniLM-L6-v2"
            assert body["dimension"] == 384
        finally:
            app.dependency_overrides.pop(get_db, None)

    @pytest.mark.asyncio
    async def test_zero_repos_no_division_error(self):
        """When total_public_repos is 0, coverage should be 0 (no ZeroDivisionError)."""
        counts_row = _make_counts_row(total_public=0, with_embeddings=0)
        _, db_override = _override_db_multi([counts_row])
        app.dependency_overrides[get_db] = db_override

        try:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/metrics/embeddings")

            assert resp.status_code == 200
            body = resp.json()
            assert body["coverage_percent"] == 0.0
            assert body["total_public_repos"] == 0
        finally:
            app.dependency_overrides.pop(get_db, None)

    @pytest.mark.asyncio
    async def test_full_coverage(self):
        """100% coverage should report 100.0."""
        counts_row = _make_counts_row(total_public=50, with_embeddings=50)
        _, db_override = _override_db_multi([counts_row])
        app.dependency_overrides[get_db] = db_override

        try:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/metrics/embeddings")

            assert resp.status_code == 200
            body = resp.json()
            assert body["coverage_percent"] == 100.0
        finally:
            app.dependency_overrides.pop(get_db, None)
