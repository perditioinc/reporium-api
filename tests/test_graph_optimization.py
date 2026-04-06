"""
Tests for graph optimization changes:
- Redis caching on /graph/edges
- Cache-Control header on /graph/edges
- Model name consistency (all-MiniLM-L6-v2)
- Migration 024 structure
"""

import importlib
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
    row = MagicMock()
    row.similarity = similarity
    row.source_name = source_name
    row.source_owner = "org"
    row.source_description = f"{source_name} desc"
    row.source_category = "ai-agents"
    row.target_name = target_name
    row.target_owner = "org"
    row.target_description = f"{target_name} desc"
    row.target_category = "rag-retrieval"
    return row


def _make_counts_row(total_public=100, with_embeddings=80):
    row = MagicMock()
    row.total_public = total_public
    row.with_embeddings = with_embeddings
    return row


def _override_db_multi(call_results: list):
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
            result.fetchall.return_value = [data]
            result.fetchone.return_value = data
        return result

    mock_db = AsyncMock()
    mock_db.execute = AsyncMock(side_effect=_execute)

    async def _override():
        yield mock_db

    return mock_db, _override


# ---------------------------------------------------------------------------
# Task 1: Redis caching on /graph/edges
# ---------------------------------------------------------------------------

class TestGraphEdgesCaching:

    @pytest.mark.asyncio
    async def test_cache_miss_computes_and_stores(self):
        """On cache miss, endpoint should compute edges and store in Redis."""
        edge_rows = [_make_edge_row("a", "b", 0.9)]
        counts_row = _make_counts_row()
        _, db_override = _override_db_multi([edge_rows, counts_row])
        app.dependency_overrides[get_db] = db_override

        try:
            with patch("app.routers.graph.cache") as mock_cache:
                mock_cache.get = AsyncMock(return_value=None)
                mock_cache.set = AsyncMock()

                async with AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://test"
                ) as client:
                    resp = await client.get("/graph/edges")

                assert resp.status_code == 200
                body = resp.json()
                assert body["total"] == 1
                assert body["edges"][0]["edgeType"] == "SIMILAR_TO"

                # Verify cache.set was called with TTL=3600
                mock_cache.set.assert_called_once()
                call_args = mock_cache.set.call_args
                assert call_args[1].get("ttl") == 3600 or call_args[0][2] == 3600
        finally:
            app.dependency_overrides.pop(get_db, None)

    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached(self):
        """On cache hit, endpoint should return cached data without DB query."""
        cached_payload = {
            "total": 2,
            "total_repos": 3,
            "total_public_repos": 100,
            "repos_with_embeddings": 80,
            "edgeTypes": ["SIMILAR_TO"],
            "edges": [{"edgeType": "SIMILAR_TO", "weight": 0.85}],
        }

        mock_db = AsyncMock()
        mock_db.execute = AsyncMock()

        async def _db_override():
            yield mock_db

        app.dependency_overrides[get_db] = _db_override

        try:
            with patch("app.routers.graph.cache") as mock_cache:
                mock_cache.get = AsyncMock(return_value=cached_payload)

                async with AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://test"
                ) as client:
                    resp = await client.get("/graph/edges")

                assert resp.status_code == 200
                assert resp.json()["total"] == 2
                # DB should NOT have been called
                mock_db.execute.assert_not_called()
        finally:
            app.dependency_overrides.pop(get_db, None)

    @pytest.mark.asyncio
    async def test_cache_control_header_on_miss(self):
        """Response should include Cache-Control: public, max-age=3600."""
        edge_rows = [_make_edge_row()]
        counts_row = _make_counts_row()
        _, db_override = _override_db_multi([edge_rows, counts_row])
        app.dependency_overrides[get_db] = db_override

        try:
            with patch("app.routers.graph.cache") as mock_cache:
                mock_cache.get = AsyncMock(return_value=None)
                mock_cache.set = AsyncMock()

                async with AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://test"
                ) as client:
                    resp = await client.get("/graph/edges")

                assert resp.status_code == 200
                assert resp.headers.get("cache-control") == "public, max-age=3600"
        finally:
            app.dependency_overrides.pop(get_db, None)

    @pytest.mark.asyncio
    async def test_cache_control_header_on_hit(self):
        """Cache-Control header should also be present on cache hits."""
        cached_payload = {"total": 0, "edges": [], "edgeTypes": ["SIMILAR_TO"],
                          "total_repos": 0, "total_public_repos": 0, "repos_with_embeddings": 0}

        async def _db_override():
            yield AsyncMock()

        app.dependency_overrides[get_db] = _db_override

        try:
            with patch("app.routers.graph.cache") as mock_cache:
                mock_cache.get = AsyncMock(return_value=cached_payload)

                async with AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://test"
                ) as client:
                    resp = await client.get("/graph/edges")

                assert resp.headers.get("cache-control") == "public, max-age=3600"
        finally:
            app.dependency_overrides.pop(get_db, None)

    @pytest.mark.asyncio
    async def test_cache_key_includes_params(self):
        """Cache key must include limit, min_similarity, and neighbours."""
        _, db_override = _override_db_multi([[], _make_counts_row()])
        app.dependency_overrides[get_db] = db_override

        try:
            with patch("app.routers.graph.cache") as mock_cache:
                mock_cache.get = AsyncMock(return_value=None)
                mock_cache.set = AsyncMock()

                async with AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://test"
                ) as client:
                    resp = await client.get(
                        "/graph/edges",
                        params={"limit": 100, "min_similarity": 0.7, "neighbours": 5},
                    )

                assert resp.status_code == 200
                # Check the cache key used
                cache_get_key = mock_cache.get.call_args[0][0]
                assert "100" in cache_get_key
                assert "0.7" in cache_get_key
                assert "5" in cache_get_key
        finally:
            app.dependency_overrides.pop(get_db, None)


# ---------------------------------------------------------------------------
# Task 2: No repo_edges references remain in intelligence.py
# ---------------------------------------------------------------------------

class TestRepoEdgesRemoved:

    def test_no_repo_edges_in_intelligence(self):
        """intelligence.py should not reference the dead repo_edges table."""
        import inspect
        from app.routers import intelligence
        source = inspect.getsource(intelligence)
        assert "repo_edges" not in source, (
            "Found 'repo_edges' reference in intelligence.py — "
            "all queries should use pgvector similarity via repo_embeddings"
        )


# ---------------------------------------------------------------------------
# Task 3: Migration 024 structure
# ---------------------------------------------------------------------------

class TestMigration024:

    def test_migration_metadata(self):
        """Migration 024 should have correct revision chain."""
        mod = importlib.import_module(
            "migrations.versions.024_add_repo_embeddings_repo_id_btree_index"
        )
        assert mod.revision == "024"
        assert mod.down_revision == "023"

    def test_migration_has_upgrade_and_downgrade(self):
        mod = importlib.import_module(
            "migrations.versions.024_add_repo_embeddings_repo_id_btree_index"
        )
        assert callable(mod.upgrade)
        assert callable(mod.downgrade)

    def test_migration_index_name(self):
        """upgrade() should create idx_repo_embeddings_repo_id."""
        import inspect
        mod = importlib.import_module(
            "migrations.versions.024_add_repo_embeddings_repo_id_btree_index"
        )
        src = inspect.getsource(mod.upgrade)
        assert "idx_repo_embeddings_repo_id" in src
        assert "repo_embeddings" in src


# ---------------------------------------------------------------------------
# Task 4: Model name consistency
# ---------------------------------------------------------------------------

class TestModelNameConsistency:

    def test_embedding_model_is_minilm(self):
        """The embedding model singleton should load all-MiniLM-L6-v2."""
        import inspect
        from app import embeddings
        source = inspect.getsource(embeddings)
        assert "all-MiniLM-L6-v2" in source
        assert "nomic-embed-text" not in source

    def test_repo_embedding_model_default(self):
        """RepoEmbedding ORM model default should be all-MiniLM-L6-v2."""
        from app.models.repo import RepoEmbedding
        # Check the column default
        col = RepoEmbedding.__table__.columns["model"]
        assert col.default.arg == "all-MiniLM-L6-v2"

    def test_admin_insert_uses_correct_model(self):
        """admin.py should insert embeddings with all-MiniLM-L6-v2."""
        import inspect
        from app.routers import admin
        source = inspect.getsource(admin)
        # Should use the correct model name in INSERT
        assert "all-MiniLM-L6-v2" in source
        # Should NOT use the old name
        assert "nomic-embed-text" not in source
