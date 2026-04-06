"""
Tests for graph visualization API enhancements:
- Enhanced /graph/edges with nodes array, temporal filter
- GET /graph/subgraph/{repo_name} — 2-hop neighbourhood
- GET /graph/clusters — category-grouped cluster stats

Uses dependency_overrides[get_db] for DB injection and mocks for Redis
cache — same pattern as test_graph_optimization.py.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.database import get_db
from app.main import app
from app.routers.graph import _extract_quality, _log_scale_stars, _parse_since


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_edge_row(
    source_name="repo-a",
    target_name="repo-b",
    similarity=0.82,
    source_stars=100,
    target_stars=50,
    source_quality=None,
    target_quality=None,
    source_updated_at=None,
    target_updated_at=None,
):
    """Simulate a DB row from the enhanced graph edges SQL."""
    row = MagicMock()
    row.similarity = similarity
    row.source_name = source_name
    row.source_owner = "perditioinc"
    row.source_description = f"{source_name} desc"
    row.source_category = "ai-agents"
    row.source_stars = source_stars
    row.source_quality_signals = source_quality
    row.source_updated_at = source_updated_at
    row.target_name = target_name
    row.target_owner = "perditioinc"
    row.target_description = f"{target_name} desc"
    row.target_category = "rag-retrieval"
    row.target_stars = target_stars
    row.target_quality_signals = target_quality
    row.target_updated_at = target_updated_at
    return row


def _make_counts_row(total_public=100, with_embeddings=80):
    row = MagicMock()
    row.total_public = total_public
    row.with_embeddings = with_embeddings
    return row


def _make_seed_row(name="langchain"):
    row = MagicMock()
    row.id = "00000000-0000-0000-0000-000000000001"
    row.name = name
    row.owner = "langchain-ai"
    row.description = f"{name} framework"
    row.primary_category = "ai-agents"
    row.stargazers_count = 5000
    row.quality_signals = {"overall": 0.85}
    return row


def _make_cluster_stats_row(category, repo_count, avg_stars, repo_name=None,
                            repo_owner=None, repo_stars=None):
    row = MagicMock()
    row.primary_category = category
    row.repo_count = repo_count
    row.avg_stars = avg_stars
    row.repo_name = repo_name
    row.repo_owner = repo_owner
    row.repo_stars = repo_stars
    return row


def _make_inter_cluster_row(cat1, cat2, edge_count):
    row = MagicMock()
    row.cat1 = cat1
    row.cat2 = cat2
    row.edge_count = edge_count
    return row


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
            result.fetchall.return_value = [data]
            result.fetchone.return_value = data
        return result

    mock_db = AsyncMock()
    mock_db.execute = AsyncMock(side_effect=_execute)

    async def _override():
        yield mock_db

    return mock_db, _override


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------

class TestHelpers:

    def test_parse_since_days(self):
        assert _parse_since("7d") == "7 day"

    def test_parse_since_hours(self):
        assert _parse_since("24h") == "24 hour"

    def test_parse_since_minutes(self):
        assert _parse_since("30m") == "30 minute"

    def test_parse_since_none(self):
        assert _parse_since(None) is None

    def test_parse_since_invalid(self):
        assert _parse_since("abc") is None
        assert _parse_since("7x") is None
        assert _parse_since("") is None

    def test_log_scale_stars_positive(self):
        result = _log_scale_stars(100)
        assert result > 0
        assert isinstance(result, float)

    def test_log_scale_stars_zero(self):
        assert _log_scale_stars(0) == 0.0

    def test_log_scale_stars_none(self):
        assert _log_scale_stars(None) == 0.0

    def test_log_scale_stars_large(self):
        # 10000 stars -> log10(10001) ~ 4.0
        result = _log_scale_stars(10000)
        assert 3.9 < result < 4.1

    def test_extract_quality_with_overall(self):
        assert _extract_quality({"overall": 0.85}) == 0.85

    def test_extract_quality_none(self):
        assert _extract_quality(None) is None

    def test_extract_quality_no_overall(self):
        assert _extract_quality({"has_tests": True}) is None


# ---------------------------------------------------------------------------
# A. Enhanced GET /graph/edges — nodes array + temporal filter
# ---------------------------------------------------------------------------

class TestGraphEdgesNodes:

    @pytest.mark.asyncio
    async def test_edges_response_includes_nodes(self):
        """Response should include a 'nodes' array with viz metadata."""
        edge_rows = [
            _make_edge_row("repo-a", "repo-b", 0.9, source_stars=500,
                           target_stars=200,
                           source_quality={"overall": 0.8},
                           target_quality={"overall": 0.6}),
        ]
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

                # Should have nodes array
                assert "nodes" in body
                assert len(body["nodes"]) == 2

                # Check node properties
                node_a = next(n for n in body["nodes"] if n["name"] == "repo-a")
                assert node_a["primary_category"] == "ai-agents"
                assert node_a["stars"] == 500
                assert node_a["stars_log"] > 0
                assert node_a["quality"] == 0.8

                node_b = next(n for n in body["nodes"] if n["name"] == "repo-b")
                assert node_b["stars"] == 200
                assert node_b["quality"] == 0.6
        finally:
            app.dependency_overrides.pop(get_db, None)

    @pytest.mark.asyncio
    async def test_edges_with_since_param(self):
        """Passing ?since=7d should include the param in cache key."""
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
                    resp = await client.get("/graph/edges", params={"since": "7d"})

                assert resp.status_code == 200

                # Verify cache key includes the since value
                cache_get_key = mock_cache.get.call_args[0][0]
                assert "7d" in cache_get_key
        finally:
            app.dependency_overrides.pop(get_db, None)

    @pytest.mark.asyncio
    async def test_edges_without_since_uses_all(self):
        """Without ?since, cache key should include 'all'."""
        counts_row = _make_counts_row()
        _, db_override = _override_db_multi([[], counts_row])
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
                cache_get_key = mock_cache.get.call_args[0][0]
                assert "all" in cache_get_key
        finally:
            app.dependency_overrides.pop(get_db, None)

    @pytest.mark.asyncio
    async def test_nodes_deduplicated(self):
        """Nodes appearing in multiple edges should only appear once."""
        edge_rows = [
            _make_edge_row("repo-a", "repo-b", 0.9),
            _make_edge_row("repo-a", "repo-c", 0.8),
        ]
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

                body = resp.json()
                # repo-a appears in both edges but should be only one node
                names = [n["name"] for n in body["nodes"]]
                assert names.count("repo-a") == 1
                assert len(body["nodes"]) == 3  # a, b, c
        finally:
            app.dependency_overrides.pop(get_db, None)


# ---------------------------------------------------------------------------
# B. GET /graph/subgraph/{repo_name} — 2-hop neighbourhood
# ---------------------------------------------------------------------------

class TestGraphSubgraph:

    @pytest.mark.asyncio
    async def test_subgraph_returns_nodes_and_edges(self):
        """Subgraph should return nodes + edges for 2-hop neighbourhood."""
        seed = _make_seed_row("langchain")
        edge_rows = [
            _make_edge_row("langchain", "llamaindex", 0.88, 5000, 3000),
            _make_edge_row("llamaindex", "chromadb", 0.72, 3000, 1000),
        ]
        _, db_override = _override_db_multi([seed, edge_rows])
        app.dependency_overrides[get_db] = db_override

        try:
            with patch("app.routers.graph.cache") as mock_cache:
                mock_cache.get = AsyncMock(return_value=None)
                mock_cache.set = AsyncMock()

                async with AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://test"
                ) as client:
                    resp = await client.get("/graph/subgraph/langchain")

                assert resp.status_code == 200
                body = resp.json()
                assert body["repo_name"] == "langchain"
                assert body["total_edges"] == 2
                assert body["total_nodes"] >= 3
                assert "nodes" in body
                assert "edges" in body

                # Cache should be set with 30-min TTL
                mock_cache.set.assert_called_once()
                call_args = mock_cache.set.call_args
                assert call_args[1].get("ttl") == 1800 or call_args[0][2] == 1800
        finally:
            app.dependency_overrides.pop(get_db, None)

    @pytest.mark.asyncio
    async def test_subgraph_404_for_missing_repo(self):
        """Should return 404 when repo not found."""
        # First execute returns no rows (repo not found)
        mock_db = AsyncMock()
        result = MagicMock()
        result.fetchone.return_value = None
        mock_db.execute = AsyncMock(return_value=result)

        async def _override():
            yield mock_db

        app.dependency_overrides[get_db] = _override

        try:
            with patch("app.routers.graph.cache") as mock_cache:
                mock_cache.get = AsyncMock(return_value=None)

                async with AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://test"
                ) as client:
                    resp = await client.get("/graph/subgraph/nonexistent-repo")

                assert resp.status_code == 404
                assert "nonexistent-repo" in resp.json()["detail"]
        finally:
            app.dependency_overrides.pop(get_db, None)

    @pytest.mark.asyncio
    async def test_subgraph_cache_hit(self):
        """Cached subgraph should be returned without DB query."""
        cached = {
            "repo_name": "langchain",
            "total_edges": 5,
            "total_nodes": 4,
            "edgeTypes": ["SIMILAR_TO"],
            "nodes": [],
            "edges": [],
        }
        mock_db = AsyncMock()

        async def _override():
            yield mock_db

        app.dependency_overrides[get_db] = _override

        try:
            with patch("app.routers.graph.cache") as mock_cache:
                mock_cache.get = AsyncMock(return_value=cached)

                async with AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://test"
                ) as client:
                    resp = await client.get("/graph/subgraph/langchain")

                assert resp.status_code == 200
                assert resp.json()["repo_name"] == "langchain"
                mock_db.execute.assert_not_called()
        finally:
            app.dependency_overrides.pop(get_db, None)

    @pytest.mark.asyncio
    async def test_subgraph_includes_seed_node(self):
        """Even if seed is not in edge rows, it should appear in nodes."""
        seed = _make_seed_row("lonely-repo")
        # No edges found
        _, db_override = _override_db_multi([seed, []])
        app.dependency_overrides[get_db] = db_override

        try:
            with patch("app.routers.graph.cache") as mock_cache:
                mock_cache.get = AsyncMock(return_value=None)
                mock_cache.set = AsyncMock()

                async with AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://test"
                ) as client:
                    resp = await client.get("/graph/subgraph/lonely-repo")

                assert resp.status_code == 200
                body = resp.json()
                assert body["total_nodes"] == 1
                assert body["nodes"][0]["name"] == "lonely-repo"
        finally:
            app.dependency_overrides.pop(get_db, None)


# ---------------------------------------------------------------------------
# C. GET /graph/clusters — category-grouped cluster stats
# ---------------------------------------------------------------------------

class TestGraphClusters:

    @pytest.mark.asyncio
    async def test_clusters_returns_categories(self):
        """Should return clusters grouped by primary_category."""
        stats_rows = [
            _make_cluster_stats_row("ai-agents", 10, 500.0, "autogen", "microsoft", 8000),
            _make_cluster_stats_row("ai-agents", 10, 500.0, "crewai", "crewai", 5000),
            _make_cluster_stats_row("rag-retrieval", 5, 200.0, "llamaindex", "run-llama", 3000),
        ]
        inter_rows = [
            _make_inter_cluster_row("ai-agents", "rag-retrieval", 12),
        ]
        _, db_override = _override_db_multi([stats_rows, inter_rows])
        app.dependency_overrides[get_db] = db_override

        try:
            with patch("app.routers.graph.cache") as mock_cache:
                mock_cache.get = AsyncMock(return_value=None)
                mock_cache.set = AsyncMock()

                async with AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://test"
                ) as client:
                    resp = await client.get("/graph/clusters")

                assert resp.status_code == 200
                body = resp.json()
                assert body["total_clusters"] == 2

                # Find the ai-agents cluster
                ai = next(c for c in body["clusters"] if c["category"] == "ai-agents")
                assert ai["repo_count"] == 10
                assert ai["avg_stars"] == 500.0
                assert len(ai["top_repos"]) == 2
                assert ai["top_repos"][0]["name"] == "autogen"

                # Inter-cluster edges
                assert ai["inter_cluster_edges"]["rag-retrieval"] == 12

                # Cache set with 1hr TTL
                mock_cache.set.assert_called_once()
                call_args = mock_cache.set.call_args
                assert call_args[1].get("ttl") == 3600 or call_args[0][2] == 3600
        finally:
            app.dependency_overrides.pop(get_db, None)

    @pytest.mark.asyncio
    async def test_clusters_cache_hit(self):
        """Cached clusters should be returned without DB query."""
        cached = {"total_clusters": 3, "clusters": []}
        mock_db = AsyncMock()

        async def _override():
            yield mock_db

        app.dependency_overrides[get_db] = _override

        try:
            with patch("app.routers.graph.cache") as mock_cache:
                mock_cache.get = AsyncMock(return_value=cached)

                async with AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://test"
                ) as client:
                    resp = await client.get("/graph/clusters")

                assert resp.status_code == 200
                assert resp.json()["total_clusters"] == 3
                mock_db.execute.assert_not_called()
        finally:
            app.dependency_overrides.pop(get_db, None)

    @pytest.mark.asyncio
    async def test_clusters_empty(self):
        """No categories should return empty cluster list."""
        _, db_override = _override_db_multi([[], []])
        app.dependency_overrides[get_db] = db_override

        try:
            with patch("app.routers.graph.cache") as mock_cache:
                mock_cache.get = AsyncMock(return_value=None)
                mock_cache.set = AsyncMock()

                async with AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://test"
                ) as client:
                    resp = await client.get("/graph/clusters")

                assert resp.status_code == 200
                body = resp.json()
                assert body["total_clusters"] == 0
                assert body["clusters"] == []
        finally:
            app.dependency_overrides.pop(get_db, None)

    @pytest.mark.asyncio
    async def test_clusters_bidirectional_inter_edges(self):
        """Inter-cluster edges should be tracked bidirectionally."""
        stats_rows = [
            _make_cluster_stats_row("cat-a", 5, 100.0),
            _make_cluster_stats_row("cat-b", 3, 50.0),
        ]
        inter_rows = [
            _make_inter_cluster_row("cat-a", "cat-b", 7),
        ]
        _, db_override = _override_db_multi([stats_rows, inter_rows])
        app.dependency_overrides[get_db] = db_override

        try:
            with patch("app.routers.graph.cache") as mock_cache:
                mock_cache.get = AsyncMock(return_value=None)
                mock_cache.set = AsyncMock()

                async with AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://test"
                ) as client:
                    resp = await client.get("/graph/clusters")

                body = resp.json()
                cat_a = next(c for c in body["clusters"] if c["category"] == "cat-a")
                cat_b = next(c for c in body["clusters"] if c["category"] == "cat-b")

                # Both sides should see the inter-cluster connection
                assert cat_a["inter_cluster_edges"]["cat-b"] == 7
                assert cat_b["inter_cluster_edges"]["cat-a"] == 7
        finally:
            app.dependency_overrides.pop(get_db, None)
