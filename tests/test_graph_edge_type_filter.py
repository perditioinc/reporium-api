"""
KAN-graph-edge-type-filter: the GET /graph/edges ``edge_type`` query param must
actually filter the returned edges by type instead of being a no-op.

Previously the endpoint accepted no ``edge_type`` param at all, so callers asking
for (e.g.) only DEPENDS_ON edges received every edge type. These tests pin the
new behaviour: the response ``edges`` list, ``edgeTypes`` summary and ``total``
count are all scoped to the requested type, while the response shape is
otherwise unchanged.

Uses dependency_overrides[get_db] for DB injection and mocks for Redis cache,
mirroring tests/test_graph_viz.py.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.database import get_db
from app.main import app
from app.routers.graph import _filter_edges_by_type


def _make_edge_row(source_name="repo-a", target_name="repo-b", similarity=0.82):
    row = MagicMock()
    row.similarity = similarity
    row.source_name = source_name
    row.source_owner = "perditioinc"
    row.source_description = f"{source_name} desc"
    row.source_category = "ai-agents"
    row.source_stars = 100
    row.source_quality_signals = None
    row.source_updated_at = None
    row.target_name = target_name
    row.target_owner = "perditioinc"
    row.target_description = f"{target_name} desc"
    row.target_category = "rag-retrieval"
    row.target_stars = 50
    row.target_quality_signals = None
    row.target_updated_at = None
    return row


def _make_typed_edge_row(edge_type, source_name, target_name, weight=0.9):
    row = MagicMock()
    row.edge_type = edge_type
    row.weight = weight
    row.source_name = source_name
    row.source_owner = "perditioinc"
    row.source_description = f"{source_name} desc"
    row.source_category = "ai-agents"
    row.source_stars = 100
    row.source_quality_signals = None
    row.target_name = target_name
    row.target_owner = "perditioinc"
    row.target_description = f"{target_name} desc"
    row.target_category = "rag-retrieval"
    row.target_stars = 50
    row.target_quality_signals = None
    return row


def _make_counts_row():
    row = MagicMock()
    row.total_public = 100
    row.with_embeddings = 80
    row.total_graph_edges = 0
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
# Unit test for the pure filter helper
# ---------------------------------------------------------------------------

class TestFilterEdgesByType:

    def _payload(self):
        return {
            "total": 3,
            "total_repos": 4,
            "edgeTypes": ["DEPENDS_ON", "SIMILAR_TO"],
            "nodes": [{"name": "a"}, {"name": "b"}, {"name": "c"}, {"name": "d"}],
            "edges": [
                {"edgeType": "SIMILAR_TO", "source": {"name": "a"}, "target": {"name": "b"}},
                {"edgeType": "SIMILAR_TO", "source": {"name": "a"}, "target": {"name": "c"}},
                {"edgeType": "DEPENDS_ON", "source": {"name": "a"}, "target": {"name": "d"}},
            ],
        }

    def test_none_returns_payload_unchanged(self):
        payload = self._payload()
        result = _filter_edges_by_type(payload, None)
        assert result["total"] == 3
        assert len(result["edges"]) == 3

    def test_filters_to_single_type(self):
        result = _filter_edges_by_type(self._payload(), "DEPENDS_ON")
        assert result["total"] == 1
        assert len(result["edges"]) == 1
        assert result["edges"][0]["edgeType"] == "DEPENDS_ON"
        assert result["edgeTypes"] == ["DEPENDS_ON"]

    def test_filter_is_case_insensitive(self):
        result = _filter_edges_by_type(self._payload(), "depends_on")
        assert result["total"] == 1
        assert result["edges"][0]["edgeType"] == "DEPENDS_ON"

    def test_filter_no_match_yields_empty(self):
        result = _filter_edges_by_type(self._payload(), "EXTENDS")
        assert result["total"] == 0
        assert result["edges"] == []
        assert result["edgeTypes"] == []

    def test_response_shape_preserved(self):
        result = _filter_edges_by_type(self._payload(), "SIMILAR_TO")
        # nodes untouched; all top-level keys retained
        assert set(result.keys()) == {
            "total", "total_repos", "edgeTypes", "nodes", "edges"
        }
        assert len(result["nodes"]) == 4


# ---------------------------------------------------------------------------
# Endpoint integration: edge_type actually filters (DB path)
# ---------------------------------------------------------------------------

class TestEdgesEndpointEdgeTypeFilter:

    @pytest.mark.asyncio
    async def test_edge_type_filters_db_path(self):
        """?edge_type=DEPENDS_ON must drop SIMILAR_TO edges from the response."""
        sim_rows = [_make_edge_row("repo-a", "repo-b", 0.9)]
        typed_rows = [_make_typed_edge_row("DEPENDS_ON", "repo-a", "repo-c", 0.95)]
        counts_row = _make_counts_row()
        _, db_override = _override_db_multi([sim_rows, typed_rows, counts_row])
        app.dependency_overrides[get_db] = db_override

        try:
            with patch("app.routers.graph.load_graph_snapshot",
                       new=AsyncMock(return_value=None)), \
                 patch("app.routers.graph.redis_cache") as mock_cache:
                mock_cache.get = AsyncMock(return_value=None)
                mock_cache.set = AsyncMock()

                async with AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://test"
                ) as client:
                    resp = await client.get(
                        "/graph/edges", params={"edge_type": "DEPENDS_ON"}
                    )

                assert resp.status_code == 200
                body = resp.json()
                assert body["edgeTypes"] == ["DEPENDS_ON"]
                assert all(e["edgeType"] == "DEPENDS_ON" for e in body["edges"])
                assert body["total"] == len(body["edges"])
                assert body["total"] >= 1
        finally:
            app.dependency_overrides.pop(get_db, None)

    @pytest.mark.asyncio
    async def test_edge_type_in_cache_key(self):
        """The edge_type must be part of the Redis cache key to avoid collisions."""
        counts_row = _make_counts_row()
        _, db_override = _override_db_multi([[], [], counts_row])
        app.dependency_overrides[get_db] = db_override

        try:
            with patch("app.routers.graph.load_graph_snapshot",
                       new=AsyncMock(return_value=None)), \
                 patch("app.routers.graph.redis_cache") as mock_cache:
                mock_cache.get = AsyncMock(return_value=None)
                mock_cache.set = AsyncMock()

                async with AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://test"
                ) as client:
                    resp = await client.get(
                        "/graph/edges", params={"edge_type": "EXTENDS"}
                    )

                assert resp.status_code == 200
                cache_get_key = mock_cache.get.call_args[0][0]
                assert "EXTENDS" in cache_get_key
        finally:
            app.dependency_overrides.pop(get_db, None)

    @pytest.mark.asyncio
    async def test_no_edge_type_returns_all(self):
        """Without edge_type, both SIMILAR_TO and typed edges are returned."""
        sim_rows = [_make_edge_row("repo-a", "repo-b", 0.9)]
        typed_rows = [_make_typed_edge_row("DEPENDS_ON", "repo-a", "repo-c", 0.95)]
        counts_row = _make_counts_row()
        _, db_override = _override_db_multi([sim_rows, typed_rows, counts_row])
        app.dependency_overrides[get_db] = db_override

        try:
            with patch("app.routers.graph.load_graph_snapshot",
                       new=AsyncMock(return_value=None)), \
                 patch("app.routers.graph.redis_cache") as mock_cache:
                mock_cache.get = AsyncMock(return_value=None)
                mock_cache.set = AsyncMock()

                async with AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://test"
                ) as client:
                    resp = await client.get("/graph/edges")

                assert resp.status_code == 200
                body = resp.json()
                assert set(body["edgeTypes"]) == {"SIMILAR_TO", "DEPENDS_ON"}
        finally:
            app.dependency_overrides.pop(get_db, None)
