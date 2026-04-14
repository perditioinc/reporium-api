from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.database import get_db
from app.graph_snapshot import GRAPH_SNAPSHOT_VERSION, build_graph_payload_from_snapshot
from app.main import app


def _sample_snapshot() -> dict:
    return {
        "snapshot_version": GRAPH_SNAPSHOT_VERSION,
        "generated_at": "2026-04-13T01:00:00+00:00",
        "stats": {
            "total_public_repos": 3,
            "repos_with_embeddings": 3,
        },
        "nodes": [
            {
                "repo_id": "repo-1",
                "name": "repo-a",
                "owner": "perditioinc",
                "description": "Repo A",
                "primary_category": "ai-agents",
                "stars": 100,
                "stars_log": 2.0,
                "quality": 0.9,
                "updated_at": "2026-04-13T01:00:00+00:00",
            },
            {
                "repo_id": "repo-2",
                "name": "repo-b",
                "owner": "perditioinc",
                "description": "Repo B",
                "primary_category": "rag-retrieval",
                "stars": 50,
                "stars_log": 1.7,
                "quality": 0.8,
                "updated_at": "2026-04-13T01:00:00+00:00",
            },
            {
                "repo_id": "repo-3",
                "name": "repo-c",
                "owner": "perditioinc",
                "description": "Repo C",
                "primary_category": "vector-db",
                "stars": 25,
                "stars_log": 1.4,
                "quality": 0.7,
                "updated_at": "2026-04-13T01:00:00+00:00",
            },
        ],
        "similarity_edges": [
            {
                "source_repo_id": "repo-1",
                "target_repo_id": "repo-2",
                "rank": 1,
                "weight": 0.82,
            },
            {
                "source_repo_id": "repo-1",
                "target_repo_id": "repo-3",
                "rank": 2,
                "weight": 0.45,
            },
            {
                "source_repo_id": "repo-2",
                "target_repo_id": "repo-1",
                "rank": 1,
                "weight": 0.82,
            },
        ],
        "typed_edges": [
            {
                "source_repo_id": "repo-2",
                "target_repo_id": "repo-3",
                "edge_type": "DEPENDS_ON",
                "weight": 1.0,
            }
        ],
    }


class TestGraphSnapshotPayload:

    def test_build_graph_payload_from_snapshot_filters_similarity_and_keeps_typed_edges(self):
        payload = build_graph_payload_from_snapshot(
            _sample_snapshot(),
            limit=10,
            min_similarity=0.5,
            neighbours=1,
        )

        assert payload["graph_source"] == "snapshot"
        assert payload["total"] == 2
        assert payload["total_repos"] == 3
        assert payload["edgeTypes"] == ["DEPENDS_ON", "SIMILAR_TO"]

        edge_types = {edge["edgeType"] for edge in payload["edges"]}
        assert "SIMILAR_TO" in edge_types
        assert "DEPENDS_ON" in edge_types


class TestGraphSnapshotServing:

    @pytest.mark.asyncio
    async def test_graph_edges_prefers_snapshot_before_db(self):
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock()

        async def _db_override():
            yield mock_db

        app.dependency_overrides[get_db] = _db_override

        try:
            with (
                patch("app.routers.graph.cache") as mock_cache,
                patch("app.routers.graph.load_graph_snapshot", new=AsyncMock(return_value=_sample_snapshot())),
            ):
                mock_cache.get = AsyncMock(return_value=None)
                mock_cache.set = AsyncMock()

                async with AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://test"
                ) as client:
                    resp = await client.get(
                        "/graph/edges",
                        params={"limit": 10, "min_similarity": 0.5, "neighbours": 1},
                    )

                assert resp.status_code == 200
                assert resp.json()["graph_source"] == "snapshot"
                mock_db.execute.assert_not_called()
                mock_cache.set.assert_called_once()
        finally:
            app.dependency_overrides.pop(get_db, None)
