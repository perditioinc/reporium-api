import numpy as np
import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient

from app.routers.search import _semantic_search
from tests.conftest import AUTH_HEADERS, TEST_REPO_FIXTURE


@pytest.mark.asyncio
async def test_search_by_name(client: AsyncClient):
    await client.post("/ingest/repos", json=[TEST_REPO_FIXTURE], headers=AUTH_HEADERS)

    response = await client.get("/search?q=test-repo")
    assert response.status_code == 200
    results = response.json()
    assert isinstance(results, list)
    names = [r["name"] for r in results]
    assert "test-repo" in names


@pytest.mark.asyncio
async def test_search_by_description(client: AsyncClient):
    await client.post("/ingest/repos", json=[TEST_REPO_FIXTURE], headers=AUTH_HEADERS)

    response = await client.get("/search?q=test+repository")
    assert response.status_code == 200
    results = response.json()
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_search_no_results(client: AsyncClient):
    response = await client.get("/search?q=zzz-nonexistent-xyz-12345")
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.asyncio
async def test_search_requires_query(client: AsyncClient):
    response = await client.get("/search")
    assert response.status_code == 422


def _fake_repo(*, repo_id: str, name: str, owner: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=repo_id,
        name=name,
        owner=owner,
        description=f"{name} description",
        is_fork=False,
        forked_from=None,
        primary_language="Python",
        github_url=f"https://github.com/{owner}/{name}",
        fork_sync_state=None,
        behind_by=0,
        ahead_by=0,
        upstream_created_at=None,
        forked_at=None,
        your_last_push_at=None,
        upstream_last_push_at=None,
        parent_stars=100,
        parent_forks=10,
        parent_is_archived=False,
        stargazers_count=25,
        open_issues_count=3,
        commits_last_7_days=1,
        commits_last_30_days=2,
        commits_last_90_days=3,
        readme_summary=f"{name} summary",
        activity_score=80,
        ingested_at="2026-03-24T00:00:00Z",
        updated_at="2026-03-24T00:00:00Z",
        github_updated_at=None,
        license_spdx=None,
        quality_signals=None,
        problem_solved=None,
        tags=[],
        categories=[],
        builders=[],
        ai_dev_skills=[],
        pm_skills=[],
        languages=[],
    )


class _FetchAllResult:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _ScalarsResult:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def all(self):
        return self._rows


@pytest.mark.asyncio
async def test_semantic_search_calls_distance_query_and_applies_limit():
    """When embeddings exist, semantic search uses pgvector distance query."""
    candidate_rows = [
        SimpleNamespace(repo_id="00000000-0000-0000-0000-000000000001", similarity=0.87),
    ]
    repos = [
        _fake_repo(repo_id="00000000-0000-0000-0000-000000000001", name="matched-repo", owner="perditioinc"),
    ]
    db = AsyncMock()
    db.execute = AsyncMock(side_effect=[
        _FetchAllResult(candidate_rows),
        _ScalarsResult(repos),
    ])
    fake_model = MagicMock()
    fake_model.encode.return_value = np.full(384, 0.1, dtype=np.float32)

    with patch("app.routers.search.get_embedding_model", return_value=fake_model):
        results = await _semantic_search(db, query="vector databases", limit=7)

    assert len(results) == 1
    assert results[0].name == "matched-repo"
    # First call is the vector distance query
    first_call_args = db.execute.await_args_list[0].args
    stmt, params = first_call_args
    assert "<=>" in str(stmt)
    assert params["limit"] == 7


@pytest.mark.asyncio
async def test_semantic_search_falls_back_to_full_text_when_no_embeddings():
    """When repo_embeddings is empty, semantic search falls back to ILIKE full-text."""
    fallback_repos = [
        _fake_repo(repo_id="00000000-0000-0000-0000-000000000003", name="fallback-repo", owner="perditioinc"),
    ]
    db = AsyncMock()
    db.execute = AsyncMock(side_effect=[
        _FetchAllResult([]),          # vector query returns nothing
        _ScalarsResult(fallback_repos),  # full-text fallback returns results
    ])
    fake_model = MagicMock()
    fake_model.encode.return_value = np.full(384, 0.1, dtype=np.float32)

    with patch("app.routers.search.get_embedding_model", return_value=fake_model):
        results = await _semantic_search(db, query="vector databases", limit=5)

    assert len(results) == 1
    assert results[0].name == "fallback-repo"
    # similarity is 0.0 for full-text fallback results
    assert results[0].similarity == 0.0
    assert db.execute.await_count == 2


@pytest.mark.asyncio
async def test_semantic_search_returns_results_ordered_by_similarity():
    candidate_rows = [
        SimpleNamespace(repo_id="00000000-0000-0000-0000-000000000002", similarity=0.95),
        SimpleNamespace(repo_id="00000000-0000-0000-0000-000000000001", similarity=0.81),
    ]
    repos = [
        _fake_repo(repo_id="00000000-0000-0000-0000-000000000001", name="second", owner="perditioinc"),
        _fake_repo(repo_id="00000000-0000-0000-0000-000000000002", name="first", owner="perditioinc"),
    ]
    db = AsyncMock()
    db.execute = AsyncMock(side_effect=[
        _FetchAllResult(candidate_rows),
        _ScalarsResult(repos),
    ])
    fake_model = MagicMock()
    fake_model.encode.return_value = np.full(384, 0.1, dtype=np.float32)

    with patch("app.routers.search.get_embedding_model", return_value=fake_model):
        results = await _semantic_search(db, query="agent orchestration", limit=2)

    assert [result.name for result in results] == ["first", "second"]
    assert [result.similarity for result in results] == [0.95, 0.81]
