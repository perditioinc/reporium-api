"""
Tests for GET /intelligence/compare — side-by-side repo comparison.

Uses dependency_overrides[get_db] with mock sessions, same pattern as
test_recommendations.py.
"""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from httpx import AsyncClient

from app.database import get_db
from app.main import app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_repo(name="langchain", owner="langchain-ai", stars=50000, tags=None,
               quality="high", maturity="mature"):
    """Return a mock Repo object with all fields the compare endpoint reads."""
    repo = MagicMock()
    repo.id = uuid4()
    repo.name = name
    repo.owner = owner
    repo.is_private = False
    repo.parent_stars = stars
    repo.stargazers_count = stars
    repo.primary_category = "rag-retrieval"
    repo.primary_language = "Python"
    repo.description = f"The {name} framework"
    repo.quality_signals = {"quality": quality, "maturity": maturity}
    repo.has_tests = True
    repo.has_ci = True
    repo.contributors_count = 120
    repo.issue_close_rate = 0.85
    repo.pr_merge_rate = 0.72
    repo.community_health_pct = 80
    repo.release_count = 45
    repo.activity_score = 90

    # Tags relationship
    tag_names = tags or ["ai", "python", "llm"]
    tag_objs = []
    for t in tag_names:
        tag_mock = MagicMock()
        tag_mock.tag = t
        tag_objs.append(tag_mock)
    repo.tags = tag_objs
    return repo


def _make_mention(repo_id, title="Show HN: langchain", score=300):
    m = MagicMock()
    m.repo_id = repo_id
    m.source = "hackernews"
    m.title = title
    m.url = "https://news.ycombinator.com/item?id=12345"
    m.score = score
    m.comment_count = 150
    return m


def _build_db_override(repos, mentions=None):
    """
    Build a mock db session. First execute() returns repos (via scalars().all()),
    second returns mentions (via scalars().all()).
    """
    call_idx = 0

    async def _execute(*args, **kwargs):
        nonlocal call_idx
        result = MagicMock()
        if call_idx == 0:
            # repos query — uses selectinload so returns via scalars().all()
            result.scalars.return_value.all.return_value = repos
        else:
            # mentions query
            result.scalars.return_value.all.return_value = mentions or []
        call_idx += 1
        return result

    mock_db = AsyncMock()
    mock_db.execute = _execute

    async def _override():
        yield mock_db

    return _override


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_compare_valid_two_repos(client: AsyncClient):
    """Two-repo comparison returns expected structure."""
    repo_a = _make_repo("langchain", tags=["ai", "python", "llm", "rag"])
    repo_b = _make_repo("llama-index", owner="run-llama", stars=30000,
                        tags=["ai", "python", "llm", "search"])
    mention = _make_mention(repo_a.id)

    override = _build_db_override([repo_a, repo_b], [mention])
    app.dependency_overrides[get_db] = override
    try:
        with patch("app.routers.compare.cache.get", new=AsyncMock(return_value=None)), \
             patch("app.routers.compare.cache.set", new=AsyncMock()):
            resp = await client.get("/intelligence/compare?repos=langchain,llama-index")
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 200
    data = resp.json()

    # Top-level keys
    assert "repos" in data
    assert "comparison_matrix" in data
    assert "shared_tags" in data
    assert "unique_tags" in data

    # Two repos returned
    assert len(data["repos"]) == 2

    # Comparison matrix keys
    matrix = data["comparison_matrix"]
    assert "stars" in matrix
    assert "contributors" in matrix
    assert "issue_close_rate" in matrix
    assert "quality" in matrix
    assert "maturity" in matrix


@pytest.mark.asyncio
async def test_compare_single_repo_error(client: AsyncClient):
    """Single repo should return 400."""
    resp = await client.get("/intelligence/compare?repos=langchain")
    assert resp.status_code == 400
    assert "At least 2" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_compare_too_many_repos_error(client: AsyncClient):
    """More than 5 repos should return 400."""
    names = ",".join([f"repo{i}" for i in range(6)])
    resp = await client.get(f"/intelligence/compare?repos={names}")
    assert resp.status_code == 400
    assert "At most 5" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_compare_unknown_repo_404(client: AsyncClient):
    """If a repo is not found, return 404 with its name."""
    repo_a = _make_repo("langchain")
    # Only langchain found, not "nonexistent"
    override = _build_db_override([repo_a])
    app.dependency_overrides[get_db] = override
    try:
        with patch("app.routers.compare.cache.get", new=AsyncMock(return_value=None)), \
             patch("app.routers.compare.cache.set", new=AsyncMock()):
            resp = await client.get("/intelligence/compare?repos=langchain,nonexistent")
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 404
    assert "nonexistent" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_compare_response_shape(client: AsyncClient):
    """Each repo dict has all expected keys."""
    repo_a = _make_repo("langchain", tags=["ai", "llm"])
    repo_b = _make_repo("haystack", tags=["ai", "search"])

    override = _build_db_override([repo_a, repo_b], [])
    app.dependency_overrides[get_db] = override
    try:
        with patch("app.routers.compare.cache.get", new=AsyncMock(return_value=None)), \
             patch("app.routers.compare.cache.set", new=AsyncMock()):
            resp = await client.get("/intelligence/compare?repos=langchain,haystack")
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 200
    data = resp.json()
    expected_keys = {
        "name", "owner", "stars", "primary_category", "language",
        "description", "quality", "maturity", "has_tests", "has_ci",
        "contributors_count", "issue_close_rate", "pr_merge_rate",
        "community_health_pct", "release_count", "activity_score",
        "enriched_tags", "pros_cons", "hn_mentions_count", "top_hn_mention",
    }
    for repo in data["repos"]:
        assert set(repo.keys()) == expected_keys


@pytest.mark.asyncio
async def test_compare_shared_and_unique_tags(client: AsyncClient):
    """Shared tags are the intersection, unique tags are per-repo differences."""
    repo_a = _make_repo("langchain", tags=["ai", "python", "rag"])
    repo_b = _make_repo("haystack", tags=["ai", "python", "search"])

    override = _build_db_override([repo_a, repo_b], [])
    app.dependency_overrides[get_db] = override
    try:
        with patch("app.routers.compare.cache.get", new=AsyncMock(return_value=None)), \
             patch("app.routers.compare.cache.set", new=AsyncMock()):
            resp = await client.get("/intelligence/compare?repos=langchain,haystack")
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 200
    data = resp.json()

    # Shared: ai, python
    assert sorted(data["shared_tags"]) == ["ai", "python"]

    # Unique
    assert data["unique_tags"]["langchain"] == ["rag"]
    assert data["unique_tags"]["haystack"] == ["search"]
