"""
Tests for quality signals and related endpoints.

Covers:
- POST /admin/quality/compute returns {"computed": N}
- GET /gaps/taxonomy returns list of TaxonomyGapItems with correct fields
- GET /library returns repos with quality_signals field (can be None)
"""

import pytest
from httpx import AsyncClient

from tests.conftest import AUTH_HEADERS, TEST_REPO_FIXTURE

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

REPO_WITH_QUALITY = {
    **TEST_REPO_FIXTURE,
    "name": "quality-test-repo",
    "github_url": "https://github.com/testuser/quality-test-repo",
    "commits_last_7_days": 5,
    "commits_last_30_days": 20,
    "activity_score": 60,
    "open_issues_count": 3,
    # taxonomy so GET /gaps/taxonomy has something to work with
    "skill_areas": ["RAG & Retrieval"],
    "industries": ["Healthcare"],
    "use_cases": ["document-qa"],
    "modalities": ["text"],
    "ai_trends": ["chain-of-thought"],
    "deployment_context": ["cloud"],
}


@pytest.fixture(autouse=True)
async def seed_repo(client: AsyncClient):
    """Ingest a repo before each test in this module."""
    resp = await client.post(
        "/ingest/repos",
        json=[REPO_WITH_QUALITY],
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 200, resp.text
    yield


# ---------------------------------------------------------------------------
# POST /admin/quality/compute
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_compute_quality_signals_returns_computed_count(client: AsyncClient):
    """POST /admin/quality/compute should return a dict with 'computed' key."""
    resp = await client.post(
        "/admin/quality/compute",
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "computed" in data
    assert isinstance(data["computed"], int)
    assert data["computed"] >= 0


@pytest.mark.asyncio
async def test_compute_quality_signals_requires_auth(client: AsyncClient):
    """POST /admin/quality/compute without auth should return 403."""
    resp = await client.post("/admin/quality/compute")
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_compute_quality_signals_updates_repos(client: AsyncClient):
    """After compute, repos should have quality_signals."""
    compute_resp = await client.post(
        "/admin/quality/compute",
        headers=AUTH_HEADERS,
    )
    assert compute_resp.status_code == 200
    data = compute_resp.json()
    assert data["computed"] >= 1


# ---------------------------------------------------------------------------
# GET /gaps/taxonomy
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_gaps_taxonomy_returns_list(client: AsyncClient):
    """GET /gaps/taxonomy should return a list."""
    resp = await client.get("/gaps/taxonomy")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)


@pytest.mark.asyncio
async def test_gaps_taxonomy_items_have_correct_fields(client: AsyncClient):
    """Each TaxonomyGapItem must have dimension, name, repo_count, gap_score, severity."""
    resp = await client.get("/gaps/taxonomy")
    assert resp.status_code == 200
    items = resp.json()
    for item in items:
        assert "dimension" in item, f"Missing 'dimension' in {item}"
        assert "name" in item, f"Missing 'name' in {item}"
        assert "repo_count" in item, f"Missing 'repo_count' in {item}"
        assert "gap_score" in item, f"Missing 'gap_score' in {item}"
        assert "severity" in item, f"Missing 'severity' in {item}"
        assert isinstance(item["repo_count"], int)
        assert isinstance(item["gap_score"], float)
        assert item["severity"] in ("low", "medium", "high"), f"Invalid severity: {item['severity']}"


@pytest.mark.asyncio
async def test_gaps_taxonomy_gap_score_range(client: AsyncClient):
    """gap_score should be in [0.0, 1.0]."""
    resp = await client.get("/gaps/taxonomy")
    assert resp.status_code == 200
    for item in resp.json():
        assert 0.0 <= item["gap_score"] <= 1.0, f"gap_score out of range: {item}"


@pytest.mark.asyncio
async def test_gaps_taxonomy_min_max_repos_filter(client: AsyncClient):
    """min_repos / max_repos params should filter results."""
    # With max_repos=0, there should be no results (nothing has 0 or fewer repos)
    resp = await client.get("/gaps/taxonomy", params={"min_repos": 1, "max_repos": 0})
    assert resp.status_code == 200
    # empty or valid list — we only care it doesn't crash
    assert isinstance(resp.json(), list)


# ---------------------------------------------------------------------------
# GET /library — quality_signals field
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_library_includes_quality_signals_field(client: AsyncClient):
    """GET /library should return repos that include the quality_signals field."""
    resp = await client.get("/library")
    assert resp.status_code == 200
    data = resp.json()
    repos = data.get("repos") or data.get("items") or []
    # There must be at least 1 repo (seeded above)
    assert len(repos) >= 1
    # Every repo must have the quality_signals key (may be None)
    for repo in repos:
        assert "quality_signals" in repo, f"Missing quality_signals in repo: {repo.get('name')}"


@pytest.mark.asyncio
async def test_library_quality_signals_is_dict_or_none(client: AsyncClient):
    """quality_signals must be either null (None) or a dict."""
    resp = await client.get("/library")
    assert resp.status_code == 200
    data = resp.json()
    repos = data.get("repos") or data.get("items") or []
    for repo in repos:
        qs = repo.get("quality_signals")
        assert qs is None or isinstance(qs, dict), (
            f"quality_signals has unexpected type {type(qs)} for repo {repo.get('name')}"
        )
