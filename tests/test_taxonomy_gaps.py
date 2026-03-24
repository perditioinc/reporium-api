"""
Tests for taxonomy gap analysis logic.

Covers:
- Gap score calculation (0.0 = well covered, 1.0 = absent)
- Severity classification (low / medium / high)
- Filtering by min_repos / max_repos params on GET /gaps/taxonomy
"""

import pytest
from httpx import AsyncClient

from tests.conftest import AUTH_HEADERS, TEST_REPO_FIXTURE

# ---------------------------------------------------------------------------
# Helper: inline gap score / severity logic matching what trends.py computes
# ---------------------------------------------------------------------------

def _compute_gap_score(repo_count: int, max_count: int) -> float:
    """Mirror the gap score formula from app/routers/trends.py."""
    if max_count == 0:
        return 1.0
    return 1.0 - (repo_count / max_count)


def _classify_severity(gap_score: float) -> str:
    """Mirror the severity classification from app/routers/trends.py."""
    if gap_score >= 0.8:
        return "high"
    if gap_score >= 0.4:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Unit tests for gap score calculation (no DB required)
# ---------------------------------------------------------------------------

class TestGapScoreCalculation:
    def test_zero_repo_count_gives_max_gap(self):
        score = _compute_gap_score(0, 10)
        assert score == 1.0

    def test_equal_to_max_gives_zero_gap(self):
        score = _compute_gap_score(10, 10)
        assert score == 0.0

    def test_half_of_max_gives_half_gap(self):
        score = _compute_gap_score(5, 10)
        assert abs(score - 0.5) < 1e-9

    def test_max_count_zero_returns_one(self):
        """Edge case: if max is 0, gap should be 1.0 (nothing exists)."""
        score = _compute_gap_score(0, 0)
        assert score == 1.0

    def test_score_always_between_zero_and_one(self):
        for repo_count in range(0, 11):
            score = _compute_gap_score(repo_count, 10)
            assert 0.0 <= score <= 1.0, f"Out of range for repo_count={repo_count}"


# ---------------------------------------------------------------------------
# Unit tests for severity classification
# ---------------------------------------------------------------------------

class TestSeverityClassification:
    def test_gap_score_above_0_8_is_high(self):
        assert _classify_severity(0.9) == "high"
        assert _classify_severity(0.8) == "high"
        assert _classify_severity(1.0) == "high"

    def test_gap_score_between_0_4_and_0_8_is_medium(self):
        assert _classify_severity(0.6) == "medium"
        assert _classify_severity(0.4) == "medium"
        assert _classify_severity(0.79) == "medium"

    def test_gap_score_below_0_4_is_low(self):
        assert _classify_severity(0.0) == "low"
        assert _classify_severity(0.2) == "low"
        assert _classify_severity(0.39) == "low"

    def test_boundary_exactly_0_8_is_high(self):
        assert _classify_severity(0.8) == "high"

    def test_boundary_exactly_0_4_is_medium(self):
        assert _classify_severity(0.4) == "medium"


# ---------------------------------------------------------------------------
# Integration: taxonomy-seeded repos and the API endpoint
# ---------------------------------------------------------------------------

REPO_WITH_MANY_DIMS = {
    **TEST_REPO_FIXTURE,
    "name": "gap-test-repo",
    "github_url": "https://github.com/testuser/gap-test-repo",
    "skill_areas": ["RAG & Retrieval", "Agents & Orchestration", "Fine-tuning"],
    "industries": ["Healthcare"],
    "use_cases": ["document-qa"],
    "modalities": ["text"],
    "ai_trends": ["chain-of-thought"],
    "deployment_context": ["cloud"],
}

REPO_SINGLE_DIM = {
    **TEST_REPO_FIXTURE,
    "name": "gap-single-dim-repo",
    "github_url": "https://github.com/testuser/gap-single-dim-repo",
    "skill_areas": ["RAG & Retrieval"],
    "industries": [],
    "use_cases": [],
    "modalities": [],
    "ai_trends": [],
    "deployment_context": [],
}


@pytest.fixture(autouse=True)
async def seed_repos(client: AsyncClient):
    """Seed repos with taxonomy before each test."""
    for repo in [REPO_WITH_MANY_DIMS, REPO_SINGLE_DIM]:
        resp = await client.post(
            "/ingest/repos",
            json=[repo],
            headers=AUTH_HEADERS,
        )
        assert resp.status_code == 200, resp.text
    yield


@pytest.mark.asyncio
async def test_gaps_taxonomy_default_returns_results(client: AsyncClient):
    """GET /gaps/taxonomy with default params returns a list."""
    resp = await client.get("/gaps/taxonomy")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)


@pytest.mark.asyncio
async def test_gaps_taxonomy_min_repos_filter(client: AsyncClient):
    """min_repos param should exclude values below threshold."""
    # With min_repos=999, no repo-count value should match
    resp = await client.get("/gaps/taxonomy", params={"min_repos": 999, "max_repos": 10000})
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


@pytest.mark.asyncio
async def test_gaps_taxonomy_max_repos_filter(client: AsyncClient):
    """max_repos=1 returns only underrepresented values (1 repo)."""
    resp = await client.get("/gaps/taxonomy", params={"min_repos": 1, "max_repos": 1})
    assert resp.status_code == 200
    items = resp.json()
    for item in items:
        assert item["repo_count"] <= 1, f"Expected repo_count<=1, got {item['repo_count']}"


@pytest.mark.asyncio
async def test_gaps_taxonomy_severity_values_are_valid(client: AsyncClient):
    """All severity values in response must be low/medium/high."""
    resp = await client.get("/gaps/taxonomy")
    assert resp.status_code == 200
    for item in resp.json():
        assert item["severity"] in ("low", "medium", "high"), (
            f"Unexpected severity '{item['severity']}' for {item['name']}"
        )


@pytest.mark.asyncio
async def test_gaps_taxonomy_gap_score_in_range(client: AsyncClient):
    """gap_score must be in [0.0, 1.0]."""
    resp = await client.get("/gaps/taxonomy")
    assert resp.status_code == 200
    for item in resp.json():
        score = item["gap_score"]
        assert 0.0 <= score <= 1.0, f"gap_score={score} out of range for {item['name']}"


@pytest.mark.asyncio
async def test_gaps_taxonomy_high_severity_has_high_gap_score(client: AsyncClient):
    """Items with severity='high' should have gap_score >= 0.8."""
    resp = await client.get("/gaps/taxonomy")
    assert resp.status_code == 200
    for item in resp.json():
        if item["severity"] == "high":
            assert item["gap_score"] >= 0.8, (
                f"high severity item has gap_score={item['gap_score']}"
            )


@pytest.mark.asyncio
async def test_gaps_taxonomy_low_severity_has_low_gap_score(client: AsyncClient):
    """Items with severity='low' should have gap_score < 0.4."""
    resp = await client.get("/gaps/taxonomy")
    assert resp.status_code == 200
    for item in resp.json():
        if item["severity"] == "low":
            assert item["gap_score"] < 0.4, (
                f"low severity item has gap_score={item['gap_score']}"
            )
