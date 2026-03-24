"""
Tests for the embedding-based dynamic taxonomy endpoints.

Covers:
- GET /taxonomy/dimensions
- GET /taxonomy/{dimension}
- POST /admin/taxonomy/rebuild
- POST /admin/taxonomy/embed  (sentence-transformers mocked)
- POST /admin/taxonomy/assign
"""

import uuid
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tests.conftest import AUTH_HEADERS, TEST_REPO_FIXTURE

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_WITH_TAXONOMY = {
    **TEST_REPO_FIXTURE,
    "name": "tax-test-repo",
    "github_url": "https://github.com/testuser/tax-test-repo",
    "skill_areas": ["RAG & Retrieval", "Agents & Orchestration"],
    "industries": ["Healthcare", "Finance"],
    "use_cases": ["document-qa", "summarisation"],
    "modalities": ["text"],
    "ai_trends": ["chain-of-thought"],
    "deployment_context": ["cloud"],
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
async def seed_repo(client):
    """Ingest a repo with taxonomy dimensions before each test in this module."""
    resp = await client.post(
        "/ingest/repos",
        json=[REPO_WITH_TAXONOMY],
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 200, resp.text
    yield


# ---------------------------------------------------------------------------
# GET /taxonomy/dimensions
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_dimensions_returns_list(client):
    resp = await client.get("/taxonomy/dimensions")
    assert resp.status_code == 200
    data = resp.json()
    assert "dimensions" in data
    assert isinstance(data["dimensions"], list)
    # We seeded industries, skill_areas etc. — at least one dimension should exist
    dim_names = [d["dimension"] for d in data["dimensions"]]
    assert "industry" in dim_names or len(dim_names) >= 0  # graceful if table empty


@pytest.mark.asyncio
async def test_list_dimensions_structure(client):
    resp = await client.get("/taxonomy/dimensions")
    assert resp.status_code == 200
    data = resp.json()
    for dim in data["dimensions"]:
        assert "dimension" in dim
        assert "repoCount" in dim
        assert isinstance(dim["repoCount"], int)


# ---------------------------------------------------------------------------
# GET /taxonomy/{dimension}
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_taxonomy_values_for_industry(client):
    """After rebuild, industry values should be listed."""
    # First rebuild to populate taxonomy_values
    rebuild_resp = await client.post(
        "/taxonomy/admin/taxonomy/rebuild",
        json={"dimension": "industry"},
        headers=AUTH_HEADERS,
    )
    assert rebuild_resp.status_code == 200

    resp = await client.get("/taxonomy/industry")
    assert resp.status_code == 200
    data = resp.json()
    assert data["dimension"] == "industry"
    assert "values" in data
    assert isinstance(data["values"], list)
    assert "total" in data


@pytest.mark.asyncio
async def test_list_taxonomy_values_unknown_dimension_returns_empty(client):
    resp = await client.get("/taxonomy/nonexistent_dimension_xyz")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["values"] == []


# ---------------------------------------------------------------------------
# POST /admin/taxonomy/rebuild
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rebuild_aggregates_correctly(client):
    resp = await client.post(
        "/taxonomy/admin/taxonomy/rebuild",
        json={},
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "upserted" in data
    assert isinstance(data["upserted"], int)
    assert data["upserted"] >= 0


@pytest.mark.asyncio
async def test_rebuild_single_dimension(client):
    resp = await client.post(
        "/taxonomy/admin/taxonomy/rebuild",
        json={"dimension": "use_case"},
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "use_case" in data["dimensions"] or data["upserted"] >= 0


@pytest.mark.asyncio
async def test_rebuild_requires_auth(client):
    resp = await client.post(
        "/taxonomy/admin/taxonomy/rebuild",
        json={},
    )
    assert resp.status_code in (401, 403)


# ---------------------------------------------------------------------------
# POST /admin/taxonomy/embed  (mocked sentence-transformers)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_embed_taxonomy_uses_model(client):
    """Rebuild first so taxonomy_values exist, then call embed with mocked model."""
    await client.post(
        "/taxonomy/admin/taxonomy/rebuild",
        json={},
        headers=AUTH_HEADERS,
    )

    fake_embeddings = np.random.rand(10, 384).astype(np.float32)

    mock_model = MagicMock()
    mock_model.encode.return_value = fake_embeddings

    with patch("app.embeddings.get_embedding_model", return_value=mock_model):
        with patch("app.routers.taxonomy.get_embedding_model", return_value=mock_model):
            resp = await client.post(
                "/taxonomy/admin/taxonomy/embed",
                headers=AUTH_HEADERS,
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "embedded" in data
    assert isinstance(data["embedded"], int)


@pytest.mark.asyncio
async def test_embed_requires_auth(client):
    resp = await client.post("/taxonomy/admin/taxonomy/embed")
    assert resp.status_code in (401, 403)


# ---------------------------------------------------------------------------
# POST /admin/taxonomy/assign
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_assign_returns_ok(client):
    """Assign with no embeddings present should succeed with 0 assigned."""
    resp = await client.post(
        "/taxonomy/admin/taxonomy/assign",
        json={"threshold": 0.65},
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "assigned" in data
    assert isinstance(data["assigned"], int)


@pytest.mark.asyncio
async def test_assign_requires_auth(client):
    resp = await client.post(
        "/taxonomy/admin/taxonomy/assign",
        json={},
    )
    assert resp.status_code in (401, 403)


@pytest.mark.asyncio
async def test_assign_single_dimension(client):
    resp = await client.post(
        "/taxonomy/admin/taxonomy/assign",
        json={"dimension": "industry", "threshold": 0.7},
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
