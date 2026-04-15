"""KAN-122: Tests for /metrics/latest frontend alias fields."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_metrics_latest_exposes_frontend_aliases(client):
    """KAN-122: /metrics/latest must include frontend-expected alias fields (additive)."""
    resp = await client.get("/metrics/latest")
    assert resp.status_code == 200
    data = resp.json()

    # Original fields must not be removed
    assert "repos_tracked" in data
    assert "repos_with_ai_skills" in data
    assert "last_sync" in data

    # Additive aliases required by the frontend dashboard
    assert "total_public_repos" in data, "Missing alias: total_public_repos"
    assert "repos_with_embeddings" in data, "Missing alias: repos_with_embeddings"
    assert "snapshot_generated_at" in data, "Missing alias: snapshot_generated_at"

    assert isinstance(data["total_public_repos"], int)
    assert isinstance(data["repos_with_embeddings"], int)
    # snapshot_generated_at may be None when no snapshot is loaded in tests
    assert data["snapshot_generated_at"] is None or isinstance(data["snapshot_generated_at"], str)
