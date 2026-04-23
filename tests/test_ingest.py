import pytest
from httpx import AsyncClient
from unittest.mock import AsyncMock, patch

from tests.conftest import AUTH_HEADERS, TEST_REPO_FIXTURE


@pytest.mark.asyncio
async def test_ingest_requires_auth(client: AsyncClient):
    response = await client.post("/ingest/repos", json=[])
    assert response.status_code == 403  # HTTPBearer returns 403 when no token


@pytest.mark.asyncio
async def test_ingest_with_invalid_key(client: AsyncClient):
    response = await client.post(
        "/ingest/repos",
        json=[TEST_REPO_FIXTURE],
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_ingest_with_valid_key(client: AsyncClient):
    response = await client.post(
        "/ingest/repos",
        json=[TEST_REPO_FIXTURE],
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["upserted"] == 1
    assert data["errors"] == []

    library = await client.get("/library/full")
    assert library.status_code == 200
    assert library.json()["repos"][0]["openIssuesCount"] == 42
    assert library.json()["repos"][0]["licenseSpdx"] == "MIT"


@pytest.mark.asyncio
async def test_ingest_is_idempotent(client: AsyncClient):
    # First ingest
    r1 = await client.post("/ingest/repos", json=[TEST_REPO_FIXTURE], headers=AUTH_HEADERS)
    assert r1.status_code == 200

    # Second ingest with same data — should update, not fail
    r2 = await client.post("/ingest/repos", json=[TEST_REPO_FIXTURE], headers=AUTH_HEADERS)
    assert r2.status_code == 200
    assert r2.json()["upserted"] == 1


@pytest.mark.asyncio
async def test_ingest_empty_arrays_preserve_existing_junction_data(client: AsyncClient):
    """Regression test for KAN-123: empty arrays in payload must not wipe existing junction rows."""
    # First ingest with full data
    r1 = await client.post("/ingest/repos", json=[TEST_REPO_FIXTURE], headers=AUTH_HEADERS)
    assert r1.status_code == 200

    # Second ingest with all junction arrays emptied — existing data must be preserved
    sparse_payload = {
        **TEST_REPO_FIXTURE,
        "tags": [],
        "categories": [],
        "builders": [],
        "ai_dev_skills": [],
        "pm_skills": [],
        "languages": [],
        "commits": [],
    }
    r2 = await client.post("/ingest/repos", json=[sparse_payload], headers=AUTH_HEADERS)
    assert r2.status_code == 200

    # Fetch the repo and confirm junction data is intact
    detail = await client.get("/repos/test-repo")
    assert detail.status_code == 200
    data = detail.json()
    assert len(data["tags"]) > 0, "tags were wiped by empty-array payload"
    assert len(data["builders"]) > 0, "builders were wiped by empty-array payload"


@pytest.mark.asyncio
async def test_ingest_cannot_republish_private_repo(client: AsyncClient):
    private_payload = {
        **TEST_REPO_FIXTURE,
        "name": "private-repo",
        "github_url": "https://github.com/testuser/private-repo",
        "is_private": True,
    }
    first = await client.post("/ingest/repos", json=[private_payload], headers=AUTH_HEADERS)
    assert first.status_code == 200
    assert (await client.get("/repos/private-repo")).status_code == 404

    stale_public_payload = {**private_payload, "is_private": False}
    second = await client.post("/ingest/repos", json=[stale_public_payload], headers=AUTH_HEADERS)
    assert second.status_code == 200

    assert (await client.get("/repos/private-repo")).status_code == 404


@pytest.mark.asyncio
async def test_stats_excludes_private_repos(client: AsyncClient):
    # Baseline the shared test DB before inserting our fixtures — earlier
    # tests in the run may have ingested public repos that bump the counts,
    # so we compare by delta instead of asserting absolute totals.
    baseline = (await client.get("/stats")).json()
    baseline_total = baseline["total_repos"]
    baseline_forks = baseline["total_forks"]

    public_payload = {
        **TEST_REPO_FIXTURE,
        "name": "stats-public-repo",
        "github_url": "https://github.com/testuser/stats-public-repo",
        "is_private": False,
    }
    private_payload = {
        **TEST_REPO_FIXTURE,
        "name": "stats-private-repo",
        "github_url": "https://github.com/testuser/stats-private-repo",
        "is_private": True,
        "primary_language": "SecretLang",
        "tags": ["secret-tag"],
        "categories": [{"category_id": "secret", "category_name": "Secret", "is_primary": True}],
    }

    response = await client.post("/ingest/repos", json=[public_payload, private_payload], headers=AUTH_HEADERS)
    assert response.status_code == 200

    stats = (await client.get("/stats")).json()
    # Only the public insert must move the needle — the private one is invisible.
    assert stats["total_repos"] == baseline_total + 1, (
        f"expected +1 public repo, got {baseline_total} -> {stats['total_repos']}"
    )
    assert stats["total_forks"] == baseline_forks + 1, (
        f"expected +1 public fork, got {baseline_forks} -> {stats['total_forks']}"
    )
    # Private-row metadata must never surface in any aggregate dimension
    assert "SecretLang" not in stats["languages"]
    assert "Secret" not in stats["categories"]
    assert "secret-tag" not in stats["top_tags"]


@pytest.mark.asyncio
async def test_ingest_batch_limit(client: AsyncClient):
    items = [
        {**TEST_REPO_FIXTURE, "name": f"batch-repo-{i}", "github_url": f"https://github.com/u/r{i}"}
        for i in range(101)
    ]
    response = await client.post("/ingest/repos", json=items, headers=AUTH_HEADERS)
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_enrich_repo(client: AsyncClient):
    await client.post("/ingest/repos", json=[TEST_REPO_FIXTURE], headers=AUTH_HEADERS)

    response = await client.post(
        "/ingest/repos/test-repo/enrich",
        json={"readme_summary": "An AI-powered test tool.", "activity_score": 90},
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200

    detail = await client.get("/repos/test-repo")
    assert detail.json()["readme_summary"] == "An AI-powered test tool."
    assert detail.json()["activity_score"] == 90


@pytest.mark.asyncio
async def test_ingest_trends(client: AsyncClient):
    response = await client.post(
        "/ingest/trends/snapshot",
        json=[{"tag": "llm", "repo_count": 42, "commit_count_7d": 10}],
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["tag"] == "llm"


@pytest.mark.asyncio
async def test_ingest_gaps(client: AsyncClient):
    response = await client.post(
        "/ingest/gaps",
        json=[{
            "skill": "fine-tuning",
            "severity": "weak",
            "repo_count": 2,
            "why": "Few repos cover fine-tuning end-to-end.",
        }],
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    data = response.json()
    assert data[0]["skill"] == "fine-tuning"


@pytest.mark.asyncio
async def test_ingest_log(client: AsyncClient):
    response = await client.post(
        "/ingest/log",
        json={"mode": "quick", "status": "running"},
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "quick"
    assert data["status"] == "running"


@pytest.mark.asyncio
async def test_repo_ingested_event_requires_ingest_key_when_configured(client: AsyncClient, monkeypatch):
    monkeypatch.setenv("INGEST_API_KEY", "secret-ingest")

    response = await client.post("/ingest/events/repo-ingested", json={"source": "test"})

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_repo_ingested_event_accepts_x_ingest_key_and_runs_refresh(client: AsyncClient, monkeypatch):
    monkeypatch.setenv("INGEST_API_KEY", "secret-ingest")

    with patch("app.routers.ingest.rebuild_taxonomy", new=AsyncMock(return_value={"status": "ok", "upserted": 3})), \
         patch("app.routers.ingest.embed_taxonomy", new=AsyncMock(return_value={"status": "ok", "embedded": 2})), \
         patch("app.routers.ingest.assign_taxonomy", new=AsyncMock(return_value={"status": "ok", "assigned": 11})), \
         patch("app.routers.ingest._rebuild_gap_analysis", new=AsyncMock(return_value={"gap_rows": 8})), \
         patch("app.routers.ingest._refresh_portfolio_intelligence", new=AsyncMock(return_value={"taxonomy_gap_count": 4, "stale_repo_count": 2, "velocity_leader_count": 3, "near_duplicate_cluster_count": 1})), \
         patch("app.routers.ingest.cache.invalidate", new=AsyncMock()) as invalidate_cache, \
         patch("app.routers.ingest.invalidate_library_cache") as invalidate_memory:
        response = await client.post(
            "/ingest/events/repo-ingested",
            json={"source": "pubsub-test"},
            headers={"X-Ingest-Key": "secret-ingest"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["received"]["source"] == "pubsub-test"
    assert data["taxonomy_rebuild"]["upserted"] == 3
    assert data["taxonomy_embed"]["embedded"] == 2
    assert data["taxonomy_assign"]["assigned"] == 11
    assert data["gap_rebuild"]["gap_rows"] == 8
    assert data["portfolio_insights"]["taxonomy_gap_count"] == 4
    assert invalidate_cache.await_count == 3
    invalidate_memory.assert_called_once()


@pytest.mark.asyncio
async def test_repo_ingested_event_decodes_pubsub_envelope(client: AsyncClient):
    import base64
    import json

    encoded = base64.b64encode(json.dumps({"batch": "nightly", "repos": 25}).encode("utf-8")).decode("utf-8")

    with patch("app.routers.ingest.rebuild_taxonomy", new=AsyncMock(return_value={"status": "ok", "upserted": 0})), \
         patch("app.routers.ingest.embed_taxonomy", new=AsyncMock(return_value={"status": "ok", "embedded": 0})), \
         patch("app.routers.ingest.assign_taxonomy", new=AsyncMock(return_value={"status": "ok", "assigned": 0})), \
         patch("app.routers.ingest._rebuild_gap_analysis", new=AsyncMock(return_value={"gap_rows": 0})), \
         patch("app.routers.ingest._refresh_portfolio_intelligence", new=AsyncMock(return_value={"taxonomy_gap_count": 0, "stale_repo_count": 0, "velocity_leader_count": 0, "near_duplicate_cluster_count": 0})), \
         patch("app.routers.ingest.cache.invalidate", new=AsyncMock()), \
         patch("app.routers.ingest.invalidate_library_cache"):
        response = await client.post(
            "/ingest/events/repo-ingested",
            json={"message": {"data": encoded}},
        )

    assert response.status_code == 200
    assert response.json()["received"] == {"batch": "nightly", "repos": 25}


@pytest.mark.asyncio
async def test_repo_ingested_event_skips_embed_failure_and_still_returns_200(client: AsyncClient, monkeypatch):
    monkeypatch.setenv("INGEST_API_KEY", "secret-ingest")

    with patch("app.routers.ingest.rebuild_taxonomy", new=AsyncMock(return_value={"status": "ok", "upserted": 3})), \
         patch("app.routers.ingest.embed_taxonomy", new=AsyncMock(side_effect=RuntimeError("model load failed"))), \
         patch("app.routers.ingest.assign_taxonomy", new=AsyncMock(return_value={"status": "ok", "assigned": 11})), \
         patch("app.routers.ingest._rebuild_gap_analysis", new=AsyncMock(return_value={"gap_rows": 8})), \
         patch("app.routers.ingest._refresh_portfolio_intelligence", new=AsyncMock(return_value={"taxonomy_gap_count": 4, "stale_repo_count": 2, "velocity_leader_count": 3, "near_duplicate_cluster_count": 1})), \
         patch("app.routers.ingest.cache.invalidate", new=AsyncMock()) as invalidate_cache, \
         patch("app.routers.ingest.invalidate_library_cache") as invalidate_memory:
        response = await client.post(
            "/ingest/events/repo-ingested",
            json={"source": "pubsub-test"},
            headers={"X-Ingest-Key": "secret-ingest"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["received"]["source"] == "pubsub-test"
    assert data["taxonomy_rebuild"]["upserted"] == 3
    assert data["taxonomy_embed"]["status"] == "skipped"
    assert data["taxonomy_embed"]["embedded"] == 0
    assert "embed_taxonomy failed" in data["taxonomy_embed"]["error"]
    assert data["taxonomy_assign"]["assigned"] == 11
    assert data["gap_rebuild"]["gap_rows"] == 8
    assert data["portfolio_insights"]["taxonomy_gap_count"] == 4
    assert invalidate_cache.await_count == 3
    invalidate_memory.assert_called_once()


@pytest.mark.asyncio
async def test_ingest_flows_tags_and_categories_into_repo_taxonomy(client: AsyncClient):
    """Regression: tags and categories from the ingest payload must be written
    into repo_taxonomy so /taxonomy/* endpoints can surface them as dimensions.
    Previously these were excluded alongside junction-table fields, leaving
    /taxonomy blind to them.
    """
    from sqlalchemy import text as _text
    import app.database as db_module

    fixture = {
        **TEST_REPO_FIXTURE,
        "name": "taxonomy-flow-repo",
        "github_url": "https://github.com/testuser/taxonomy-flow-repo",
        "tags": ["ai", "rag", "vector-db"],
        "categories": [
            {"category_id": "ai-agents", "category_name": "AI Agents", "is_primary": True},
            {"category_id": "dev-tools", "category_name": "Developer Tools", "is_primary": False},
        ],
    }

    response = await client.post("/ingest/repos", json=[fixture], headers=AUTH_HEADERS)
    assert response.status_code == 200
    assert response.json()["upserted"] == 1

    async with db_module.async_session_factory() as session:
        result = await session.execute(
            _text(
                "SELECT dimension, raw_value FROM repo_taxonomy rt "
                "JOIN repos r ON r.id = rt.repo_id WHERE r.name = :name"
            ),
            {"name": "taxonomy-flow-repo"},
        )
        rows = {(dim, val) for dim, val in result.all()}

    assert ("tag", "ai") in rows
    assert ("tag", "rag") in rows
    assert ("tag", "vector-db") in rows
    assert ("category", "AI Agents") in rows
    assert ("category", "Developer Tools") in rows
