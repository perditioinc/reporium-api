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
