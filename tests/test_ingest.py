import pytest
from httpx import AsyncClient

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
