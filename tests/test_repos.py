import pytest
from httpx import AsyncClient

from tests.conftest import AUTH_HEADERS, TEST_REPO_FIXTURE


@pytest.mark.asyncio
async def test_get_library_returns_repos(client: AsyncClient):
    # Seed a repo first
    await client.post("/ingest/repos", json=[TEST_REPO_FIXTURE], headers=AUTH_HEADERS)

    response = await client.get("/library")
    assert response.status_code == 200
    data = response.json()
    assert "repos" in data
    assert "stats" in data
    assert "categories" in data
    assert "tag_metrics" in data


@pytest.mark.asyncio
async def test_filter_by_category(client: AsyncClient):
    await client.post("/ingest/repos", json=[TEST_REPO_FIXTURE], headers=AUTH_HEADERS)

    response = await client.get("/repos?category=ai-agents")
    assert response.status_code == 200
    repos = response.json()["repos"]
    assert len(repos) > 0
    assert all("AI Agents" in r["allCategories"] for r in repos)


@pytest.mark.asyncio
async def test_filter_by_tag(client: AsyncClient):
    await client.post("/ingest/repos", json=[TEST_REPO_FIXTURE], headers=AUTH_HEADERS)

    response = await client.get("/repos?tag=ai")
    assert response.status_code == 200
    repos = response.json()["repos"]
    assert all("ai" in r["tags"] for r in repos)


@pytest.mark.asyncio
async def test_get_repo_detail(client: AsyncClient):
    await client.post("/ingest/repos", json=[TEST_REPO_FIXTURE], headers=AUTH_HEADERS)

    response = await client.get("/repos/test-repo")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "test-repo"
    assert data["owner"] == "testuser"
    assert "commits" in data
    assert data["license_spdx"] == "MIT"


@pytest.mark.asyncio
async def test_filter_by_license(client: AsyncClient):
    await client.post("/ingest/repos", json=[TEST_REPO_FIXTURE], headers=AUTH_HEADERS)

    response = await client.get("/repos?license=MIT")
    assert response.status_code == 200
    repos = response.json()["repos"]
    assert len(repos) > 0
    assert all(r["license_spdx"] == "MIT" for r in repos)


@pytest.mark.asyncio
async def test_get_repo_not_found(client: AsyncClient):
    response = await client.get("/repos/does-not-exist")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_health(client: AsyncClient):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "database" in data
    assert "cache" in data
