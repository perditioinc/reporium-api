import pytest
from httpx import AsyncClient

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
