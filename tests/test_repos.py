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
    assert "db" in data


# KAN-53: /repos/{name} must include taxonomy dimensions
@pytest.mark.asyncio
async def test_repo_detail_includes_taxonomy(client: AsyncClient):
    """Repo detail endpoint must return taxonomy list (KAN-53)."""
    fixture_with_taxonomy = {
        **TEST_REPO_FIXTURE,
        "name": "taxonomy-test-repo",
        "skill_areas": ["machine-learning"],
        "industries": ["fintech"],
        "use_cases": ["recommendation"],
        "modalities": [],
        "ai_trends": [],
        "deployment_context": [],
    }
    await client.post("/ingest/repos", json=[fixture_with_taxonomy], headers=AUTH_HEADERS)

    response = await client.get("/repos/taxonomy-test-repo")
    assert response.status_code == 200
    data = response.json()
    assert "taxonomy" in data, "taxonomy key missing from repo detail response"
    assert isinstance(data["taxonomy"], list)

    dimensions = {entry["dimension"] for entry in data["taxonomy"]}
    assert "skill_area" in dimensions, "skill_area dimension missing from taxonomy"
    assert "industry" in dimensions, "industry dimension missing from taxonomy"
    assert "use_case" in dimensions, "use_case dimension missing from taxonomy"

    skill_values = [e["value"] for e in data["taxonomy"] if e["dimension"] == "skill_area"]
    assert "machine-learning" in skill_values


# Stargazers: built repos must have stargazers_count in response (schema test)
@pytest.mark.asyncio
async def test_repo_detail_includes_stargazers_count(client: AsyncClient):
    """stargazers_count must be present in the repo detail response."""
    built_repo = {
        **TEST_REPO_FIXTURE,
        "name": "built-repo-stars",
        "is_fork": False,
        "forked_from": None,
        "stargazers_count": 42,
        "parent_stars": None,
    }
    await client.post("/ingest/repos", json=[built_repo], headers=AUTH_HEADERS)

    response = await client.get("/repos/built-repo-stars")
    assert response.status_code == 200
    data = response.json()
    assert "stargazers_count" in data
    assert data["stargazers_count"] == 42
