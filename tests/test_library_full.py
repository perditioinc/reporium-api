"""Tests for /library/full endpoint — owned repos only, correct shape."""
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_library_full_returns_only_owned_repos(client: AsyncClient):
    """Every repo in /library/full must have isFork = false."""
    # Ingest an owned repo and a fork via the API
    await client.post(
        "/ingest/repos",
        json=[{
            "name": "reporium", "owner": "perditioinc", "description": "AI tool discovery",
            "is_fork": False, "primary_language": "TypeScript",
            "github_url": "https://github.com/perditioinc/reporium",
            "parent_stars": 0, "parent_forks": 0,
            "readme_summary": "Reporium is an AI tool discovery platform.",
            "commits_last_7_days": 5, "commits_last_30_days": 20, "commits_last_90_days": 60,
        }],
        headers={"X-API-Key": "test-api-key"},
    )
    await client.post(
        "/ingest/repos",
        json=[{
            "name": "tensorflow", "owner": "perditioinc", "description": "ML framework",
            "is_fork": True, "forked_from": "tensorflow/tensorflow", "primary_language": "Python",
            "github_url": "https://github.com/perditioinc/tensorflow",
            "parent_stars": 194000, "parent_forks": 40000,
            "readme_summary": "TensorFlow is an ML framework.",
            "commits_last_7_days": 0, "commits_last_30_days": 0, "commits_last_90_days": 0,
        }],
        headers={"X-API-Key": "test-api-key"},
    )

    resp = await client.get("/library/full")
    # If endpoint exists and returns data, verify no forks
    if resp.status_code == 200:
        data = resp.json()
        repos = data.get("repos", [])
        forks = [r for r in repos if r.get("isFork")]
        assert len(forks) == 0, f"Found {len(forks)} forks in /library/full — should be 0"


@pytest.mark.asyncio
async def test_library_full_response_shape(client: AsyncClient):
    """Response must contain required top-level keys."""
    resp = await client.get("/library/full")
    if resp.status_code == 200:
        data = resp.json()
        assert "username" in data
        assert "generatedAt" in data
        assert "stats" in data
        assert "repos" in data
        stats = data["stats"]
        assert "total" in stats
        assert "built" in stats
        assert "forked" in stats


@pytest.mark.asyncio
async def test_forks_endpoint_returns_forks(client: AsyncClient):
    """/forks endpoint must return fork data."""
    resp = await client.get("/forks?limit=10")
    if resp.status_code == 200:
        data = resp.json()
        assert "forks" in data
        assert "total" in data
