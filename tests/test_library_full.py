"""Tests for /library/full endpoint — owned repos only, correct shape."""
import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy import text

from tests.conftest import TEST_REPO_FIXTURE


@pytest_asyncio.fixture
async def _seed_owned_and_fork(client, _setup_db):
    """Insert one owned repo and one fork into the test DB."""
    import app.database as db_mod

    async with db_mod.async_session_factory() as session:
        # Insert an owned (non-fork) repo
        await session.execute(text("""
            INSERT INTO repos (id, name, owner, description, is_fork, forked_from, primary_language,
                             github_url, parent_stars, parent_forks, readme_summary,
                             commits_last_7_days, commits_last_30_days, commits_last_90_days)
            VALUES ('aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa', 'reporium', 'perditioinc', 'AI tool discovery', false, NULL, 'TypeScript',
                    'https://github.com/perditioinc/reporium', 0, 0, 'Reporium is an AI tool discovery platform.',
                    5, 20, 60)
            ON CONFLICT DO NOTHING;
        """))
        # Insert a fork repo
        await session.execute(text("""
            INSERT INTO repos (id, name, owner, description, is_fork, forked_from, primary_language,
                             github_url, parent_stars, parent_forks, readme_summary,
                             commits_last_7_days, commits_last_30_days, commits_last_90_days)
            VALUES ('bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb', 'tensorflow', 'perditioinc', 'ML framework', true, 'tensorflow/tensorflow', 'Python',
                    'https://github.com/perditioinc/tensorflow', 194000, 40000, 'TensorFlow is an ML framework.',
                    0, 0, 0)
            ON CONFLICT DO NOTHING;
        """))
        await session.commit()
    yield


@pytest.mark.asyncio
async def test_library_full_returns_only_owned_repos(client: AsyncClient, _seed_owned_and_fork):
    """Every repo in /library/full must have isFork = false."""
    resp = await client.get("/library/full")
    assert resp.status_code == 200
    data = resp.json()
    repos = data["repos"]
    forks = [r for r in repos if r.get("isFork")]
    assert len(forks) == 0, f"Found {len(forks)} forks in /library/full — should be 0"


@pytest.mark.asyncio
async def test_library_full_response_shape(client: AsyncClient, _seed_owned_and_fork):
    """Response must match LibraryData TypeScript interface."""
    resp = await client.get("/library/full")
    assert resp.status_code == 200
    data = resp.json()
    # Top-level keys
    assert "username" in data
    assert "generatedAt" in data
    assert "stats" in data
    assert "repos" in data
    assert "tagMetrics" in data
    assert "categories" in data
    # Stats shape
    stats = data["stats"]
    assert "total" in stats
    assert "built" in stats
    assert "forked" in stats


@pytest.mark.asyncio
async def test_forks_endpoint_returns_forks(client: AsyncClient, _seed_owned_and_fork):
    """/forks endpoint must return fork repos."""
    resp = await client.get("/forks?limit=10")
    assert resp.status_code == 200
    data = resp.json()
    assert "forks" in data
    assert "total" in data


@pytest.mark.asyncio
async def test_no_null_descriptions_owned_repos(client: AsyncClient, _seed_owned_and_fork):
    """Owned repos in /library/full should have descriptions."""
    resp = await client.get("/library/full")
    data = resp.json()
    for repo in data["repos"]:
        # All seeded owned repos have descriptions
        if repo["name"] == "reporium":
            assert repo["description"] is not None
            assert repo["description"] != ""
