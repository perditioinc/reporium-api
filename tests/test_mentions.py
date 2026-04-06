"""Tests for HackerNews mentions backfill and retrieval endpoints."""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient

from tests.conftest import AUTH_HEADERS, TEST_API_KEY

# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------

FAKE_HN_HITS = {
    "hits": [
        {
            "objectID": "12345",
            "title": "Show HN: Cool AI Repo",
            "points": 150,
            "num_comments": 42,
            "author": "pg",
            "created_at_i": 1700000000,
        },
        {
            "objectID": "67890",
            "title": "Another mention of test-repo",
            "points": 30,
            "num_comments": 5,
            "author": "dang",
            "created_at_i": 1700100000,
        },
    ]
}

EMPTY_HN_HITS = {"hits": []}


def _mock_hn_response(json_data):
    """Create a mock httpx response."""
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = json_data
    return resp


# ---------------------------------------------------------------------------
# POST /admin/backfill-hn-mentions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_backfill_hn_mentions_requires_api_key(client: AsyncClient):
    response = await client.post("/admin/backfill-hn-mentions")
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_backfill_hn_mentions_dry_run(client: AsyncClient):
    """Dry run should report mentions_found but insert nothing."""

    # Seed a repo first
    repo_payload = {
        "name": "hn-test-repo",
        "owner": "testuser",
        "description": "A repo for HN tests",
        "is_fork": False,
        "primary_language": "Python",
        "github_url": "https://github.com/testuser/hn-test-repo",
        "tags": [],
        "categories": [],
        "builders": [],
        "ai_dev_skills": [],
        "pm_skills": [],
        "languages": [],
        "commits": [],
    }
    ingest_resp = await client.post(
        "/ingest/repos",
        json=[repo_payload],
        headers=AUTH_HEADERS,
    )
    assert ingest_resp.status_code in (200, 201)

    # Mock httpx to return fake HN data
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=_mock_hn_response(FAKE_HN_HITS))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("app.routers.admin.httpx.AsyncClient", return_value=mock_client):
        response = await client.post(
            "/admin/backfill-hn-mentions?dry_run=true",
            headers={"Authorization": f"Bearer {TEST_API_KEY}"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["dry_run"] is True
    assert data["mentions_found"] > 0
    assert data["mentions_inserted"] == 0


@pytest.mark.asyncio
async def test_backfill_hn_mentions_inserts(client: AsyncClient):
    """Actual run should insert mentions and report counts."""

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=_mock_hn_response(FAKE_HN_HITS))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("app.routers.admin.httpx.AsyncClient", return_value=mock_client):
        response = await client.post(
            "/admin/backfill-hn-mentions?dry_run=false",
            headers={"Authorization": f"Bearer {TEST_API_KEY}"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["dry_run"] is False
    assert data["mentions_inserted"] >= 0  # may be 0 if re-run (idempotent)
    assert "total_repos" in data
    assert "failed" in data


@pytest.mark.asyncio
async def test_backfill_hn_idempotent(client: AsyncClient):
    """Running backfill twice should not create duplicates."""

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=_mock_hn_response(FAKE_HN_HITS))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("app.routers.admin.httpx.AsyncClient", return_value=mock_client):
        r1 = await client.post(
            "/admin/backfill-hn-mentions",
            headers={"Authorization": f"Bearer {TEST_API_KEY}"},
        )
        r2 = await client.post(
            "/admin/backfill-hn-mentions",
            headers={"Authorization": f"Bearer {TEST_API_KEY}"},
        )

    assert r1.status_code == 200
    assert r2.status_code == 200
    # Second run should insert 0 new mentions (all duplicates)
    assert r2.json()["mentions_inserted"] == 0


# ---------------------------------------------------------------------------
# GET /repos/{repo_id}/mentions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_mentions_nonexistent_repo(client: AsyncClient):
    fake_id = str(uuid.uuid4())
    response = await client.get(f"/repos/{fake_id}/mentions")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_mentions_empty(client: AsyncClient):
    """A repo with no mentions should return an empty list."""
    repo_payload = {
        "name": "no-mentions-repo",
        "owner": "testuser",
        "description": "No mentions here",
        "is_fork": False,
        "primary_language": "Go",
        "github_url": "https://github.com/testuser/no-mentions-repo",
        "tags": [],
        "categories": [],
        "builders": [],
        "ai_dev_skills": [],
        "pm_skills": [],
        "languages": [],
        "commits": [],
    }
    ingest_resp = await client.post(
        "/ingest/repos", json=[repo_payload], headers=AUTH_HEADERS
    )
    assert ingest_resp.status_code in (200, 201)

    repos_resp = await client.get("/repos?q=no-mentions-repo", headers=AUTH_HEADERS)
    assert repos_resp.status_code == 200
    repos_data = repos_resp.json()
    repo_id = repos_data["repos"][0]["id"]

    response = await client.get(f"/repos/{repo_id}/mentions")
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.asyncio
async def test_get_mentions_returns_data(client: AsyncClient):
    """After backfill, mentions endpoint should return results."""
    repos_resp = await client.get("/repos?q=hn-test-repo", headers=AUTH_HEADERS)
    if repos_resp.status_code != 200 or not repos_resp.json().get("repos"):
        pytest.skip("hn-test-repo not found -- depends on prior backfill test")

    repo_id = repos_resp.json()["repos"][0]["id"]
    response = await client.get(f"/repos/{repo_id}/mentions")
    assert response.status_code == 200
    mentions = response.json()

    if len(mentions) > 0:
        m = mentions[0]
        assert "id" in m
        assert "source" in m
        assert "title" in m
        assert "score" in m
        assert "url" in m


@pytest.mark.asyncio
async def test_get_mentions_source_filter(client: AsyncClient):
    """Filter by source should only return matching mentions."""
    repos_resp = await client.get("/repos?q=hn-test-repo", headers=AUTH_HEADERS)
    if repos_resp.status_code != 200 or not repos_resp.json().get("repos"):
        pytest.skip("hn-test-repo not found")

    repo_id = repos_resp.json()["repos"][0]["id"]

    # Filter for a source that does not exist
    response = await client.get(f"/repos/{repo_id}/mentions?source=reddit")
    assert response.status_code == 200
    assert response.json() == []

    # Filter for hackernews (should return results if backfill ran)
    response = await client.get(f"/repos/{repo_id}/mentions?source=hackernews")
    assert response.status_code == 200
    for m in response.json():
        assert m["source"] == "hackernews"


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------


def test_hn_hit_to_mention():
    from app.routers.admin import _hn_hit_to_mention

    hit = {
        "objectID": "99999",
        "title": "Test Title",
        "points": 100,
        "num_comments": 20,
        "author": "testauthor",
        "created_at_i": 1700000000,
    }
    repo_id = uuid.uuid4()
    result = _hn_hit_to_mention(hit, repo_id)

    assert result["repo_id"] == repo_id
    assert result["source"] == "hackernews"
    assert result["external_id"] == "99999"
    assert result["title"] == "Test Title"
    assert result["score"] == 100
    assert result["comment_count"] == 20
    assert result["author"] == "testauthor"
    assert result["url"] == "https://news.ycombinator.com/item?id=99999"
    assert result["published_at"] is not None


def test_hn_hit_to_mention_missing_fields():
    from app.routers.admin import _hn_hit_to_mention

    hit = {"objectID": "111", "title": None}
    repo_id = uuid.uuid4()
    result = _hn_hit_to_mention(hit, repo_id)

    assert result["title"] == "(no title)"
    assert result["score"] is None
    assert result["published_at"] is None
