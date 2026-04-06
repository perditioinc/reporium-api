"""Tests for POST /admin/backfill-licenses endpoint."""

import uuid
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from httpx import AsyncClient

from tests.conftest import AUTH_HEADERS, TEST_API_KEY


# ---------------------------------------------------------------------------
# Auth gating
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_backfill_licenses_requires_api_key(client: AsyncClient):
    """Endpoint rejects requests without a valid API key."""
    resp = await client.post("/admin/backfill-licenses")
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_backfill_licenses_requires_admin_key(client: AsyncClient):
    """Endpoint rejects requests that have an API key but no admin key."""
    resp = await client.post(
        "/admin/backfill-licenses",
        headers={"Authorization": f"Bearer {TEST_API_KEY}"},
    )
    # Should fail because X-Admin-Key header is missing
    assert resp.status_code in (403, 401)


# ---------------------------------------------------------------------------
# Dry-run mode
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_backfill_licenses_dry_run(client: AsyncClient):
    """Dry-run returns total count but does not update any rows."""
    resp = await client.post(
        "/admin/backfill-licenses?dry_run=true",
        headers={"Authorization": f"Bearer {TEST_API_KEY}"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["dry_run"] is True
    assert data["updated"] == 0
    assert "total" in data


# ---------------------------------------------------------------------------
# Happy path — mock GitHub API
# ---------------------------------------------------------------------------

def _make_gh_response(spdx_id: str | None, status: int = 200) -> httpx.Response:
    """Build a fake httpx.Response mimicking GitHub's repos endpoint."""
    if spdx_id is None:
        body = {"license": None}
    else:
        body = {"license": {"spdx_id": spdx_id}}
    return httpx.Response(status_code=status, json=body, request=httpx.Request("GET", "https://api.github.com"))


@pytest.mark.asyncio
async def test_backfill_licenses_updates_repos(client: AsyncClient):
    """With mocked GitHub API, repos without license_spdx get updated."""
    from sqlalchemy import text

    import app.database as db_module

    # Insert a repo with no license
    repo_id = str(uuid.uuid4())
    async with db_module.async_session_factory() as session:
        await session.execute(text(
            "INSERT INTO repos (id, name, owner, github_url, is_fork, is_private, license_spdx) "
            "VALUES (:id, :name, :owner, :url, false, false, NULL) "
            "ON CONFLICT (name) DO UPDATE SET license_spdx = NULL"
        ), {"id": repo_id, "name": "test-license-repo", "owner": "testowner", "url": "https://github.com/testowner/test-license-repo"})
        await session.commit()

    # Mock httpx to return MIT license
    async def mock_get(self, url, **kwargs):
        return _make_gh_response("MIT")

    with patch.object(httpx.AsyncClient, "get", mock_get):
        resp = await client.post(
            "/admin/backfill-licenses",
            headers={"Authorization": f"Bearer {TEST_API_KEY}"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] >= 1
    assert data["updated"] >= 1
    assert data["dry_run"] is False

    # Verify the DB was updated
    async with db_module.async_session_factory() as session:
        row = await session.execute(
            text("SELECT license_spdx FROM repos WHERE id = :id"),
            {"id": repo_id},
        )
        spdx = row.scalar()
        assert spdx == "MIT"

    # Clean up
    async with db_module.async_session_factory() as session:
        await session.execute(text("DELETE FROM repos WHERE id = :id"), {"id": repo_id})
        await session.commit()


@pytest.mark.asyncio
async def test_backfill_licenses_skips_noassertion(client: AsyncClient):
    """Repos where GitHub returns NOASSERTION should be skipped, not updated."""
    from sqlalchemy import text

    import app.database as db_module

    repo_id = str(uuid.uuid4())
    async with db_module.async_session_factory() as session:
        await session.execute(text(
            "INSERT INTO repos (id, name, owner, github_url, is_fork, is_private, license_spdx) "
            "VALUES (:id, :name, :owner, :url, false, false, NULL) "
            "ON CONFLICT (name) DO UPDATE SET license_spdx = NULL"
        ), {"id": repo_id, "name": "test-noassertion-repo", "owner": "testowner", "url": "https://github.com/testowner/test-noassertion-repo"})
        await session.commit()

    async def mock_get(self, url, **kwargs):
        return _make_gh_response("NOASSERTION")

    with patch.object(httpx.AsyncClient, "get", mock_get):
        resp = await client.post(
            "/admin/backfill-licenses",
            headers={"Authorization": f"Bearer {TEST_API_KEY}"},
        )

    assert resp.status_code == 200
    data = resp.json()
    # NOASSERTION repos should be skipped, not updated
    assert data["skipped"] >= 1

    # Verify DB still NULL
    async with db_module.async_session_factory() as session:
        row = await session.execute(
            text("SELECT license_spdx FROM repos WHERE id = :id"),
            {"id": repo_id},
        )
        spdx = row.scalar()
        assert spdx is None

    # Clean up
    async with db_module.async_session_factory() as session:
        await session.execute(text("DELETE FROM repos WHERE id = :id"), {"id": repo_id})
        await session.commit()


@pytest.mark.asyncio
async def test_backfill_licenses_handles_404(client: AsyncClient):
    """Repos not found on GitHub (404) should be skipped gracefully."""
    from sqlalchemy import text

    import app.database as db_module

    repo_id = str(uuid.uuid4())
    async with db_module.async_session_factory() as session:
        await session.execute(text(
            "INSERT INTO repos (id, name, owner, github_url, is_fork, is_private, license_spdx) "
            "VALUES (:id, :name, :owner, :url, false, false, NULL) "
            "ON CONFLICT (name) DO UPDATE SET license_spdx = NULL"
        ), {"id": repo_id, "name": "test-404-repo", "owner": "testowner", "url": "https://github.com/testowner/test-404-repo"})
        await session.commit()

    async def mock_get(self, url, **kwargs):
        return _make_gh_response(None, status=404)

    with patch.object(httpx.AsyncClient, "get", mock_get):
        resp = await client.post(
            "/admin/backfill-licenses",
            headers={"Authorization": f"Bearer {TEST_API_KEY}"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["skipped"] >= 1

    # Clean up
    async with db_module.async_session_factory() as session:
        await session.execute(text("DELETE FROM repos WHERE id = :id"), {"id": repo_id})
        await session.commit()


# ---------------------------------------------------------------------------
# Response shape
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_backfill_licenses_response_shape(client: AsyncClient):
    """Response contains all expected keys with correct types."""
    async def mock_get(self, url, **kwargs):
        return _make_gh_response("Apache-2.0")

    with patch.object(httpx.AsyncClient, "get", mock_get):
        resp = await client.post(
            "/admin/backfill-licenses",
            headers={"Authorization": f"Bearer {TEST_API_KEY}"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert set(data.keys()) == {"total", "updated", "failed", "skipped", "dry_run"}
    assert isinstance(data["total"], int)
    assert isinstance(data["updated"], int)
    assert isinstance(data["failed"], int)
    assert isinstance(data["skipped"], int)
    assert isinstance(data["dry_run"], bool)
