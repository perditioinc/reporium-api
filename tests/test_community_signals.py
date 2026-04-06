"""Tests for POST /admin/backfill-community-signals and helpers."""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from httpx import AsyncClient

from app.routers.admin import _fetch_community_signals, _parse_last_page
from tests.conftest import AUTH_HEADERS


# ---------------------------------------------------------------------------
# Unit tests for _parse_last_page
# ---------------------------------------------------------------------------


class TestParseLinkHeader:
    def test_last_page_present(self):
        header = (
            '<https://api.github.com/repos/o/n/contributors?per_page=1&page=2>; rel="next", '
            '<https://api.github.com/repos/o/n/contributors?per_page=1&page=47>; rel="last"'
        )
        assert _parse_last_page(header) == 47

    def test_no_last_rel(self):
        header = '<https://api.github.com/repos/o/n/contributors?per_page=1&page=2>; rel="next"'
        assert _parse_last_page(header) is None

    def test_none_header(self):
        assert _parse_last_page(None) is None

    def test_empty_string(self):
        assert _parse_last_page("") is None

    def test_single_page(self):
        # When all results fit in one page, GitHub omits the Link header entirely
        assert _parse_last_page(None) is None


# ---------------------------------------------------------------------------
# Unit tests for _fetch_community_signals
# ---------------------------------------------------------------------------


def _mock_response(status_code=200, json_data=None, headers=None):
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.headers = headers or {}
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=resp,
        )
    return resp


@pytest.mark.asyncio
async def test_fetch_signals_404_returns_none():
    """If the main repo endpoint returns 404, the helper should return None."""
    import asyncio

    client = AsyncMock(spec=httpx.AsyncClient)
    client.get = AsyncMock(return_value=_mock_response(404))

    result = await _fetch_community_signals(client, asyncio.Semaphore(1), "owner", "repo")
    assert result is None


@pytest.mark.asyncio
async def test_fetch_signals_success():
    """Happy path: all sub-requests succeed."""
    import asyncio

    repo_resp = _mock_response(200, {
        "has_discussions": True,
        "open_issues_count": 10,
    })

    contrib_resp = _mock_response(200, [{"id": 1}], {
        "link": '<https://api.github.com/repos/o/n/contributors?per_page=1&page=25>; rel="last"',
    })

    release_resp = _mock_response(200, [{"published_at": "2026-01-15T00:00:00Z"}], {
        "link": '<https://api.github.com/repos/o/n/releases?per_page=1&page=8>; rel="last"',
    })

    community_resp = _mock_response(200, {"health_percentage": 85})

    closed_issues_resp = _mock_response(200, {"total_count": 40})
    merged_pr_resp = _mock_response(200, {"total_count": 30})
    total_pr_resp = _mock_response(200, {"total_count": 50})

    call_index = 0
    responses = [
        repo_resp,       # GET /repos/{owner}/{name}
        contrib_resp,    # GET /repos/.../contributors
        release_resp,    # GET /repos/.../releases
        community_resp,  # GET /repos/.../community/profile
        closed_issues_resp,  # search/issues (closed issues)
        merged_pr_resp,      # search/issues (merged PRs)
        total_pr_resp,       # search/issues (total PRs)
    ]

    async def mock_get(*args, **kwargs):
        nonlocal call_index
        idx = call_index
        call_index += 1
        return responses[idx]

    client = AsyncMock(spec=httpx.AsyncClient)
    client.get = mock_get

    result = await _fetch_community_signals(client, asyncio.Semaphore(1), "owner", "repo")

    assert result is not None
    assert result["has_discussions"] is True
    assert result["contributors_count"] == 25
    assert result["release_count"] == 8
    assert result["latest_release_date"] == "2026-01-15T00:00:00Z"
    assert result["community_health_pct"] == 85
    assert result["issue_close_rate"] == round(40 / 50, 4)
    assert result["pr_merge_rate"] == round(30 / 50, 4)


@pytest.mark.asyncio
async def test_fetch_signals_no_link_header_falls_back():
    """When no Link header exists, use length of the returned JSON array."""
    import asyncio

    repo_resp = _mock_response(200, {"has_discussions": False, "open_issues_count": 0})
    contrib_resp = _mock_response(200, [{"id": 1}, {"id": 2}, {"id": 3}])
    release_resp = _mock_response(200, [])
    community_resp = _mock_response(200, {"health_percentage": 40})
    closed_issues_resp = _mock_response(200, {"total_count": 0})
    merged_pr_resp = _mock_response(200, {"total_count": 0})
    total_pr_resp = _mock_response(200, {"total_count": 0})

    call_index = 0
    responses = [repo_resp, contrib_resp, release_resp, community_resp,
                 closed_issues_resp, merged_pr_resp, total_pr_resp]

    async def mock_get(*args, **kwargs):
        nonlocal call_index
        idx = call_index
        call_index += 1
        return responses[idx]

    client = AsyncMock(spec=httpx.AsyncClient)
    client.get = mock_get

    result = await _fetch_community_signals(client, asyncio.Semaphore(1), "o", "n")
    assert result["contributors_count"] == 3
    assert result["release_count"] == 0


@pytest.mark.asyncio
async def test_fetch_signals_exception_returns_none():
    """Network errors should be caught and return None."""
    import asyncio

    client = AsyncMock(spec=httpx.AsyncClient)
    client.get = AsyncMock(side_effect=httpx.ConnectError("connection refused"))

    result = await _fetch_community_signals(client, asyncio.Semaphore(1), "o", "n")
    assert result is None


# ---------------------------------------------------------------------------
# Integration-style tests (via test client)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_backfill_community_signals_requires_auth(client: AsyncClient):
    """Endpoint should reject unauthenticated requests."""
    resp = await client.post("/admin/backfill-community-signals")
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_backfill_community_signals_dry_run(client: AsyncClient):
    """dry_run should return counts without modifying the database."""
    resp = await client.post(
        "/admin/backfill-community-signals?dry_run=true",
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "total" in data
    assert "updated" in data
    assert "failed" in data
    assert "skipped" in data
    assert data["dry_run"] is True
    assert data["updated"] == 0


@pytest.mark.asyncio
async def test_backfill_community_signals_batch_size(client: AsyncClient):
    """batch_size param should be accepted and validated."""
    resp = await client.post(
        "/admin/backfill-community-signals?dry_run=true&batch_size=5",
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["dry_run"] is True


@pytest.mark.asyncio
async def test_backfill_community_signals_invalid_batch_size(client: AsyncClient):
    """batch_size > 500 should be rejected."""
    resp = await client.post(
        "/admin/backfill-community-signals?batch_size=999",
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 422
