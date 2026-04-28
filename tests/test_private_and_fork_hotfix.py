"""Hotfix regression tests — 2026-04-28 P0.

Two related bugs landed at the same time:

1. Private repo `perditioinc/hippo-harvest-assignment` was returned by
   GET /repos/perditioinc/hippo-harvest-assignment with HTTP 200 — see
   .audit/2026-04-28/api-private-and-fork-hotfix.md for the live curl
   evidence and the contradiction investigation.

   The hotfix centralizes the visibility predicate as
   `app.db_filters.public_repo_filter()` and routes every repo-facing
   query through it. These tests pin the contract so a future endpoint
   that forgets the predicate fails CI immediately.

2. Smart-route SQL handlers in app/routers/intelligence.py hardcoded
   `forked_from: None` in their `sources` dicts even though the row had
   a real upstream parent. Users asking "Which repos support MCP?" got
   `perditioinc/markitdown` cited instead of `microsoft/markitdown`.

   The hotfix wires `forked_from` through every smart-route SQL and
   re-shapes the source dict via `_build_smart_route_source` /
   `app.source_canonical.canonical_owner_name`. These tests verify the
   canonical upstream name surfaces when forked_from is set, and that
   nothing changes when forked_from is null.

The fixture ingests two probe repos:
  - public-probe-{slug}: is_fork=False, forked_from=None, is_private=False
  - private-probe-{slug}: is_fork=False, forked_from=None, is_private=True
  - fork-probe-{slug}:    is_fork=True,  forked_from=upstream/probe-X
                            is_private=False
The slug is per-test so concurrent tests don't trip uniqueness.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from app.source_canonical import canonical_owner_name
from tests.conftest import AUTH_HEADERS, TEST_REPO_FIXTURE


# ---------------------------------------------------------------------------
# canonical_owner_name unit tests — pure-function, no DB
# ---------------------------------------------------------------------------


def test_canonical_owner_name_uses_upstream_when_forked_from_set():
    """When `forked_from` is `<owner>/<name>`, canonical pivots to upstream."""
    owner, name = canonical_owner_name(
        forked_from="microsoft/markitdown",
        own_owner="perditioinc",
        own_name="markitdown",
    )
    assert (owner, name) == ("microsoft", "markitdown")


def test_canonical_owner_name_falls_back_when_forked_from_null():
    """When `forked_from` is None, canonical is the row's own identity."""
    owner, name = canonical_owner_name(
        forked_from=None,
        own_owner="perditioinc",
        own_name="hippo-harvest-assignment",
    )
    assert (owner, name) == ("perditioinc", "hippo-harvest-assignment")


def test_canonical_owner_name_falls_back_when_forked_from_empty():
    owner, name = canonical_owner_name(
        forked_from="",
        own_owner="perditioinc",
        own_name="thing",
    )
    assert (owner, name) == ("perditioinc", "thing")


def test_canonical_owner_name_falls_back_when_forked_from_malformed():
    """`forked_from` without "/" is treated as no fork — never invent a parent."""
    owner, name = canonical_owner_name(
        forked_from="malformed-no-slash",
        own_owner="perditioinc",
        own_name="thing",
    )
    assert (owner, name) == ("perditioinc", "thing")


def test_canonical_owner_name_falls_back_when_forked_from_only_slash():
    """`forked_from` of "/" yields empty parts — fall back."""
    owner, name = canonical_owner_name(
        forked_from="/",
        own_owner="perditioinc",
        own_name="thing",
    )
    assert (owner, name) == ("perditioinc", "thing")


def test_canonical_owner_name_strips_whitespace():
    """Whitespace around upstream parts is stripped."""
    owner, name = canonical_owner_name(
        forked_from="  microsoft / markitdown  ",
        own_owner="perditioinc",
        own_name="markitdown",
    )
    assert (owner, name) == ("microsoft", "markitdown")


# ---------------------------------------------------------------------------
# Repo visibility — list, owner-filter, detail (no-owner), detail (owner+name)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_repos_list_excludes_private(client: AsyncClient):
    """GET /repos must never include a private repo.

    Reproduces the live-API check from the 2026-04-28 audit: even though
    the per-endpoint predicate was correct, this test now anchors the
    invariant in code so any future regression flips CI red.
    """
    private_payload = {
        **TEST_REPO_FIXTURE,
        "name": "private-probe-list",
        "github_url": "https://github.com/testuser/private-probe-list",
        "is_fork": False,
        "forked_from": None,
        "is_private": True,
    }
    public_payload = {
        **TEST_REPO_FIXTURE,
        "name": "public-probe-list",
        "github_url": "https://github.com/testuser/public-probe-list",
        "is_fork": False,
        "forked_from": None,
        "is_private": False,
    }
    r = await client.post(
        "/ingest/repos",
        json=[private_payload, public_payload],
        headers=AUTH_HEADERS,
    )
    assert r.status_code == 200

    resp = await client.get("/repos?limit=200")
    assert resp.status_code == 200
    names = {r["name"] for r in resp.json()["repos"]}
    assert "public-probe-list" in names
    assert "private-probe-list" not in names, (
        "PRIVACY LEAK: private-probe-list surfaced in /repos — "
        "centralized public_repo_filter() is not being applied."
    )


@pytest.mark.asyncio
async def test_repos_detail_returns_404_for_private_repo(client: AsyncClient):
    """GET /repos/{owner}/{name} must return 404 for a private row.

    This is the exact bug from 2026-04-28: live API returned HTTP 200
    with the full row body for `perditioinc/hippo-harvest-assignment`.
    """
    private_payload = {
        **TEST_REPO_FIXTURE,
        "name": "private-probe-detail",
        "owner": "testuser",
        "github_url": "https://github.com/testuser/private-probe-detail",
        "is_fork": False,
        "forked_from": None,
        "is_private": True,
    }
    r = await client.post(
        "/ingest/repos", json=[private_payload], headers=AUTH_HEADERS
    )
    assert r.status_code == 200

    # Two-segment path — matches /repos/{owner}/{repo}
    resp = await client.get("/repos/testuser/private-probe-detail")
    assert resp.status_code == 404, (
        f"PRIVACY LEAK: /repos/testuser/private-probe-detail returned "
        f"{resp.status_code}, expected 404. This reproduces the "
        f"hippo-harvest-assignment leak from 2026-04-28."
    )


@pytest.mark.asyncio
async def test_repos_detail_single_segment_returns_404_for_private_repo(
    client: AsyncClient,
):
    """GET /repos/{name} (no owner) must also return 404 for private rows."""
    private_payload = {
        **TEST_REPO_FIXTURE,
        "name": "private-probe-single",
        "github_url": "https://github.com/testuser/private-probe-single",
        "is_fork": False,
        "forked_from": None,
        "is_private": True,
    }
    r = await client.post(
        "/ingest/repos", json=[private_payload], headers=AUTH_HEADERS
    )
    assert r.status_code == 200

    resp = await client.get("/repos/private-probe-single")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_search_excludes_private(client: AsyncClient):
    """GET /search must filter out private repos."""
    private_payload = {
        **TEST_REPO_FIXTURE,
        "name": "private-probe-search-uniquetoken",
        "description": "uniquetoken-private description for search test",
        "github_url": "https://github.com/testuser/private-probe-search-uniquetoken",
        "is_fork": False,
        "forked_from": None,
        "is_private": True,
    }
    public_payload = {
        **TEST_REPO_FIXTURE,
        "name": "public-probe-search-uniquetoken",
        "description": "uniquetoken-public description for search test",
        "github_url": "https://github.com/testuser/public-probe-search-uniquetoken",
        "is_fork": False,
        "forked_from": None,
        "is_private": False,
    }
    r = await client.post(
        "/ingest/repos",
        json=[private_payload, public_payload],
        headers=AUTH_HEADERS,
    )
    assert r.status_code == 200

    resp = await client.get("/search?q=uniquetoken")
    assert resp.status_code == 200
    names = {r["name"] for r in resp.json()}
    assert "public-probe-search-uniquetoken" in names
    assert "private-probe-search-uniquetoken" not in names, (
        "PRIVACY LEAK: private repo surfaced in /search."
    )


@pytest.mark.asyncio
async def test_library_excludes_private(client: AsyncClient):
    """/library response must not contain private repos."""
    private_payload = {
        **TEST_REPO_FIXTURE,
        "name": "private-probe-library",
        "github_url": "https://github.com/testuser/private-probe-library",
        "is_fork": False,
        "forked_from": None,
        "is_private": True,
    }
    r = await client.post(
        "/ingest/repos", json=[private_payload], headers=AUTH_HEADERS
    )
    assert r.status_code == 200

    resp = await client.get("/library?limit=500")
    assert resp.status_code == 200
    names = {r["name"] for r in resp.json()["repos"]}
    assert "private-probe-library" not in names, (
        "PRIVACY LEAK: private repo surfaced in /library."
    )


@pytest.mark.asyncio
async def test_forks_endpoint_excludes_private_fork(client: AsyncClient):
    """/forks (internal-use endpoint) must also filter private repos.

    Private forks are still private. The /forks endpoint was missing the
    is_private predicate before this hotfix — see audit doc for details.
    """
    private_fork_payload = {
        **TEST_REPO_FIXTURE,
        "name": "private-probe-fork",
        "github_url": "https://github.com/testuser/private-probe-fork",
        "is_fork": True,
        "forked_from": "upstream/private-probe-fork",
        "is_private": True,
    }
    public_fork_payload = {
        **TEST_REPO_FIXTURE,
        "name": "public-probe-fork",
        "github_url": "https://github.com/testuser/public-probe-fork",
        "is_fork": True,
        "forked_from": "upstream/public-probe-fork",
        "is_private": False,
    }
    r = await client.post(
        "/ingest/repos",
        json=[private_fork_payload, public_fork_payload],
        headers=AUTH_HEADERS,
    )
    assert r.status_code == 200

    resp = await client.get("/forks?limit=500")
    assert resp.status_code == 200
    names = {f["name"] for f in resp.json()["forks"]}
    assert "public-probe-fork" in names
    assert "private-probe-fork" not in names, (
        "PRIVACY LEAK: private fork surfaced in /forks."
    )


# ---------------------------------------------------------------------------
# ASK source canonicalization — fork rows cite upstream, originals stay as-is
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ask_smart_route_source_canonicalizes_fork(client: AsyncClient):
    """When the smart-route returns a fork row, sources cite the upstream.

    Uses the `_ROUTE_REPO_INFO` smart-route which fires on questions like
    "tell me about <repo-name>". The forked row gets ingested with
    `forked_from = "microsoft/markitdown-fork-probe"`; the cited source's
    `name`/`owner` should be those upstream values, with the row's own
    `forked_from` field still populated for transparency.
    """
    # Use a unique, contrived name so the smart-route blacklist (general
    # words like "ai", "python") never blocks our probe.
    fork_payload = {
        **TEST_REPO_FIXTURE,
        "name": "askprobeforkcanonical",
        "owner": "perditioinc",
        "github_url": "https://github.com/perditioinc/askprobeforkcanonical",
        "is_fork": True,
        "forked_from": "upstream-org/askprobeforkcanonical",
        "is_private": False,
    }
    r = await client.post(
        "/ingest/repos", json=[fork_payload], headers=AUTH_HEADERS
    )
    assert r.status_code == 200

    # Probe the smart-route _ROUTE_REPO_INFO directly — bypasses the LLM
    # path so the test stays fast and deterministic. We import the router
    # function and call it against a real DB session; this is the same
    # call the /intelligence/ask handler makes in production.
    from app.database import async_session_factory
    from app.routers.intelligence import _try_smart_route_inner

    async with async_session_factory() as db:
        result = await _try_smart_route_inner(
            "tell me about askprobeforkcanonical", db
        )

    assert result is not None, (
        "Smart-route _ROUTE_REPO_INFO did not match — adjust the probe name "
        "if the regex was tightened."
    )
    assert result["sources"], "Expected the route to attach a source row."
    src = result["sources"][0]
    # Canonical pivot: cite the upstream parent, NOT the perditioinc mirror.
    assert src["owner"] == "upstream-org", (
        f"FORK CANONICALIZATION REGRESSION: ASK source owner is "
        f"{src['owner']!r}, expected 'upstream-org'. The smart-route "
        f"hardcoded forked_from=None before the 2026-04-28 hotfix."
    )
    assert src["name"] == "askprobeforkcanonical"
    # forked_from is still populated for the client-side "(forked from X)" badge.
    assert src["forked_from"] == "upstream-org/askprobeforkcanonical"


@pytest.mark.asyncio
async def test_ask_smart_route_source_preserves_original_when_not_a_fork(
    client: AsyncClient,
):
    """Non-fork rows must keep their own owner/name in sources."""
    original_payload = {
        **TEST_REPO_FIXTURE,
        "name": "askprobeoriginal",
        "owner": "perditioinc",
        "github_url": "https://github.com/perditioinc/askprobeoriginal",
        "is_fork": False,
        "forked_from": None,
        "is_private": False,
    }
    r = await client.post(
        "/ingest/repos", json=[original_payload], headers=AUTH_HEADERS
    )
    assert r.status_code == 200

    from app.database import async_session_factory
    from app.routers.intelligence import _try_smart_route_inner

    async with async_session_factory() as db:
        result = await _try_smart_route_inner(
            "tell me about askprobeoriginal", db
        )

    assert result is not None
    assert result["sources"]
    src = result["sources"][0]
    assert src["owner"] == "perditioinc"
    assert src["name"] == "askprobeoriginal"
    assert src["forked_from"] is None
