"""Regression tests for /library (issue #344).

The /library response includes a `stats` object whose fields are named
`total_repos`, `total_forks`, `total_non_forks`, `languages`, etc. Historically
`total_forks`, `total_non_forks`, and `languages` were computed over the
paginated page (e.g. `sum(1 for r in repos if r.is_fork)`), which made
`total_forks` track `limit` instead of the true corpus count. A request of
`/library?limit=1` would return `stats.total_forks == 1`.

These tests pin the corpus-wide semantics so the fields can't drift again.
"""

import pytest
from httpx import AsyncClient

from tests.conftest import AUTH_HEADERS, TEST_REPO_FIXTURE


@pytest.mark.asyncio
async def test_library_stats_totals_are_corpus_wide_not_page_scoped(client: AsyncClient):
    fork_payloads = [
        {
            **TEST_REPO_FIXTURE,
            "name": f"lib-fork-{i}",
            "github_url": f"https://github.com/testuser/lib-fork-{i}",
            "is_fork": True,
            "is_private": False,
        }
        for i in range(3)
    ]
    non_fork_payload = {
        **TEST_REPO_FIXTURE,
        "name": "lib-builtbyme-1",
        "github_url": "https://github.com/testuser/lib-builtbyme-1",
        "is_fork": False,
        "forked_from": None,
        "is_private": False,
    }

    baseline = (await client.get("/library?limit=1")).json()["stats"]

    r = await client.post(
        "/ingest/repos",
        json=[*fork_payloads, non_fork_payload],
        headers=AUTH_HEADERS,
    )
    assert r.status_code == 200

    # Request with limit=1 — before the #344 fix this would return
    # total_forks=1 because the sum iterated only the paginated page.
    resp = await client.get("/library?limit=1")
    assert resp.status_code == 200
    stats = resp.json()["stats"]

    assert stats["total_forks"] == baseline["total_forks"] + 3
    assert stats["total_non_forks"] == baseline["total_non_forks"] + 1
    assert stats["total_repos"] == baseline["total_repos"] + 4
    assert stats["total_forks"] + stats["total_non_forks"] == stats["total_repos"]


@pytest.mark.asyncio
async def test_library_stats_languages_are_corpus_wide(client: AsyncClient):
    """languages dict on /library.stats must reflect the whole public corpus."""
    payload = {
        **TEST_REPO_FIXTURE,
        "name": "lib-rust-probe",
        "github_url": "https://github.com/testuser/lib-rust-probe",
        "primary_language": "Rust",
        "is_private": False,
    }

    baseline_rust = (
        (await client.get("/library?limit=1")).json()["stats"]["languages"].get("Rust", 0)
    )

    r = await client.post("/ingest/repos", json=[payload], headers=AUTH_HEADERS)
    assert r.status_code == 200

    # Paginated page of 1 should NOT cap the language bucket at 1.
    stats = (await client.get("/library?limit=1")).json()["stats"]
    assert stats["languages"].get("Rust", 0) == baseline_rust + 1


@pytest.mark.asyncio
async def test_library_stats_matches_stats_endpoint(client: AsyncClient):
    """/library.stats totals must agree with /stats on the same corpus."""
    stats_endpoint = (await client.get("/stats")).json()
    library_stats = (await client.get("/library?limit=1")).json()["stats"]

    assert library_stats["total_repos"] == stats_endpoint["total_repos"]
    assert library_stats["total_forks"] == stats_endpoint["total_forks"]
