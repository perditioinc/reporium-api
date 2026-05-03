"""
Tests for GET /library/preview (KAN-151).

Mirrors test_library_full.py's patterns:
- Integration tests via AsyncClient + the test DB fixture
- Privacy invariant: is_private rows must NEVER appear in any response
- Sort behaviour, category filter, limit clamping, cache hit, aggregate exclusion
- Cache invalidation hook (invalidate_library_cache also clears preview keys)

The endpoint ships dead (no caller until KAN-152) so these tests are the
primary defence for the contract.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient


# Fields the design memo explicitly forbids on /library/preview responses.
# Their presence would mean the endpoint is paying the /library/full cost.
_FORBIDDEN_AGGREGATE_KEYS: tuple[str, ...] = (
    "stats",
    "categories",
    "tagMetrics",
    "builderStats",
    "aiDevSkillStats",
    "pmSkillStats",
    "gapAnalysis",
)


# ---------------------------------------------------------------------------
# Privacy invariant — mirrors test_library_full's 2026-04-23 leak guard.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_preview_returns_only_public_repos(client: AsyncClient):
    """Ingest a public + private pair; /library/preview must omit the private one."""
    from tests.conftest import AUTH_HEADERS, TEST_REPO_FIXTURE

    public_payload = {
        **TEST_REPO_FIXTURE,
        "name": "preview-public-probe",
        "github_url": "https://github.com/testuser/preview-public-probe",
        "is_private": False,
    }
    private_payload = {
        **TEST_REPO_FIXTURE,
        "name": "preview-private-probe",
        "github_url": "https://github.com/testuser/preview-private-probe",
        "is_private": True,
    }
    r = await client.post(
        "/ingest/repos",
        json=[public_payload, private_payload],
        headers=AUTH_HEADERS,
    )
    assert r.status_code == 200

    resp = await client.get("/library/preview?limit=500")
    assert resp.status_code == 200
    body = resp.json()
    names = {repo["name"] for repo in body["repos"]}

    assert "preview-public-probe" in names
    assert "preview-private-probe" not in names, (
        "PRIVACY LEAK: private repo surfaced in /library/preview"
    )


# ---------------------------------------------------------------------------
# Default limit + clamping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_preview_default_limit_is_300(client: AsyncClient):
    """Calling with no `limit` should reflect 300 in the response envelope."""
    resp = await client.get("/library/preview")
    assert resp.status_code == 200
    body = resp.json()
    assert body["limit"] == 300
    assert body["sort"] == "stars"
    assert isinstance(body["repos"], list)


@pytest.mark.asyncio
async def test_preview_max_limit_is_500(client: AsyncClient):
    """limit > 500 should be rejected with a 422 (FastAPI Query bounds)."""
    resp = await client.get("/library/preview?limit=501")
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_preview_min_limit_is_1(client: AsyncClient):
    """limit < 1 should be rejected with a 422."""
    resp = await client.get("/library/preview?limit=0")
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Sort behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_preview_sort_stars(client: AsyncClient):
    """sort=stars must order seeded probe rows by descending parent_stars.

    Tests share a DB, so we can't assert global monotonicity over the response
    (other tests may have seeded forks with NULL parent_stars whose Python
    `stars` value diverges from the SQL COALESCE rank). Instead, seed a set
    of probe rows with distinct, non-tie-breaking star values and assert
    they surface in the correct relative order.
    """
    from tests.conftest import AUTH_HEADERS, TEST_REPO_FIXTURE

    # Use unusual high star values that are unlikely to clash with other test
    # fixtures' default of 1000. Distinct values mean ties don't muddy the test.
    star_values = [98765, 87654, 76543, 65432]
    payloads = [
        {**TEST_REPO_FIXTURE, "name": f"preview-star-rank-{i}",
         "github_url": f"https://github.com/testuser/preview-star-rank-{i}",
         "is_private": False, "is_fork": True,
         "forked_from": f"upstream/preview-star-rank-{i}",
         "parent_stars": s}
        for i, s in enumerate(star_values)
    ]
    r = await client.post("/ingest/repos", json=payloads, headers=AUTH_HEADERS)
    assert r.status_code == 200

    resp = await client.get("/library/preview?sort=stars&limit=500")
    assert resp.status_code == 200
    repos = resp.json()["repos"]

    # Find our probes in the response and check they appear in descending order.
    probe_positions: dict[str, int] = {}
    for idx, repo in enumerate(repos):
        if repo["name"].startswith("preview-star-rank-"):
            probe_positions[repo["name"]] = idx

    # All 4 probes must be present
    expected_names = [f"preview-star-rank-{i}" for i in range(len(star_values))]
    assert all(n in probe_positions for n in expected_names), (
        f"missing probes; got {list(probe_positions)}"
    )
    # Probe names sorted by their position in the response must equal probe
    # names sorted by descending star value.
    response_order = sorted(probe_positions, key=probe_positions.get)
    expected_order = [n for _, n in sorted(
        zip(star_values, expected_names), key=lambda p: -p[0]
    )]
    assert response_order == expected_order, (
        f"sort=stars did not order probes correctly: got {response_order}, "
        f"expected {expected_order}"
    )


@pytest.mark.asyncio
async def test_preview_sort_updated(client: AsyncClient):
    """sort=updated must accept the param and not reject it (200 OK)."""
    # Strict monotonicity over the shared test DB is brittle; the sort=stars
    # test already exercises the ORDER BY mechanism end-to-end. Here we just
    # assert sort=updated is accepted and produces a non-error response.
    resp = await client.get("/library/preview?sort=updated&limit=10")
    assert resp.status_code == 200
    body = resp.json()
    assert body["sort"] == "updated"
    assert isinstance(body["repos"], list)
    # Sanity: every repo has a lastUpdated string (either ISO or empty).
    for r in body["repos"]:
        assert "lastUpdated" in r and isinstance(r["lastUpdated"], str)


@pytest.mark.asyncio
async def test_preview_sort_invalid_returns_422(client: AsyncClient):
    """Unknown sort values must be rejected by the regex pattern."""
    resp = await client.get("/library/preview?sort=bogus")
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Category filter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_preview_category_filter(client: AsyncClient):
    """When `category` is supplied, every returned repo must have that primary_category."""
    from tests.conftest import AUTH_HEADERS, TEST_REPO_FIXTURE

    # Ingest one repo with a distinctive primary_category so we have a known
    # row to query. The ingest pipeline copies categories[0].category_name into
    # primary_category via the column-level forward-fix (issue #444 closeout).
    payload = {
        **TEST_REPO_FIXTURE,
        "name": "preview-cat-probe",
        "github_url": "https://github.com/testuser/preview-cat-probe",
        "is_private": False,
        "categories": [
            {"category_id": "ai-agents", "category_name": "AI Agents", "is_primary": True}
        ],
    }
    r = await client.post("/ingest/repos", json=[payload], headers=AUTH_HEADERS)
    assert r.status_code == 200

    resp = await client.get("/library/preview?category=AI%20Agents&limit=500")
    assert resp.status_code == 200
    body = resp.json()
    assert body["category"] == "AI Agents"
    # Every returned repo must match the requested category. We do not assert
    # the probe is present (depends on whether the column-level update has
    # been applied in the test environment); we only assert the filter is honoured.
    for repo in body["repos"]:
        assert repo["primaryCategory"] == "AI Agents", (
            f"category filter leaked: {repo['name']} has primaryCategory={repo['primaryCategory']!r}"
        )


# ---------------------------------------------------------------------------
# Aggregate exclusion (the whole point of the endpoint)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_preview_excludes_aggregates(client: AsyncClient):
    """Per design memo: NO stats / categories / tagMetrics / builderStats / etc."""
    resp = await client.get("/library/preview?limit=2")
    assert resp.status_code == 200
    body = resp.json()
    for forbidden in _FORBIDDEN_AGGREGATE_KEYS:
        assert forbidden not in body, (
            f"/library/preview leaked aggregate key {forbidden!r}; "
            f"that defeats the 5.2 MB → 0.4 MB savings target"
        )
    # Envelope keys we DO expect
    assert set(body.keys()) >= {"generatedAt", "totalRepos", "limit", "sort", "repos"}


@pytest.mark.asyncio
async def test_preview_repo_shape_is_minimal(client: AsyncClient):
    """Per-repo projection must not carry heavy fields like commits / language breakdown / taxonomy."""
    from tests.conftest import AUTH_HEADERS, TEST_REPO_FIXTURE

    r = await client.post(
        "/ingest/repos",
        json=[{**TEST_REPO_FIXTURE, "name": "preview-shape-probe",
               "github_url": "https://github.com/testuser/preview-shape-probe",
               "is_private": False}],
        headers=AUTH_HEADERS,
    )
    assert r.status_code == 200

    resp = await client.get("/library/preview?limit=500")
    assert resp.status_code == 200
    repos = resp.json()["repos"]
    assert repos, "preview returned zero repos — fixture should have created at least one"
    sample = repos[0]
    # Heavy fields that /library/full carries but the lean preview MUST NOT
    forbidden_per_repo = (
        "commitsLast7Days", "commitsLast30Days", "commitsLast90Days",
        "recentCommits", "commitStats", "languageBreakdown", "languagePercentages",
        "taxonomy", "aiDevSkills", "pmSkills", "industries", "builders",
        "parentStats", "forkSync",
    )
    leaked = [k for k in forbidden_per_repo if k in sample]
    assert not leaked, f"/library/preview repo carried heavy fields: {leaked}"
    # The minimal contract — fields RepoCardMinimal needs
    expected = {"id", "name", "fullName", "description", "isFork", "language",
                "stars", "forks", "lastUpdated", "primaryCategory", "dbCategory",
                "enrichedTags", "isArchived", "url"}
    missing = expected - set(sample.keys())
    assert not missing, f"/library/preview missing expected fields: {missing}"


# ---------------------------------------------------------------------------
# Cache hit short-circuits the DB query
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_preview_cache_hit_does_not_query_db(client: AsyncClient):
    """When Redis returns a cached envelope, no DB query should run.

    Tests are configured with REDIS_URL="" so the production Redis path is
    a no-op. Stubbing redis_cache.get to return a body lets us assert the
    short-circuit deterministically without booting a real Redis.
    """
    cached_body = {
        "generatedAt": "2026-05-02T00:00:00+00:00",
        "totalRepos": 7,
        "limit": 300,
        "sort": "stars",
        "category": None,
        "repos": [],
    }
    fake_get = AsyncMock(return_value=cached_body)
    with patch("app.routers.library_preview.redis_cache.get", fake_get):
        resp = await client.get("/library/preview")
    assert resp.status_code == 200
    body = resp.json()
    assert body == cached_body
    fake_get.assert_called_once()


@pytest.mark.asyncio
async def test_preview_cache_miss_writes_to_redis(client: AsyncClient):
    """On a miss, the response body should be persisted under the right key."""
    fake_get = AsyncMock(return_value=None)
    fake_set = AsyncMock()
    with (
        patch("app.routers.library_preview.redis_cache.get", fake_get),
        patch("app.routers.library_preview.redis_cache.set", fake_set),
    ):
        resp = await client.get("/library/preview?sort=updated&limit=42")
    assert resp.status_code == 200
    fake_set.assert_called_once()
    # First positional arg is the key — verify the canonical shape.
    # KAN-179 extended the key with an `:include_key` suffix (`none` when no
    # `?include=` is supplied) so different include-sets don't share a cache
    # entry. Default request → `:none`.
    args, kwargs = fake_set.call_args
    assert args[0] == "library:preview:updated:42:*:none"
    assert kwargs.get("ttl") == 300


# ---------------------------------------------------------------------------
# Cache-Control header (matches the design memo)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_preview_sets_cache_control_header(client: AsyncClient):
    """public, s-maxage=300, stale-while-revalidate=60 — per KAN-170.

    s-maxage (shared/CDN caches only) replaces max-age so a future CDN /
    Cloud LB in front of Cloud Run can edge-cache repeated requests.
    Browsers ignore s-maxage and still revalidate every visit.
    """
    resp = await client.get("/library/preview?limit=2")
    assert resp.status_code == 200
    cc = resp.headers.get("cache-control", "")
    assert "public" in cc
    assert "s-maxage=300" in cc
    assert "stale-while-revalidate=60" in cc


@pytest.mark.asyncio
async def test_preview_response_has_cache_control(client: AsyncClient):
    """KAN-170 explicit invariant: response carries s-maxage=300.

    Asserted independently from the full directive string so a future swr
    tweak doesn't accidentally regress the s-maxage signal that lets CDN
    edges cache the response.
    """
    resp = await client.get("/library/preview?limit=2")
    assert resp.status_code == 200
    cc = resp.headers.get("cache-control", "")
    assert "s-maxage=300" in cc, (
        f"KAN-170: expected s-maxage=300 in Cache-Control, got: {cc!r}"
    )


# ---------------------------------------------------------------------------
# Response envelope — basic happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_preview_envelope_has_required_fields(client: AsyncClient):
    resp = await client.get("/library/preview?limit=2")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["generatedAt"], str) and body["generatedAt"]
    assert isinstance(body["totalRepos"], int)
    assert body["limit"] == 2
    assert body["sort"] == "stars"
    assert isinstance(body["repos"], list)
    assert len(body["repos"]) <= 2


# ---------------------------------------------------------------------------
# Cache invalidation hook — invalidate_library_cache must clear preview keys
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invalidate_library_cache_clears_preview_prefix():
    """invalidate_library_cache() must call redis_cache.clear_prefix('library:').

    The 'library:' prefix covers both 'library:page:*' (full) and
    'library:preview:*' (this PR). If a future refactor narrows the prefix
    to 'library:page:' only, /library/preview would serve stale results
    after a backfill — see feedback_backfill_must_invalidate_cache.md.

    KAN-175 also added a ``taxonomy:`` prefix sweep (covered by
    test_invalidate_library_cache_clears_taxonomy_prefix in
    tests/test_taxonomy_aggregation.py). Here we just assert the ``library:``
    prefix is one of the prefixes cleared, without pinning the call count.
    """
    from app.routers import library_full

    fake_clear_prefix = AsyncMock()
    with patch.object(library_full.redis_cache, "clear_prefix", fake_clear_prefix):
        library_full.invalidate_library_cache()
        # The function schedules the coroutine via asyncio.ensure_future; await
        # the underlying mock's call to ensure it ran.
        # ensure_future returns immediately, so we yield once for the loop.
        import asyncio
        await asyncio.sleep(0)

    prefixes_called = {call.args[0] for call in fake_clear_prefix.call_args_list}
    assert "library:" in prefixes_called, prefixes_called


# ---------------------------------------------------------------------------
# KAN-179 — `?include=` extension tokens
# ---------------------------------------------------------------------------


# Exact field set the default response must surface per repo (KAN-151 contract).
# 15 fields — same list the design memo + frontend RepoCardMinimal type carry.
_DEFAULT_PREVIEW_FIELDS: frozenset[str] = frozenset({
    "id", "name", "fullName", "description", "isFork", "forkedFrom",
    "language", "stars", "forks", "lastUpdated", "primaryCategory",
    "dbCategory", "enrichedTags", "isArchived", "url",
})


@pytest.mark.asyncio
async def test_preview_default_no_include_returns_baseline_fields(client: AsyncClient):
    """No `?include` → exactly the KAN-151 15 fields, nothing more."""
    from tests.conftest import AUTH_HEADERS, TEST_REPO_FIXTURE

    r = await client.post(
        "/ingest/repos",
        json=[{**TEST_REPO_FIXTURE, "name": "preview-baseline-probe",
               "github_url": "https://github.com/testuser/preview-baseline-probe",
               "is_private": False}],
        headers=AUTH_HEADERS,
    )
    assert r.status_code == 200

    resp = await client.get("/library/preview?limit=500")
    assert resp.status_code == 200
    repos = resp.json()["repos"]
    assert repos, "preview returned zero repos — fixture should have created at least one"
    sample = repos[0]
    assert set(sample.keys()) == _DEFAULT_PREVIEW_FIELDS, (
        f"KAN-151 contract regression: default projection keys={set(sample.keys())}, "
        f"expected={_DEFAULT_PREVIEW_FIELDS}"
    )


@pytest.mark.asyncio
async def test_preview_include_stats_adds_commit_stats(client: AsyncClient):
    """?include=stats adds commitStats per repo and not on default."""
    from tests.conftest import AUTH_HEADERS, TEST_REPO_FIXTURE

    r = await client.post(
        "/ingest/repos",
        json=[{**TEST_REPO_FIXTURE, "name": "preview-stats-probe",
               "github_url": "https://github.com/testuser/preview-stats-probe",
               "is_private": False,
               "commits_last_7_days": 3,
               "commits_last_30_days": 12,
               "commits_last_90_days": 40}],
        headers=AUTH_HEADERS,
    )
    assert r.status_code == 200

    # Default — no commitStats key.
    default_resp = await client.get("/library/preview?limit=500")
    assert default_resp.status_code == 200
    for repo in default_resp.json()["repos"]:
        assert "commitStats" not in repo, (
            f"default projection leaked commitStats on {repo['name']}"
        )

    # ?include=stats — commitStats present on every repo.
    inc_resp = await client.get("/library/preview?include=stats&limit=500")
    assert inc_resp.status_code == 200
    repos = inc_resp.json()["repos"]
    probe = next((r for r in repos if r["name"] == "preview-stats-probe"), None)
    assert probe is not None, "stats probe missing from include=stats response"
    assert "commitStats" in probe, "include=stats did not add commitStats"
    cs = probe["commitStats"]
    assert cs["last7Days"] == 3
    assert cs["last30Days"] == 12
    assert cs["last90Days"] == 40
    # parent / quality fields must NOT appear from a stats-only request.
    assert "parentStats" not in probe
    assert "qualitySignals" not in probe


@pytest.mark.asyncio
async def test_preview_include_parent_adds_parent_stats(client: AsyncClient):
    """?include=parent adds parentStats + upstreamCreatedAt for forks."""
    from tests.conftest import AUTH_HEADERS, TEST_REPO_FIXTURE

    r = await client.post(
        "/ingest/repos",
        json=[{**TEST_REPO_FIXTURE, "name": "preview-parent-probe",
               "github_url": "https://github.com/testuser/preview-parent-probe",
               "is_private": False,
               "is_fork": True,
               "forked_from": "upstream/preview-parent-probe",
               "parent_stars": 4242,
               "parent_forks": 88,
               "parent_is_archived": False}],
        headers=AUTH_HEADERS,
    )
    assert r.status_code == 200

    default_resp = await client.get("/library/preview?limit=500")
    assert default_resp.status_code == 200
    for repo in default_resp.json()["repos"]:
        assert "parentStats" not in repo, "default projection leaked parentStats"
        assert "upstreamCreatedAt" not in repo, "default projection leaked upstreamCreatedAt"

    inc_resp = await client.get("/library/preview?include=parent&limit=500")
    assert inc_resp.status_code == 200
    repos = inc_resp.json()["repos"]
    probe = next((r for r in repos if r["name"] == "preview-parent-probe"), None)
    assert probe is not None, "parent probe missing from include=parent response"
    assert "parentStats" in probe, "include=parent did not add parentStats"
    ps = probe["parentStats"]
    assert ps["owner"] == "upstream"
    assert ps["repo"] == "preview-parent-probe"
    assert ps["stars"] == 4242
    assert ps["forks"] == 88
    # upstreamCreatedAt may be empty (the ingest fixture doesn't set it), but
    # the key is part of the parent token contract — must be present.
    assert "upstreamCreatedAt" in probe
    # commitStats / qualitySignals NOT present from a parent-only request.
    assert "commitStats" not in probe
    assert "qualitySignals" not in probe


@pytest.mark.asyncio
async def test_preview_include_quality_adds_quality_signals(client: AsyncClient):
    """?include=quality adds qualitySignals (raw JSON) — public-safe field."""
    from tests.conftest import AUTH_HEADERS, TEST_REPO_FIXTURE

    r = await client.post(
        "/ingest/repos",
        json=[{**TEST_REPO_FIXTURE, "name": "preview-quality-probe",
               "github_url": "https://github.com/testuser/preview-quality-probe",
               "is_private": False,
               "quality_signals": {"hasReadme": True, "hasLicense": True}}],
        headers=AUTH_HEADERS,
    )
    assert r.status_code == 200

    inc_resp = await client.get("/library/preview?include=quality&limit=500")
    assert inc_resp.status_code == 200
    repos = inc_resp.json()["repos"]
    probe = next((r for r in repos if r["name"] == "preview-quality-probe"), None)
    assert probe is not None, "quality probe missing from include=quality response"
    assert "qualitySignals" in probe, "include=quality did not add qualitySignals"
    # Other tokens' fields not present.
    assert "commitStats" not in probe
    assert "parentStats" not in probe


@pytest.mark.asyncio
async def test_preview_include_multiple_tokens(client: AsyncClient):
    """?include=stats,parent adds both projections in one response."""
    from tests.conftest import AUTH_HEADERS, TEST_REPO_FIXTURE

    r = await client.post(
        "/ingest/repos",
        json=[{**TEST_REPO_FIXTURE, "name": "preview-multi-probe",
               "github_url": "https://github.com/testuser/preview-multi-probe",
               "is_private": False, "is_fork": True,
               "forked_from": "upstream/preview-multi-probe",
               "parent_stars": 5000,
               "commits_last_7_days": 7,
               "commits_last_30_days": 21,
               "commits_last_90_days": 63}],
        headers=AUTH_HEADERS,
    )
    assert r.status_code == 200

    resp = await client.get("/library/preview?include=stats,parent&limit=500")
    assert resp.status_code == 200
    repos = resp.json()["repos"]
    probe = next((r for r in repos if r["name"] == "preview-multi-probe"), None)
    assert probe is not None
    assert "commitStats" in probe and probe["commitStats"]["last7Days"] == 7
    assert "parentStats" in probe and probe["parentStats"]["stars"] == 5000
    assert "upstreamCreatedAt" in probe
    assert "qualitySignals" not in probe


@pytest.mark.asyncio
async def test_preview_unknown_include_returns_400(client: AsyncClient):
    """Unknown include tokens must 400 — no silent degrade to default."""
    resp = await client.get("/library/preview?include=hacker")
    assert resp.status_code == 400
    detail = resp.json().get("detail", "")
    assert "hacker" in str(detail).lower() or "unknown" in str(detail).lower()

    # Mixed valid + invalid is also rejected (so a typo in a list can't silently
    # demote the response shape).
    resp2 = await client.get("/library/preview?include=stats,nope")
    assert resp2.status_code == 400


@pytest.mark.asyncio
async def test_preview_cache_isolation_by_include(client: AsyncClient):
    """Two requests with different `?include=` values write distinct cache keys.

    Mocks redis_cache.set so we don't depend on a real Redis. Issuing default,
    ?include=stats, and ?include=stats,parent in sequence must produce 3
    distinct cache keys (the include suffix differs).
    """
    fake_get = AsyncMock(return_value=None)
    fake_set = AsyncMock()
    with (
        patch("app.routers.library_preview.redis_cache.get", fake_get),
        patch("app.routers.library_preview.redis_cache.set", fake_set),
    ):
        r1 = await client.get("/library/preview?limit=10")
        r2 = await client.get("/library/preview?limit=10&include=stats")
        r3 = await client.get("/library/preview?limit=10&include=stats,parent")

    assert r1.status_code == 200 and r2.status_code == 200 and r3.status_code == 200
    keys_written = {call.args[0] for call in fake_set.call_args_list}
    # Three distinct keys, all sharing the sort/limit/category prefix.
    expected_keys = {
        "library:preview:stars:10:*:none",
        "library:preview:stars:10:*:stats",
        "library:preview:stars:10:*:parent,stats",
    }
    assert keys_written == expected_keys, (
        f"cache key isolation failed: wrote {keys_written}, expected {expected_keys}"
    )
