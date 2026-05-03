"""
Tests for GET /library/aggregates (KAN-188).

Mirrors test_library_preview.py / test_library_full.py patterns:
- Lean response shape contract: aggregates fields PRESENT, no per-repo array
- Cache hit short-circuit
- Cache invalidation hook (invalidate_library_cache also clears the new key)
- Cache-Control header (CDN-friendly s-maxage=300, KAN-170 pattern)

Per the 4h perf audit P2: this endpoint exists so lean callers (StatsBar,
MetricsSidebar, LibraryInsightsWidget) can drop /library/full's 1.46 MB
per-request cost. Frontend migration of those callers is a follow-up.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient


# Aggregate fields the endpoint MUST return — same shape /library/full embeds.
_REQUIRED_AGGREGATE_KEYS: tuple[str, ...] = (
    "stats",
    "gapAnalysis",
    "tagMetrics",
    "categories",
    "builderStats",
    "aiDevSkillStats",
    "pmSkillStats",
)


# ---------------------------------------------------------------------------
# Lean response shape — the whole point of the endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aggregates_returns_all_fields(client: AsyncClient):
    """All aggregate fields plus envelope keys (generatedAt, totalRepos)."""
    resp = await client.get("/library/aggregates")
    assert resp.status_code == 200
    body = resp.json()

    # Envelope keys
    assert "generatedAt" in body
    assert isinstance(body["generatedAt"], str) and body["generatedAt"]
    assert "totalRepos" in body
    assert isinstance(body["totalRepos"], int)

    # Aggregate fields — every one must be present
    for k in _REQUIRED_AGGREGATE_KEYS:
        assert k in body, f"missing aggregate field {k!r}"


@pytest.mark.asyncio
async def test_aggregates_does_NOT_return_repos_array(client: AsyncClient):
    """The whole point of the lean endpoint: no per-repo array.

    /library/full ships ~1.46 MB warm largely because of the per-repo array.
    /library/aggregates exists so callers that only need the aggregate slice
    don't pay that cost. If `repos` ever leaks back in, the perf budget
    regresses by orders of magnitude.
    """
    resp = await client.get("/library/aggregates")
    assert resp.status_code == 200
    body = resp.json()
    assert "repos" not in body, (
        "/library/aggregates leaked the per-repo array — that defeats the "
        "1.46 MB → ~50-300 KiB savings target (KAN-188 4h audit P2)"
    )


@pytest.mark.asyncio
async def test_aggregates_tag_metrics_does_NOT_include_per_tag_repos_array(client: AsyncClient):
    """KAN-193 contract: tagMetrics entries MUST NOT carry a per-tag `repos[]` array.

    KAN-188 (PR #476) shipped /library/aggregates with the same tagMetrics
    shape /library/full embeds — including a per-tag `repos: [name1, ...]`
    array (capped to 20 names per tag). At ~5,329 tags this dominated the
    3.8 MB payload.

    Consumer audit (perditioinc): no production reader of tagMetric.repos
    in reporium frontend, reporium-mcp, reporium-evals, or reporium-audit.
    Callers needing tag → repos mapping should derive from per-repo
    `enrichedTags` in /library/preview or /library/full.
    """
    resp = await client.get("/library/aggregates")
    assert resp.status_code == 200
    body = resp.json()

    tag_metrics = body.get("tagMetrics") or []
    # If the corpus has any tags, at least one tagMetrics entry should be
    # present and we can check the shape. If empty (test fixtures with no
    # enriched tags), the assertion is trivially satisfied.
    for tm in tag_metrics:
        assert "repos" not in tm, (
            "KAN-193 regression: tagMetrics entry leaked the per-tag `repos` "
            "array. That field dominated the ~3.8 MB payload before KAN-193 "
            "and was dropped because no consumer reads it. If a real consumer "
            "now needs it, restore as a top-5 cap or revisit the design."
        )


@pytest.mark.asyncio
async def test_aggregates_field_shapes_match_library_full(client: AsyncClient):
    """Each aggregate field's runtime shape matches /library/full's wire shape.

    Both endpoints feed the exact same `_fetch_aggregates` builder pipeline,
    so this is mostly a smoke check that the migration didn't drop a field
    type.
    """
    resp = await client.get("/library/aggregates")
    assert resp.status_code == 200
    body = resp.json()

    # stats is a dict with total/built/forked/languages/topTags
    assert isinstance(body["stats"], dict)
    for k in ("total", "built", "forked"):
        assert k in body["stats"]
        assert isinstance(body["stats"][k], int)

    # gapAnalysis is {generatedAt, gaps[]}
    assert isinstance(body["gapAnalysis"], dict)
    assert "generatedAt" in body["gapAnalysis"]
    assert "gaps" in body["gapAnalysis"]
    assert isinstance(body["gapAnalysis"]["gaps"], list)

    # The list-shaped aggregates
    for k in ("tagMetrics", "categories", "builderStats",
              "aiDevSkillStats", "pmSkillStats"):
        assert isinstance(body[k], list), f"{k} must be a list, got {type(body[k]).__name__}"


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aggregates_cache_hit_short_circuits_db(client: AsyncClient):
    """When Redis returns a cached envelope, the endpoint short-circuits.

    Tests are configured with REDIS_URL="" so the production Redis path is
    a no-op. Stubbing redis_cache.get to return a body lets us assert the
    short-circuit deterministically without booting a real Redis.
    """
    cached_body = {
        "generatedAt": "2026-05-02T00:00:00+00:00",
        "totalRepos": 99,
        "stats": {"total": 99, "built": 5, "forked": 94, "languages": [], "topTags": []},
        "gapAnalysis": {"generatedAt": "2026-05-02T00:00:00+00:00", "gaps": []},
        "tagMetrics": [],
        "categories": [],
        "builderStats": [],
        "aiDevSkillStats": [],
        "pmSkillStats": [],
    }
    fake_get = AsyncMock(return_value=cached_body)
    with patch("app.routers.library_aggregates.redis_cache.get", fake_get):
        resp = await client.get("/library/aggregates")
    assert resp.status_code == 200
    body = resp.json()
    assert body == cached_body
    fake_get.assert_called_once()


@pytest.mark.asyncio
async def test_aggregates_cache_miss_writes_to_redis(client: AsyncClient):
    """On a miss, the response body is persisted under `library:aggregates:v1`."""
    fake_get = AsyncMock(return_value=None)
    fake_set = AsyncMock()
    with (
        patch("app.routers.library_aggregates.redis_cache.get", fake_get),
        patch("app.routers.library_aggregates.redis_cache.set", fake_set),
    ):
        resp = await client.get("/library/aggregates")
    assert resp.status_code == 200
    fake_set.assert_called_once()
    # First positional arg is the key — verify the KAN-188 canonical shape.
    args, kwargs = fake_set.call_args
    assert args[0] == "library:aggregates:v1"
    assert kwargs.get("ttl") == 300


# ---------------------------------------------------------------------------
# Cache invalidation hook — invalidate_library_cache must clear the new key
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aggregates_cache_invalidated_on_invalidate_library_cache():
    """invalidate_library_cache() clears the `library:` prefix, which covers
    `library:aggregates:v1` along with /library/full + /library/preview keys.

    KAN-188 piggybacks on the existing prefix-sweep (which has been live since
    KAN-151) rather than enumerating exact keys. A future refactor that
    narrows the prefix to `library:page:` only would silently leave aggregates
    stale after a backfill — exactly the failure mode covered by
    feedback_backfill_must_invalidate_cache.md. This test pins the contract.
    """
    from app.routers import library_full

    fake_clear_prefix = AsyncMock()
    with patch.object(library_full.redis_cache, "clear_prefix", fake_clear_prefix):
        library_full.invalidate_library_cache()
        # ensure_future schedules the coroutine; yield once for the loop.
        import asyncio
        await asyncio.sleep(0)

    prefixes_called = {call.args[0] for call in fake_clear_prefix.call_args_list}
    assert "library:" in prefixes_called, (
        f"expected `library:` prefix sweep to cover library:aggregates:v1, "
        f"got {prefixes_called}"
    )


# ---------------------------------------------------------------------------
# Cache-Control header (KAN-170 / matching /library/preview)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aggregates_response_has_cache_control(client: AsyncClient):
    """public, s-maxage=300, stale-while-revalidate=60 — KAN-170 pattern.

    s-maxage replaces max-age so a future CDN/Cloud LB in front of Cloud Run
    can edge-cache repeated requests; browsers ignore s-maxage and still
    revalidate every visit.
    """
    resp = await client.get("/library/aggregates")
    assert resp.status_code == 200
    cc = resp.headers.get("cache-control", "")
    assert "public" in cc
    assert "s-maxage=300" in cc, (
        f"KAN-170: expected s-maxage=300 in Cache-Control, got: {cc!r}"
    )
    assert "stale-while-revalidate=60" in cc


# ---------------------------------------------------------------------------
# Sanity: privacy filter — totalRepos counts public corpus only
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aggregates_total_repos_excludes_private(client: AsyncClient):
    """totalRepos must mirror /library/full's count semantics: public only.

    Same `WHERE is_private = false` predicate that protects /library/full
    and /library/preview (app.db_filters.PUBLIC_REPO_SQL_PREDICATE).
    """
    from tests.conftest import AUTH_HEADERS, TEST_REPO_FIXTURE

    baseline = (await client.get("/library/aggregates")).json()
    baseline_total = baseline["totalRepos"]
    baseline_stats_total = baseline["stats"]["total"]

    payloads = [
        {**TEST_REPO_FIXTURE, "name": "agg-public-1",
         "github_url": "https://github.com/testuser/agg-public-1", "is_private": False},
        {**TEST_REPO_FIXTURE, "name": "agg-public-2",
         "github_url": "https://github.com/testuser/agg-public-2", "is_private": False},
        {**TEST_REPO_FIXTURE, "name": "agg-private-1",
         "github_url": "https://github.com/testuser/agg-private-1", "is_private": True},
    ]
    r = await client.post("/ingest/repos", json=payloads, headers=AUTH_HEADERS)
    assert r.status_code == 200

    resp = await client.get("/library/aggregates")
    assert resp.status_code == 200
    body = resp.json()

    # Only the 2 public repos move the needle. The private one must be invisible.
    assert body["totalRepos"] == baseline_total + 2, (
        f"totalRepos should only count public repos, "
        f"got {baseline_total} -> {body['totalRepos']} (expected +2)"
    )
    assert body["stats"]["total"] == baseline_stats_total + 2, (
        f"stats.total should only count public repos, "
        f"got {baseline_stats_total} -> {body['stats']['total']} (expected +2)"
    )
