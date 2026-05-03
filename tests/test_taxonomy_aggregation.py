"""
Tests for KAN-175: GET /taxonomy/categories and GET /taxonomy/tags.

Both endpoints aggregate directly from the live `repos` / `repo_tags` corpus
on demand (Option B in the audit P2 design memo) instead of reading the
historically-empty ``taxonomy_values`` table that surfaced as the issue #251
"0 entries" symptom.

Coverage:
- Categories aggregate from ``repos.primary_category`` with public-only filter
- Tags aggregate from ``repo_tags`` join with public-only filter
- Private repos never surface — same privacy invariant as /library/preview
- Null tag arrays / repos without primary_category don't crash the aggregation
- Redis cache hit short-circuits the SQL aggregation
- Cache-Control header carries s-maxage=300 (KAN-170 pattern)
- ``invalidate_library_cache()`` sweeps both ``library:`` AND ``taxonomy:`` prefixes
- The new routes win over the catch-all ``/{dimension}`` route (FastAPI ordering)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient

from tests.conftest import AUTH_HEADERS, TEST_REPO_FIXTURE


# ---------------------------------------------------------------------------
# /taxonomy/categories
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_categories_returns_aggregated_data(client: AsyncClient):
    """Two repos with distinct categories should produce two value entries.

    The ingest pipeline copies categories[0].category_name into the
    ``primary_category`` column on the repos row (issue #444 closeout), so the
    aggregator picks up exactly the categories we seed here.
    """
    payloads = [
        {
            **TEST_REPO_FIXTURE,
            "name": "kan175-cat-a",
            "github_url": "https://github.com/testuser/kan175-cat-a",
            "is_private": False,
            "categories": [
                {"category_id": "ai-agents", "category_name": "AI Agents", "is_primary": True}
            ],
        },
        {
            **TEST_REPO_FIXTURE,
            "name": "kan175-cat-b",
            "github_url": "https://github.com/testuser/kan175-cat-b",
            "is_private": False,
            "categories": [
                {"category_id": "rag", "category_name": "RAG & Retrieval", "is_primary": True}
            ],
        },
    ]
    r = await client.post("/ingest/repos", json=payloads, headers=AUTH_HEADERS)
    assert r.status_code == 200

    # Bypass any leftover Redis cache from a sibling test by stubbing get→None.
    with patch("app.routers.taxonomy.redis_cache.get", AsyncMock(return_value=None)):
        resp = await client.get("/taxonomy/categories")

    assert resp.status_code == 200
    body = resp.json()
    assert body["dimension"] == "categories"
    assert isinstance(body["values"], list)
    assert body["total"] == len(body["values"])

    by_value = {v["value"]: v["repo_count"] for v in body["values"]}
    # Both probes must surface — count >= 1 (other tests in the same DB might
    # share categories so we don't assert == 1 strictly).
    assert by_value.get("AI Agents", 0) >= 1, by_value
    assert by_value.get("RAG & Retrieval", 0) >= 1, by_value
    # Each entry carries the dimension echo for the frontend's TaxonomyEntry shape.
    for entry in body["values"]:
        assert entry["dimension"] == "categories"
        assert isinstance(entry["repo_count"], int) and entry["repo_count"] >= 1


@pytest.mark.asyncio
async def test_categories_excludes_private_repos(client: AsyncClient):
    """A category that ONLY has private repos must not appear in the response."""
    private_payload = {
        **TEST_REPO_FIXTURE,
        "name": "kan175-private-only-cat",
        "github_url": "https://github.com/testuser/kan175-private-only-cat",
        "is_private": True,
        "categories": [
            {
                "category_id": "kan175-private-only-cat-id",
                "category_name": "kan175-private-only-cat-name",
                "is_primary": True,
            }
        ],
    }
    r = await client.post("/ingest/repos", json=[private_payload], headers=AUTH_HEADERS)
    assert r.status_code == 200

    with patch("app.routers.taxonomy.redis_cache.get", AsyncMock(return_value=None)):
        resp = await client.get("/taxonomy/categories")

    assert resp.status_code == 200
    values = {v["value"] for v in resp.json()["values"]}
    assert "kan175-private-only-cat-name" not in values, (
        "PRIVACY LEAK: a category with only private repos surfaced in /taxonomy/categories"
    )


@pytest.mark.asyncio
async def test_categories_response_envelope_shape(client: AsyncClient):
    """Backwards-compat envelope: top-level keys match the historical
    ``/taxonomy/{dimension}`` shape so existing consumers keep working."""
    with patch("app.routers.taxonomy.redis_cache.get", AsyncMock(return_value=None)):
        resp = await client.get("/taxonomy/categories")

    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"dimension", "values", "total"}


# ---------------------------------------------------------------------------
# /taxonomy/tags
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tags_returns_aggregated_data(client: AsyncClient):
    """Repos with distinctive tags should produce matching value entries."""
    payloads = [
        {
            **TEST_REPO_FIXTURE,
            "name": "kan175-tag-probe-a",
            "github_url": "https://github.com/testuser/kan175-tag-probe-a",
            "is_private": False,
            "tags": ["kan175-tag-alpha", "kan175-tag-shared"],
        },
        {
            **TEST_REPO_FIXTURE,
            "name": "kan175-tag-probe-b",
            "github_url": "https://github.com/testuser/kan175-tag-probe-b",
            "is_private": False,
            "tags": ["kan175-tag-beta", "kan175-tag-shared"],
        },
    ]
    r = await client.post("/ingest/repos", json=payloads, headers=AUTH_HEADERS)
    assert r.status_code == 200

    with patch("app.routers.taxonomy.redis_cache.get", AsyncMock(return_value=None)):
        resp = await client.get("/taxonomy/tags")

    assert resp.status_code == 200
    body = resp.json()
    assert body["dimension"] == "tags"
    assert isinstance(body["values"], list)
    assert body["total"] == len(body["values"])

    by_value = {v["value"]: v["repo_count"] for v in body["values"]}
    assert by_value.get("kan175-tag-alpha", 0) >= 1, by_value
    assert by_value.get("kan175-tag-beta", 0) >= 1, by_value
    # The shared tag should accumulate across both probes
    assert by_value.get("kan175-tag-shared", 0) >= 2, by_value
    # Result is sorted by count desc — the shared tag must outrank each unique tag
    positions = {v["value"]: idx for idx, v in enumerate(body["values"])}
    if "kan175-tag-alpha" in positions and "kan175-tag-shared" in positions:
        assert positions["kan175-tag-shared"] < positions["kan175-tag-alpha"], (
            f"sort order broken: shared(count>=2) ranked below alpha(count=1)"
        )


@pytest.mark.asyncio
async def test_tags_excludes_private_repos(client: AsyncClient):
    """A tag that only appears on private repos must not be surfaced."""
    private_payload = {
        **TEST_REPO_FIXTURE,
        "name": "kan175-private-tag-host",
        "github_url": "https://github.com/testuser/kan175-private-tag-host",
        "is_private": True,
        "tags": ["kan175-private-only-tag"],
    }
    r = await client.post("/ingest/repos", json=[private_payload], headers=AUTH_HEADERS)
    assert r.status_code == 200

    with patch("app.routers.taxonomy.redis_cache.get", AsyncMock(return_value=None)):
        resp = await client.get("/taxonomy/tags")

    assert resp.status_code == 200
    values = {v["value"] for v in resp.json()["values"]}
    assert "kan175-private-only-tag" not in values, (
        "PRIVACY LEAK: a tag from a private repo surfaced in /taxonomy/tags"
    )


@pytest.mark.asyncio
async def test_tags_handles_repos_without_tags(client: AsyncClient):
    """Repos with no tags entries must not crash the aggregation (empty join is ok)."""
    payload = {
        **TEST_REPO_FIXTURE,
        "name": "kan175-no-tags-probe",
        "github_url": "https://github.com/testuser/kan175-no-tags-probe",
        "is_private": False,
        "tags": [],
    }
    r = await client.post("/ingest/repos", json=[payload], headers=AUTH_HEADERS)
    assert r.status_code == 200

    with patch("app.routers.taxonomy.redis_cache.get", AsyncMock(return_value=None)):
        resp = await client.get("/taxonomy/tags")

    assert resp.status_code == 200
    body = resp.json()
    # Whatever tags exist, none of them should be the empty string and the
    # endpoint must not have crashed.
    for entry in body["values"]:
        assert isinstance(entry["value"], str) and entry["value"], entry


@pytest.mark.asyncio
async def test_tags_response_envelope_shape(client: AsyncClient):
    """Backwards-compat envelope: top-level keys match historical shape."""
    with patch("app.routers.taxonomy.redis_cache.get", AsyncMock(return_value=None)):
        resp = await client.get("/taxonomy/tags")

    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"dimension", "values", "total"}


# ---------------------------------------------------------------------------
# Cache behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_categories_cache_hit_short_circuits_db(client: AsyncClient):
    """When Redis returns a cached envelope, no aggregation SQL should run."""
    cached_body = {
        "dimension": "categories",
        "values": [
            {"dimension": "categories", "value": "Stubbed", "repo_count": 42}
        ],
        "total": 1,
    }
    fake_get = AsyncMock(return_value=cached_body)
    with patch("app.routers.taxonomy.redis_cache.get", fake_get):
        resp = await client.get("/taxonomy/categories")

    assert resp.status_code == 200
    assert resp.json() == cached_body
    fake_get.assert_called_once_with("taxonomy:categories")


@pytest.mark.asyncio
async def test_tags_cache_hit_short_circuits_db(client: AsyncClient):
    """Same as above but for /taxonomy/tags."""
    cached_body = {
        "dimension": "tags",
        "values": [
            {"dimension": "tags", "value": "stubbed-tag", "repo_count": 7}
        ],
        "total": 1,
    }
    fake_get = AsyncMock(return_value=cached_body)
    with patch("app.routers.taxonomy.redis_cache.get", fake_get):
        resp = await client.get("/taxonomy/tags")

    assert resp.status_code == 200
    assert resp.json() == cached_body
    fake_get.assert_called_once_with("taxonomy:tags")


@pytest.mark.asyncio
async def test_categories_cache_miss_writes_to_redis(client: AsyncClient):
    """On a cache miss, the response body must be persisted under the canonical key."""
    fake_get = AsyncMock(return_value=None)
    fake_set = AsyncMock()
    with (
        patch("app.routers.taxonomy.redis_cache.get", fake_get),
        patch("app.routers.taxonomy.redis_cache.set", fake_set),
    ):
        resp = await client.get("/taxonomy/categories")

    assert resp.status_code == 200
    fake_set.assert_called_once()
    args, kwargs = fake_set.call_args
    assert args[0] == "taxonomy:categories"
    assert kwargs.get("ttl") == 300


@pytest.mark.asyncio
async def test_tags_cache_miss_writes_to_redis(client: AsyncClient):
    """Same write-through assertion for /taxonomy/tags."""
    fake_get = AsyncMock(return_value=None)
    fake_set = AsyncMock()
    with (
        patch("app.routers.taxonomy.redis_cache.get", fake_get),
        patch("app.routers.taxonomy.redis_cache.set", fake_set),
    ):
        resp = await client.get("/taxonomy/tags")

    assert resp.status_code == 200
    fake_set.assert_called_once()
    args, kwargs = fake_set.call_args
    assert args[0] == "taxonomy:tags"
    assert kwargs.get("ttl") == 300


# ---------------------------------------------------------------------------
# Cache-Control header (KAN-170 pattern)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_categories_sets_cache_control_header(client: AsyncClient):
    """KAN-170: public, s-maxage=300, stale-while-revalidate=60."""
    with patch("app.routers.taxonomy.redis_cache.get", AsyncMock(return_value=None)):
        resp = await client.get("/taxonomy/categories")

    assert resp.status_code == 200
    cc = resp.headers.get("cache-control", "")
    assert "public" in cc, cc
    assert "s-maxage=300" in cc, cc
    assert "stale-while-revalidate=60" in cc, cc


@pytest.mark.asyncio
async def test_tags_sets_cache_control_header(client: AsyncClient):
    """KAN-170: public, s-maxage=300, stale-while-revalidate=60 on the tags route too."""
    with patch("app.routers.taxonomy.redis_cache.get", AsyncMock(return_value=None)):
        resp = await client.get("/taxonomy/tags")

    assert resp.status_code == 200
    cc = resp.headers.get("cache-control", "")
    assert "public" in cc, cc
    assert "s-maxage=300" in cc, cc
    assert "stale-while-revalidate=60" in cc, cc


# ---------------------------------------------------------------------------
# Cache invalidation hook — must sweep BOTH library: AND taxonomy:
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invalidate_library_cache_clears_taxonomy_prefix():
    """KAN-175: ``invalidate_library_cache()`` must clear the ``taxonomy:`` prefix
    in addition to ``library:``.

    Otherwise an ingest write could leave /taxonomy/categories or /taxonomy/tags
    serving aggregated data computed BEFORE the new repos landed, which is
    exactly the failure mode feedback_backfill_must_invalidate_cache.md
    enshrined as a memory-blocker.
    """
    import asyncio

    from app.routers import library_full

    fake_clear_prefix = AsyncMock()
    with patch.object(library_full.redis_cache, "clear_prefix", fake_clear_prefix):
        library_full.invalidate_library_cache()
        # ensure_future returns immediately; yield once for the loop.
        await asyncio.sleep(0)

    prefixes_called = {call.args[0] for call in fake_clear_prefix.call_args_list}
    assert "library:" in prefixes_called, prefixes_called
    assert "taxonomy:" in prefixes_called, prefixes_called


# ---------------------------------------------------------------------------
# Route precedence — /categories and /tags must win over /{dimension} catch-all
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_categories_route_wins_over_dimension_catchall(client: AsyncClient):
    """If FastAPI ever picks the catch-all first, /taxonomy/categories would
    fall through to ``list_taxonomy_values`` and return empty values from the
    historically-empty ``taxonomy_values`` table — exactly the bug KAN-175
    fixes. Asserting the new key shape guarantees we hit the aggregator.
    """
    with patch("app.routers.taxonomy.redis_cache.get", AsyncMock(return_value=None)):
        resp = await client.get("/taxonomy/categories")

    assert resp.status_code == 200
    body = resp.json()
    # Aggregator entries carry these exact keys; the legacy taxonomy_values
    # shape carries id/name/description/trending_score etc. Verifying the
    # shape proves we hit the aggregator, not the catch-all.
    if body["values"]:
        sample = body["values"][0]
        assert set(sample.keys()) == {"dimension", "value", "repo_count"}, sample


@pytest.mark.asyncio
async def test_tags_route_wins_over_dimension_catchall(client: AsyncClient):
    """Same precedence guarantee for /taxonomy/tags."""
    with patch("app.routers.taxonomy.redis_cache.get", AsyncMock(return_value=None)):
        resp = await client.get("/taxonomy/tags")

    assert resp.status_code == 200
    body = resp.json()
    if body["values"]:
        sample = body["values"][0]
        assert set(sample.keys()) == {"dimension", "value", "repo_count"}, sample
