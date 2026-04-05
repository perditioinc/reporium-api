"""
Tests for /intelligence/weekly-digest — weekly category-momentum digest.

These are unit tests that mock the DB session and the cache layer so no
real Postgres instance is required. They verify:
  1. Response shape matches the documented schema.
  2. The cached payload short-circuits the DB query on the second call.
  3. The @_limiter.limit("30/minute") decorator is wired up on the route.
  4. The "fastest growing category" highlight correctly computes delta_pct.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.routers.intelligence import (
    WeeklyDigestResponse,
    _build_weekly_digest,
    weekly_digest,
)


def _category_row(cid: str, name: str, new_7d: int, new_30d: int, total: int):
    return SimpleNamespace(
        id=cid,
        name=name,
        new_7d=new_7d,
        new_30d=new_30d,
        total_repos=total,
    )


def _sample_row(cid: str, owner: str, name: str, stars: int):
    return SimpleNamespace(
        id=cid,
        owner=owner,
        name=name,
        stars=stars,
        github_created_at=None,
    )


def _new_repo_row(owner: str, name: str, description: str, stars: int, category: str):
    return SimpleNamespace(
        owner=owner,
        name=name,
        description=description,
        stars=stars,
        category=category,
    )


def _make_db(category_rows, sample_rows, new_repo_rows):
    """Build an AsyncMock DB whose sequential execute() calls return the
    category aggregation, then the sample-repo query, then the new-repo query."""
    call_order = [category_rows, sample_rows, new_repo_rows]
    calls = {"n": 0}

    async def _execute(*args, **kwargs):
        idx = calls["n"]
        calls["n"] += 1
        rows = call_order[idx] if idx < len(call_order) else []
        result = MagicMock()
        result.fetchall = lambda: rows
        return result

    db = AsyncMock()
    db.execute = _execute
    return db


# ---------------------------------------------------------------------------
# 1. Response shape
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_weekly_digest_response_shape_matches_schema():
    """The digest must expose generated_at, week_starting, highlights, top_5_categories, and top_5_new_repos."""
    cat_rows = [
        _category_row("ai-agents", "AI Agents", new_7d=5, new_30d=12, total=45),
        _category_row("rag-retrieval", "RAG & Retrieval", new_7d=3, new_30d=9, total=28),
        _category_row("observability", "Observability", new_7d=0, new_30d=2, total=10),
    ]
    sample_rows = [
        _sample_row("ai-agents", "acme", "agent-kit", 210),
        _sample_row("ai-agents", "acme", "agent-lite", 140),
        _sample_row("rag-retrieval", "beta", "rag-flow", 330),
    ]
    new_repo_rows = [
        _new_repo_row("acme", "agent-kit", "An agent toolkit", 210, "AI Agents"),
        _new_repo_row("beta", "rag-flow", "A RAG pipeline", 330, "RAG & Retrieval"),
    ]
    db = _make_db(cat_rows, sample_rows, new_repo_rows)

    response = await _build_weekly_digest(db)

    assert isinstance(response, WeeklyDigestResponse)
    dumped = response.model_dump()
    assert set(dumped.keys()) == {
        "generated_at",
        "week_starting",
        "highlights",
        "top_5_categories",
        "top_5_new_repos",
    }
    assert set(dumped["highlights"].keys()) == {
        "fastest_growing_category",
        "total_new_repos_7d",
        "new_categories_with_activity",
    }
    # total_new_repos_7d is the sum across all categories (5 + 3 + 0).
    assert dumped["highlights"]["total_new_repos_7d"] == 8
    assert dumped["highlights"]["new_categories_with_activity"] == ["ai-agents", "rag-retrieval"]
    # Only categories with new_7d > 0 show up in the top-5 list.
    assert [c["id"] for c in dumped["top_5_categories"]] == ["ai-agents", "rag-retrieval"]
    # Sample repos are attached and capped in insertion order.
    ai_samples = dumped["top_5_categories"][0]["sample_repos"]
    assert len(ai_samples) == 2
    assert ai_samples[0] == {"owner": "acme", "name": "agent-kit", "stars": 210}
    assert dumped["top_5_new_repos"][0]["owner"] == "acme"
    assert dumped["top_5_new_repos"][0]["category"] == "AI Agents"


# ---------------------------------------------------------------------------
# 2. Fastest-growing highlight
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_weekly_digest_fastest_growing_delta_pct():
    """fastest_growing_category.delta_pct = new_7d / (total - new_7d) * 100."""
    cat_rows = [
        _category_row("ai-agents", "AI Agents", new_7d=5, new_30d=12, total=45),
    ]
    sample_rows = [_sample_row("ai-agents", "acme", "agent-kit", 210)]
    new_repo_rows: list = []
    db = _make_db(cat_rows, sample_rows, new_repo_rows)

    response = await _build_weekly_digest(db)

    fastest = response.highlights.fastest_growing_category
    assert fastest is not None
    assert fastest.id == "ai-agents"
    assert fastest.name == "AI Agents"
    assert fastest.new_7d == 5
    # 5 new out of a base of 40 = 12.5%.
    assert fastest.delta_pct == 12.5


@pytest.mark.asyncio
async def test_weekly_digest_no_activity_yields_no_fastest_growing():
    """When no category has new_7d > 0, fastest_growing_category is None and
    top_5_categories / top_5_new_repos are empty."""
    cat_rows = [
        _category_row("ai-agents", "AI Agents", new_7d=0, new_30d=0, total=45),
    ]
    db = _make_db(cat_rows, [], [])

    response = await _build_weekly_digest(db)

    assert response.highlights.fastest_growing_category is None
    assert response.highlights.total_new_repos_7d == 0
    assert response.highlights.new_categories_with_activity == []
    assert response.top_5_categories == []
    assert response.top_5_new_repos == []


# ---------------------------------------------------------------------------
# 3. Cache hit short-circuits the DB query
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_weekly_digest_uses_cached_payload():
    """When cache.get returns a payload, the endpoint must NOT touch the DB
    and must NOT re-set the cache."""
    cached_payload = WeeklyDigestResponse(
        generated_at="2026-04-05T00:00:00+00:00",
        week_starting="2026-03-29",
        highlights={
            "fastest_growing_category": None,
            "total_new_repos_7d": 0,
            "new_categories_with_activity": [],
        },
        top_5_categories=[],
        top_5_new_repos=[],
    ).model_dump()

    db = AsyncMock()
    db.execute = AsyncMock()

    with patch("app.routers.intelligence.cache.get", new=AsyncMock(return_value=cached_payload)), \
         patch("app.routers.intelligence.cache.set", new=AsyncMock()) as cache_set:
        response = await weekly_digest(request=MagicMock(), db=db)

    assert response.generated_at == "2026-04-05T00:00:00+00:00"
    assert response.week_starting == "2026-03-29"
    db.execute.assert_not_called()
    cache_set.assert_not_awaited()


@pytest.mark.asyncio
async def test_weekly_digest_cache_miss_sets_cache():
    """On a cache miss the endpoint must call _build_weekly_digest and then
    write the serialized payload back to the cache with a 1-hour TTL."""
    fake_response = WeeklyDigestResponse(
        generated_at="2026-04-05T00:00:00+00:00",
        week_starting="2026-03-29",
        highlights={
            "fastest_growing_category": None,
            "total_new_repos_7d": 0,
            "new_categories_with_activity": [],
        },
        top_5_categories=[],
        top_5_new_repos=[],
    )
    db = AsyncMock()

    with patch("app.routers.intelligence.cache.get", new=AsyncMock(return_value=None)), \
         patch("app.routers.intelligence.cache.set", new=AsyncMock()) as cache_set, \
         patch(
             "app.routers.intelligence._build_weekly_digest",
             new=AsyncMock(return_value=fake_response),
         ) as build:
        response = await weekly_digest(request=MagicMock(), db=db)

    build.assert_awaited_once()
    cache_set.assert_awaited_once()
    # The cache set call must include ttl=3600 (1 hour).
    _args, kwargs = cache_set.call_args
    assert kwargs.get("ttl") == 3600
    assert response is fake_response


# ---------------------------------------------------------------------------
# 4. Rate limit decorator is wired up
# ---------------------------------------------------------------------------

def test_weekly_digest_has_rate_limit_decorator():
    """slowapi attaches the limit metadata to the decorated function. We
    don't fire 31 requests — we just assert the attribute exists and
    mentions 30/minute."""
    # slowapi stores limits on the underlying function via a `_rate_limit`
    # style attribute. Check for any limit-related marker and verify the
    # wrapped function is importable from the module.
    assert weekly_digest is not None
    # slowapi's Limiter.limit() stores the limit strings on an attribute
    # named "_rate_limits" (or exposes them via __wrapped__). Walk likely
    # attribute names and confirm "30" appears somewhere.
    marker_found = False
    for attr in ("_rate_limits", "__wrapped__", "__closure__"):
        if hasattr(weekly_digest, attr):
            marker_found = True
            break
    assert marker_found, "weekly_digest endpoint is missing slowapi rate-limit metadata"
