"""Tests for POST /admin/backfill/primary_category_column endpoint."""

import uuid
from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient
from sqlalchemy import text

import app.database as db_module
from app.routers.admin import backfill_primary_category_column
from tests.conftest import AUTH_HEADERS, TEST_API_KEY


class _CoverageRow:
    """Tuple-shaped row stand-in for the coverage_sql `.one()` call."""

    def __init__(self, public_total: int, public_with_col: int, drift_to_heal: int):
        self.public_total = public_total
        self.public_with_col = public_with_col
        self.drift_to_heal = drift_to_heal


class _CoverageResult:
    def __init__(self, row: _CoverageRow):
        self._row = row

    def one(self):
        return self._row


class _UpdateReturningResult:
    """Stand-in for the UPDATE ... RETURNING name result.

    The handler iterates `result.fetchall()` and reads `row[0]`, so each
    row needs to be subscriptable. A 1-tuple matches both that shape and
    the actual SQLAlchemy `Row` protocol used at runtime.
    """

    def __init__(self, names: list[str]):
        self._names = names

    def fetchall(self):
        return [(name,) for name in self._names]


async def _delete_repos(ids: list[str]) -> None:
    """Tear down inserted repos and their junction rows."""
    async with db_module.async_session_factory() as session:
        await session.execute(
            text("DELETE FROM repo_categories WHERE repo_id = ANY(:ids)"),
            {"ids": ids},
        )
        await session.execute(
            text("DELETE FROM repos WHERE id = ANY(:ids)"),
            {"ids": ids},
        )
        await session.commit()


@pytest.mark.asyncio
async def test_backfill_primary_category_column_requires_api_key(client: AsyncClient):
    resp = await client.post("/admin/backfill/primary_category_column")
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_backfill_primary_category_column_requires_admin_key(
    client: AsyncClient, monkeypatch
):
    monkeypatch.setenv("ADMIN_API_KEY", "test-admin-key")
    resp = await client.post(
        "/admin/backfill/primary_category_column",
        headers={"Authorization": f"Bearer {TEST_API_KEY}"},
    )
    assert resp.status_code in (401, 403)


@pytest.mark.asyncio
async def test_backfill_heals_drift_and_is_idempotent(client: AsyncClient):
    """Repos with is_primary=true in junction but NULL column get healed.

    Repos with no primary in the junction are left NULL.
    A second call is a no-op (idempotency).
    """
    drift_id = str(uuid.uuid4())
    empty_id = str(uuid.uuid4())
    already_set_id = str(uuid.uuid4())
    inserted_ids = [drift_id, empty_id, already_set_id]

    try:
        async with db_module.async_session_factory() as session:
            for repo_id, name, primary in [
                (drift_id, "pcc-drift-repo", None),
                (empty_id, "pcc-empty-repo", None),
                (already_set_id, "pcc-already-set-repo", "Foundation Models"),
            ]:
                await session.execute(text(
                    "INSERT INTO repos (id, name, owner, github_url, is_fork, is_private, primary_category) "
                    "VALUES (:id, :name, 'testuser', :url, false, false, :primary) "
                    "ON CONFLICT (name) DO UPDATE SET primary_category = EXCLUDED.primary_category"
                ), {"id": repo_id, "name": name,
                    "url": f"https://github.com/testuser/{name}",
                    "primary": primary})
            for repo_id, cat_name in [
                (drift_id, "AI Agents"),
                (already_set_id, "Foundation Models"),
            ]:
                await session.execute(text(
                    "INSERT INTO repo_categories (repo_id, category_id, category_name, is_primary) "
                    "VALUES (:repo_id, :cat_id, :cat_name, true) "
                    "ON CONFLICT (repo_id, category_id) DO UPDATE SET is_primary = true"
                ), {"repo_id": repo_id,
                    "cat_id": cat_name.lower().replace(" ", "-").replace("&", "and"),
                    "cat_name": cat_name})
            await session.commit()

        resp = await client.post(
            "/admin/backfill/primary_category_column",
            headers=AUTH_HEADERS,
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["dry_run"] is False
        assert data["updated"] >= 1
        assert data["before"]["drift_rows"] >= 1
        assert data["after"]["drift_rows"] == 0
        assert data["after"]["public_with_primary_category"] >= data["before"]["public_with_primary_category"]

        async with db_module.async_session_factory() as session:
            drift_col = (await session.execute(
                text("SELECT primary_category FROM repos WHERE id = :id"),
                {"id": drift_id},
            )).scalar_one()
            empty_col = (await session.execute(
                text("SELECT primary_category FROM repos WHERE id = :id"),
                {"id": empty_id},
            )).scalar_one()
            already_col = (await session.execute(
                text("SELECT primary_category FROM repos WHERE id = :id"),
                {"id": already_set_id},
            )).scalar_one()

        assert drift_col == "AI Agents"
        assert empty_col is None
        assert already_col == "Foundation Models"

        resp2 = await client.post(
            "/admin/backfill/primary_category_column",
            headers=AUTH_HEADERS,
        )
        assert resp2.status_code == 200
        assert resp2.json()["updated"] == 0
        assert resp2.json()["after"]["drift_rows"] == 0
    finally:
        await _delete_repos(inserted_ids)


@pytest.mark.asyncio
async def test_backfill_dry_run_does_not_write(client: AsyncClient):
    drift_id = str(uuid.uuid4())
    inserted_ids = [drift_id]

    try:
        async with db_module.async_session_factory() as session:
            await session.execute(text(
                "INSERT INTO repos (id, name, owner, github_url, is_fork, is_private, primary_category) "
                "VALUES (:id, 'pcc-dry-run-drift-repo', 'testuser', :url, false, false, NULL) "
                "ON CONFLICT (name) DO UPDATE SET primary_category = NULL"
            ), {"id": drift_id, "url": "https://github.com/testuser/pcc-dry-run-drift-repo"})
            await session.execute(text(
                "INSERT INTO repo_categories (repo_id, category_id, category_name, is_primary) "
                "VALUES (:id, 'rag-retrieval', 'RAG & Retrieval', true) "
                "ON CONFLICT (repo_id, category_id) DO UPDATE SET is_primary = true"
            ), {"id": drift_id})
            await session.commit()

        resp = await client.post(
            "/admin/backfill/primary_category_column?dry_run=true",
            headers=AUTH_HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["dry_run"] is True
        assert data["updated"] == 0
        assert data["before"]["drift_rows"] >= 1
        assert data["after"]["drift_rows"] == data["before"]["drift_rows"]

        async with db_module.async_session_factory() as session:
            col = (await session.execute(
                text("SELECT primary_category FROM repos WHERE id = :id"),
                {"id": drift_id},
            )).scalar_one()
        assert col is None
    finally:
        await _delete_repos(inserted_ids)


# ---------------------------------------------------------------------------
# Unit test (no DB) — verifies cache invalidation contract per KAN-API-CACHE-INVALIDATE
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_backfill_invalidates_repos_detail_cache_for_each_healed_row():
    """Every row whose primary_category was healed should have its
    `repos:detail:<name>` cache key invalidated.

    Memory entry 4931 caught that the post-#445 backfill committed UPDATEs but
    did not bust the per-repo detail cache, so /repos/{name} could keep
    returning the stale (NULL or pre-rewrite) primary_category for up to
    CACHE_TTL_REPO_DETAIL (1h) after a backfill. This unit test pins the
    invalidation contract so a future refactor can't silently regress it.
    """
    healed_names = ["repo-alpha", "repo-beta", "repo-gamma"]

    db = AsyncMock()
    # The handler issues exactly 3 db.execute calls in the non-dry-run path:
    #   1) coverage_sql BEFORE
    #   2) UPDATE ... RETURNING name
    #   3) coverage_sql AFTER
    db.execute = AsyncMock(side_effect=[
        _CoverageResult(_CoverageRow(public_total=100, public_with_col=80, drift_to_heal=3)),
        _UpdateReturningResult(healed_names),
        _CoverageResult(_CoverageRow(public_total=100, public_with_col=83, drift_to_heal=0)),
    ])

    with patch("app.routers.admin.cache.invalidate", new=AsyncMock()) as invalidate, \
         patch("app.routers.admin.invalidate_library_cache") as invalidate_memory:
        result = await backfill_primary_category_column(
            request=None,  # @_limiter.limit decorator wraps but isn't exercised in unit call
            dry_run=False,
            db=db,
            _api_key="test",
            _admin_key=None,
        )

    # Sanity: handler reported the rows we said we healed.
    assert result["dry_run"] is False
    assert result["updated"] == 3
    assert result["after"]["drift_rows"] == 0

    # Commit happened.
    db.commit.assert_awaited_once()

    # The library/list bulk invalidations are still expected (existing contract).
    bulk_calls = {call.args[0] for call in invalidate.await_args_list}
    assert "library:full*" in bulk_calls
    assert "repos:list:*" in bulk_calls

    # Per-row invalidation: each healed repo's detail cache must be busted.
    for name in healed_names:
        assert f"repos:detail:{name}" in bulk_calls, (
            f"expected `repos:detail:{name}` to be invalidated after backfill "
            f"(KAN-API-CACHE-INVALIDATE / mem 4931). saw: {sorted(bulk_calls)}"
        )

    # In-memory library cache invalidation also still fires.
    invalidate_memory.assert_called_once()


@pytest.mark.asyncio
async def test_backfill_dry_run_does_not_invalidate_cache():
    """Dry-run path must not touch the cache (no rows changed)."""
    db = AsyncMock()
    # In dry-run the handler issues 2 execute calls: BEFORE coverage + AFTER coverage.
    db.execute = AsyncMock(side_effect=[
        _CoverageResult(_CoverageRow(public_total=100, public_with_col=80, drift_to_heal=3)),
        _CoverageResult(_CoverageRow(public_total=100, public_with_col=80, drift_to_heal=3)),
    ])

    with patch("app.routers.admin.cache.invalidate", new=AsyncMock()) as invalidate, \
         patch("app.routers.admin.invalidate_library_cache") as invalidate_memory:
        result = await backfill_primary_category_column(
            request=None,
            dry_run=True,
            db=db,
            _api_key="test",
            _admin_key=None,
        )

    assert result["dry_run"] is True
    assert result["updated"] == 0
    db.commit.assert_not_awaited()
    invalidate.assert_not_awaited()
    invalidate_memory.assert_not_called()
