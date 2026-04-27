"""Tests for POST /admin/backfill/primary_category_column endpoint."""

import uuid

import pytest
from httpx import AsyncClient
from sqlalchemy import text

import app.database as db_module
from tests.conftest import AUTH_HEADERS, TEST_API_KEY


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
