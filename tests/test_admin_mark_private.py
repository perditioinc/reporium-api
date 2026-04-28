"""POST /admin/repos/mark-private — bleed-stop endpoint for marking a single
repo as ``is_private = true`` and invalidating affected caches.

Designed for incident response (2026-04-27 hippo-harvest-assignment leak):
operator must be able to flip a wrongly-public row to private through a
deployed admin path, with a dry-run preview AND post-mutation cache
invalidation, since direct SQL access is gated by Cloud SQL private-IP
networking.

The endpoint must:
  1. Require X-Admin-Key (re-uses ``require_admin_key``).
  2. 404 when no row matches owner+name.
  3. ``dry_run=true`` returns match info without mutating.
  4. ``dry_run=false`` sets ``is_private = true`` and invalidates caches.
  5. Idempotent — applying to an already-private row is a no-op success.
  6. Write an AuditLog entry for the mutation.
"""
from __future__ import annotations

import uuid

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy import text

import app.database as db_module


_OWNER = "perditioinc"
_NAME = "mark-private-fixture"


async def _seed_repo(*, is_private: bool = False) -> str:
    """Insert a public repo for the test. Returns the new id (str)."""
    repo_id = str(uuid.uuid4())
    async with db_module.async_session_factory() as session:
        await session.execute(
            text(
                """
                INSERT INTO repos
                    (id, name, owner, github_url, description,
                     is_fork, is_private, primary_language)
                VALUES
                    (:id, :name, :owner, :github_url, :description,
                     false, :is_private, 'Python')
                ON CONFLICT (name) DO UPDATE
                    SET is_private = EXCLUDED.is_private,
                        owner = EXCLUDED.owner
                RETURNING id::text
                """
            ),
            {
                "id": repo_id,
                "name": _NAME,
                "owner": _OWNER,
                "github_url": f"https://github.com/{_OWNER}/{_NAME}",
                "description": "fixture for mark-private",
                "is_private": is_private,
            },
        )
        row = (
            await session.execute(
                text("SELECT id::text FROM repos WHERE name = :name"),
                {"name": _NAME},
            )
        ).first()
        await session.commit()
    return row[0]


@pytest_asyncio.fixture
async def public_repo(_setup_db) -> str:
    """Seed a public repo, yield its id, clean up after."""
    rid = await _seed_repo(is_private=False)
    yield rid
    async with db_module.async_session_factory() as session:
        await session.execute(
            text("DELETE FROM repos WHERE name = :name"),
            {"name": _NAME},
        )
        await session.commit()


@pytest_asyncio.fixture
async def private_repo(_setup_db) -> str:
    """Seed an already-private repo (idempotency test)."""
    rid = await _seed_repo(is_private=True)
    yield rid
    async with db_module.async_session_factory() as session:
        await session.execute(
            text("DELETE FROM repos WHERE name = :name"),
            {"name": _NAME},
        )
        await session.commit()


_ADMIN_HEADER = {"X-Admin-Key": "test-admin-key"}


# ---------------------------------------------------------------------------
# Route registration (runs without a DB — first canary)
# ---------------------------------------------------------------------------


def test_mark_private_route_is_registered():
    """The endpoint must be wired into the app router. Catches forgetting
    ``include_router`` regardless of test DB availability."""
    from app.main import app

    routes = set()
    for r in app.routes:
        methods = getattr(r, "methods", set()) or set()
        path = getattr(r, "path", "")
        for m in methods:
            routes.add(f"{m} {path}")
    assert "POST /admin/repos/mark-private" in routes, (
        "POST /admin/repos/mark-private must be registered on the app — "
        "the bleed-stop endpoint is missing"
    )


# ---------------------------------------------------------------------------
# Authn / 404 / shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mark_private_requires_admin_key(client: AsyncClient, monkeypatch):
    """Without the X-Admin-Key header (when ADMIN_API_KEY is set), 403."""
    monkeypatch.setenv("ADMIN_API_KEY", "test-admin-key")
    resp = await client.post(
        "/admin/repos/mark-private",
        json={"owner": _OWNER, "name": _NAME, "dry_run": True},
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_mark_private_404_for_unknown_repo(
    client: AsyncClient, _setup_db, monkeypatch
):
    monkeypatch.setenv("ADMIN_API_KEY", "test-admin-key")
    resp = await client.post(
        "/admin/repos/mark-private",
        json={"owner": "nobody", "name": "does-not-exist", "dry_run": True},
        headers=_ADMIN_HEADER,
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Dry-run mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mark_private_dry_run_returns_match_without_mutation(
    client: AsyncClient, public_repo: str, monkeypatch
):
    monkeypatch.setenv("ADMIN_API_KEY", "test-admin-key")
    resp = await client.post(
        "/admin/repos/mark-private",
        json={"owner": _OWNER, "name": _NAME, "dry_run": True},
        headers=_ADMIN_HEADER,
    )
    assert resp.status_code == 200
    body = resp.json()

    assert body["applied"] is False
    assert body["match_count"] == 1

    match = body["match"]
    assert match["id"] == public_repo
    assert match["owner"] == _OWNER
    assert match["name"] == _NAME
    assert match["current_is_private"] is False
    # ingested_at may be None for raw inserts that didn't set it; just check key.
    assert "ingested_at" in match

    # Cache prefixes are listed so the operator can review what would be touched.
    assert isinstance(body["would_invalidate_prefixes"], list)
    assert len(body["would_invalidate_prefixes"]) >= 5

    # Confirm DB unchanged.
    async with db_module.async_session_factory() as session:
        is_private = (
            await session.execute(
                text("SELECT is_private FROM repos WHERE name = :name"),
                {"name": _NAME},
            )
        ).scalar_one()
        assert is_private is False, "dry-run must not mutate"


# ---------------------------------------------------------------------------
# Apply mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mark_private_apply_flips_is_private(
    client: AsyncClient, public_repo: str, monkeypatch
):
    monkeypatch.setenv("ADMIN_API_KEY", "test-admin-key")
    resp = await client.post(
        "/admin/repos/mark-private",
        json={"owner": _OWNER, "name": _NAME, "dry_run": False},
        headers=_ADMIN_HEADER,
    )
    assert resp.status_code == 200
    body = resp.json()

    assert body["applied"] is True
    assert body["match"]["current_is_private"] is True
    assert "invalidated_prefixes" in body
    assert isinstance(body["invalidated_prefixes"], list)

    # Confirm DB row flipped.
    async with db_module.async_session_factory() as session:
        is_private = (
            await session.execute(
                text("SELECT is_private FROM repos WHERE name = :name"),
                {"name": _NAME},
            )
        ).scalar_one()
        assert is_private is True


@pytest.mark.asyncio
async def test_mark_private_apply_invalidates_caches(
    client: AsyncClient, public_repo: str, monkeypatch
):
    """The endpoint must call redis_cache.clear_prefix for every documented
    prefix. We patch the clear_prefix method to record invocations rather
    than depending on a real Redis instance — both production and the test
    DB use redis_cache, so capturing the calls proves the invalidation
    logic ran."""
    monkeypatch.setenv("ADMIN_API_KEY", "test-admin-key")

    captured_prefixes: list[str] = []

    from app import cache_redis

    async def _capture(prefix: str) -> None:
        captured_prefixes.append(prefix)

    monkeypatch.setattr(cache_redis.redis_cache, "clear_prefix", _capture)

    resp = await client.post(
        "/admin/repos/mark-private",
        json={"owner": _OWNER, "name": _NAME, "dry_run": False},
        headers=_ADMIN_HEADER,
    )
    assert resp.status_code == 200

    # Sanity: every documented prefix was actually invalidated.
    assert "library:" in captured_prefixes
    assert "repos:" in captured_prefixes
    assert "graph_" in captured_prefixes
    assert "intelligence:" in captured_prefixes
    assert "similar:" in captured_prefixes


@pytest.mark.asyncio
async def test_mark_private_idempotent_on_already_private(
    client: AsyncClient, private_repo: str, monkeypatch
):
    """Applying mark-private to an already-private repo is a no-op success.
    Critical for re-running the same incident-response command without
    surprises."""
    monkeypatch.setenv("ADMIN_API_KEY", "test-admin-key")
    resp = await client.post(
        "/admin/repos/mark-private",
        json={"owner": _OWNER, "name": _NAME, "dry_run": False},
        headers=_ADMIN_HEADER,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["applied"] is True
    assert body["match"]["current_is_private"] is True


@pytest.mark.asyncio
async def test_mark_private_writes_audit_log(
    client: AsyncClient, public_repo: str, monkeypatch
):
    """Every mutation writes an AuditLog row capturing endpoint+payload+200."""
    monkeypatch.setenv("ADMIN_API_KEY", "test-admin-key")

    async with db_module.async_session_factory() as session:
        before = (
            await session.execute(
                text(
                    "SELECT COUNT(*) FROM audit_logs "
                    "WHERE endpoint = 'admin.mark_private'"
                )
            )
        ).scalar_one()

    resp = await client.post(
        "/admin/repos/mark-private",
        json={"owner": _OWNER, "name": _NAME, "dry_run": False},
        headers=_ADMIN_HEADER,
    )
    assert resp.status_code == 200

    async with db_module.async_session_factory() as session:
        after = (
            await session.execute(
                text(
                    "SELECT COUNT(*) FROM audit_logs "
                    "WHERE endpoint = 'admin.mark_private'"
                )
            )
        ).scalar_one()
        latest = (
            await session.execute(
                text(
                    "SELECT response_status, request_summary FROM audit_logs "
                    "WHERE endpoint = 'admin.mark_private' "
                    "ORDER BY id DESC LIMIT 1"
                )
            )
        ).first()

    assert after == before + 1
    assert latest.response_status == 200
    # Request summary should contain owner & name so an auditor can verify the
    # exact target of the mutation without reading the full payload.
    assert _OWNER in (latest.request_summary or "")
    assert _NAME in (latest.request_summary or "")


@pytest.mark.asyncio
async def test_mark_private_dry_run_does_not_invalidate_cache(
    client: AsyncClient, public_repo: str, monkeypatch
):
    """Dry-run must not touch the cache — operators rely on dry-run being
    truly read-only."""
    monkeypatch.setenv("ADMIN_API_KEY", "test-admin-key")

    captured_prefixes: list[str] = []
    from app import cache_redis

    async def _capture(prefix: str) -> None:
        captured_prefixes.append(prefix)

    monkeypatch.setattr(cache_redis.redis_cache, "clear_prefix", _capture)

    resp = await client.post(
        "/admin/repos/mark-private",
        json={"owner": _OWNER, "name": _NAME, "dry_run": True},
        headers=_ADMIN_HEADER,
    )
    assert resp.status_code == 200
    assert captured_prefixes == [], (
        "dry-run must not call clear_prefix — invalidation is reserved for "
        "the apply path"
    )
