"""Tests for ask_sessions retention, RTBF endpoint, and PII redaction (issue #238)."""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from httpx import AsyncClient
from sqlalchemy import text

import app.database as db_module
from app.privacy import redact_pii
from app.retention import purge_expired_ask_sessions


# ---------------------------------------------------------------------------
# redact_pii unit tests
# ---------------------------------------------------------------------------


def test_redact_pii_email():
    out = redact_pii("contact me at alice@example.com please")
    assert "alice@example.com" not in out
    assert "[REDACTED_EMAIL]" in out


def test_redact_pii_phone_long_digits():
    out = redact_pii("my number is 14155551234 thanks")
    assert "14155551234" not in out
    assert "[REDACTED_NUMBER]" in out


def test_redact_pii_short_digits_preserved():
    """Short numbers (e.g. "top 50 repos") must not be redacted."""
    out = redact_pii("show me the top 50 repos with 1000 stars")
    assert "50" in out
    assert "1000" in out
    assert "[REDACTED_NUMBER]" not in out


def test_redact_pii_api_key_sk():
    out = redact_pii("use sk-abcdefghijklmnopqrstuvwxyz1234 as the key")
    assert "sk-abcdefghijklmnopqrstuvwxyz1234" not in out
    assert "[REDACTED_KEY]" in out


def test_redact_pii_github_token():
    out = redact_pii("token ghp_ABCDEFGHIJKLMNOPQRSTUVWX1234567890 here")
    assert "ghp_ABCDEFGHIJKLMNOPQRSTUVWX1234567890" not in out
    assert "[REDACTED_KEY]" in out


def test_redact_pii_pass_through():
    """Text without PII is returned unchanged."""
    text_in = "what are the best RAG frameworks?"
    assert redact_pii(text_in) == text_in


def test_redact_pii_empty():
    assert redact_pii("") == ""


def test_redact_pii_combined():
    raw = "email bob@foo.com phone 15551234567 key sk-aaaaaaaaaaaaaaaaaaaaaaa"
    out = redact_pii(raw)
    assert "[REDACTED_EMAIL]" in out
    assert "[REDACTED_NUMBER]" in out
    assert "[REDACTED_KEY]" in out
    assert "bob@foo.com" not in out


# ---------------------------------------------------------------------------
# purge_expired_ask_sessions — exercises real test DB via conftest fixtures
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_purge_expired_ask_sessions_deletes_old_preserves_recent(_setup_db):
    """Rows older than max_age_days are deleted; recent rows stay."""
    old_session = uuid.uuid4()
    new_session = uuid.uuid4()

    async with db_module.async_session_factory() as db:
        # Clean slate — this test owns the ask_sessions table.
        await db.execute(text("DELETE FROM ask_sessions"))
        await db.execute(
            text(
                "INSERT INTO ask_sessions (id, session_id, turn_number, question, answer, created_at) "
                "VALUES (gen_random_uuid(), CAST(:sid AS uuid), 0, :q, :a, :ts)"
            ),
            {
                "sid": str(old_session),
                "q": "old question",
                "a": "old answer",
                "ts": datetime.now(timezone.utc) - timedelta(days=120),
            },
        )
        await db.execute(
            text(
                "INSERT INTO ask_sessions (id, session_id, turn_number, question, answer, created_at) "
                "VALUES (gen_random_uuid(), CAST(:sid AS uuid), 0, :q, :a, :ts)"
            ),
            {
                "sid": str(new_session),
                "q": "new question",
                "a": "new answer",
                "ts": datetime.now(timezone.utc) - timedelta(days=5),
            },
        )
        await db.commit()

    count = await purge_expired_ask_sessions(max_age_days=90)
    assert count == 1

    async with db_module.async_session_factory() as db:
        remaining = (
            await db.execute(text("SELECT session_id FROM ask_sessions"))
        ).fetchall()
        remaining_ids = {str(r[0]) for r in remaining}
        assert str(new_session) in remaining_ids
        assert str(old_session) not in remaining_ids
        await db.execute(text("DELETE FROM ask_sessions"))
        await db.commit()


# ---------------------------------------------------------------------------
# Admin endpoint integration tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_admin_purge_ask_sessions_requires_admin_key(
    client: AsyncClient, monkeypatch
):
    """When ADMIN_API_KEY is set, calling without the header yields 403."""
    monkeypatch.setenv("ADMIN_API_KEY", "test-admin-key")
    resp = await client.post("/admin/purge-ask-sessions?days=90")
    assert resp.status_code == 403

    resp_ok = await client.post(
        "/admin/purge-ask-sessions?days=90",
        headers={"X-Admin-Key": "test-admin-key"},
    )
    assert resp_ok.status_code == 200
    body = resp_ok.json()
    assert "purged" in body
    assert body["max_age_days"] == 90


@pytest.mark.asyncio
async def test_admin_purge_ask_sessions_days_bounds(client: AsyncClient):
    """days parameter must be bounded to [7, 365]."""
    # Below min
    resp = await client.post("/admin/purge-ask-sessions?days=1")
    assert resp.status_code == 422
    # Above max
    resp = await client.post("/admin/purge-ask-sessions?days=9999")
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_admin_delete_ask_session_requires_admin_key(
    client: AsyncClient, monkeypatch
):
    monkeypatch.setenv("ADMIN_API_KEY", "test-admin-key")
    sid = str(uuid.uuid4())
    resp = await client.delete(f"/admin/ask-sessions/{sid}")
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_admin_delete_ask_session_deletes_rows(client: AsyncClient):
    """Inserting two rows for a session and then DELETEing should remove both."""
    sid = uuid.uuid4()
    other_sid = uuid.uuid4()

    async with db_module.async_session_factory() as db:
        for turn in (0, 1):
            await db.execute(
                text(
                    "INSERT INTO ask_sessions (id, session_id, turn_number, question, answer) "
                    "VALUES (gen_random_uuid(), CAST(:sid AS uuid), :t, :q, :a)"
                ),
                {"sid": str(sid), "t": turn, "q": f"q{turn}", "a": f"a{turn}"},
            )
        await db.execute(
            text(
                "INSERT INTO ask_sessions (id, session_id, turn_number, question, answer) "
                "VALUES (gen_random_uuid(), CAST(:sid AS uuid), 0, :q, :a)"
            ),
            {"sid": str(other_sid), "q": "other", "a": "other"},
        )
        await db.commit()

    resp = await client.delete(f"/admin/ask-sessions/{sid}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["deleted"] == 2
    assert body["session_id"] == str(sid)

    async with db_module.async_session_factory() as db:
        remaining = (
            await db.execute(
                text("SELECT session_id FROM ask_sessions WHERE session_id = CAST(:sid AS uuid)"),
                {"sid": str(sid)},
            )
        ).fetchall()
        assert remaining == []
        # Unrelated session still present.
        other = (
            await db.execute(
                text("SELECT session_id FROM ask_sessions WHERE session_id = CAST(:sid AS uuid)"),
                {"sid": str(other_sid)},
            )
        ).fetchall()
        assert len(other) == 1
        await db.execute(text("DELETE FROM ask_sessions"))
        await db.commit()


@pytest.mark.asyncio
async def test_admin_delete_ask_session_invalid_uuid(client: AsyncClient):
    resp = await client.delete("/admin/ask-sessions/not-a-uuid")
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_admin_delete_ask_session_idempotent(client: AsyncClient):
    """Deleting a non-existent session returns 200 with deleted=0."""
    sid = str(uuid.uuid4())
    resp = await client.delete(f"/admin/ask-sessions/{sid}")
    assert resp.status_code == 200
    assert resp.json() == {"deleted": 0, "session_id": sid}
