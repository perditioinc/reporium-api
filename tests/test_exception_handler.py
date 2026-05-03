"""KAN-190: structured exception handler tests.

Verifies the global Exception handler emits a JSON log line with the agreed
schema (route, method, error_class, error_message, stack_hash, client_host)
AND returns a 500 response with a generic detail. Also verifies that:

  - HTTPException 4xx are NOT routed through the unhandled-exception handler
    (otherwise every 404 would page on-call).
  - `error_message` is truncated at 500 chars so noisy exceptions can't
    blow up Cloud Logging payload limits or leak large user input.
  - `stack_hash` is stable across two raises of the same exception (so
    Sentry/log alerts can group recurrences).

These tests do NOT touch the database — they exercise the handler in
isolation via direct invocation, then a small ASGI integration test using
a fresh FastAPI app to avoid coupling to the production app's DB-bound
lifespan.
"""
from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from httpx import ASGITransport, AsyncClient
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.main import (
    _EXTRA_FIELDS,
    _JsonFormatter,
    structured_exception_handler,
    _passthrough_http_exception_handler,
)


# ---------------------------------------------------------------------------
# Direct handler invocation tests (no ASGI / no DB)
# ---------------------------------------------------------------------------


def _fake_request(path: str = "/intelligence/ask", method: str = "POST", host: str = "10.0.0.5") -> MagicMock:
    """Build a minimal request stub with .url.path, .method, .client.host."""
    req = MagicMock()
    req.url.path = path
    req.method = method
    req.client = MagicMock()
    req.client.host = host
    return req


@pytest.mark.asyncio
async def test_handler_emits_all_required_fields(caplog):
    """The handler must emit every field listed in the KAN-190 schema."""
    caplog.set_level(logging.ERROR, logger="app.main")
    req = _fake_request(path="/intelligence/ask", method="POST", host="1.2.3.4")
    exc = ValueError("boom")
    response = await structured_exception_handler(req, exc)

    # Response shape: 500 with generic detail (no info leak).
    assert response.status_code == 500
    body = json.loads(response.body)
    assert body == {"detail": "Internal Server Error"}

    # Log emission: exactly one ERROR record from app.main, with all six
    # structured fields populated.
    records = [r for r in caplog.records if r.name == "app.main" and r.levelno == logging.ERROR]
    assert len(records) == 1, f"expected exactly one ERROR log, got {len(records)}"
    rec = records[0]
    assert rec.message == "api.unhandled_exception"
    assert rec.route == "/intelligence/ask"
    assert rec.method == "POST"
    assert rec.error_class == "ValueError"
    assert rec.error_message == "boom"
    assert isinstance(rec.stack_hash, str) and len(rec.stack_hash) == 16
    assert rec.client_host == "1.2.3.4"


@pytest.mark.asyncio
async def test_handler_truncates_long_error_messages(caplog):
    """error_message MUST be capped at 500 chars to bound payload + leak surface."""
    caplog.set_level(logging.ERROR, logger="app.main")
    huge = "x" * 5000
    response = await structured_exception_handler(_fake_request(), RuntimeError(huge))
    assert response.status_code == 500

    rec = next(r for r in caplog.records if r.name == "app.main" and r.levelno == logging.ERROR)
    assert len(rec.error_message) == 500
    assert rec.error_message == "x" * 500


@pytest.mark.asyncio
async def test_handler_handles_missing_client(caplog):
    """If request.client is None (e.g. internal call), client_host must be None, not crash."""
    caplog.set_level(logging.ERROR, logger="app.main")
    req = _fake_request()
    req.client = None
    response = await structured_exception_handler(req, KeyError("missing"))
    assert response.status_code == 500
    rec = next(r for r in caplog.records if r.name == "app.main" and r.levelno == logging.ERROR)
    assert rec.client_host is None


def test_extra_fields_includes_structured_exception_keys():
    """The JSON formatter's extra-field allow-list must include the new keys —
    otherwise they'd be dropped from the Cloud Logging payload.
    """
    for key in ("error_class", "error_message", "stack_hash", "client_host"):
        assert key in _EXTRA_FIELDS, f"_EXTRA_FIELDS missing {key!r} — JSON formatter will drop it"


def test_json_formatter_serializes_extra_fields():
    """End-to-end: a LogRecord with the structured-exception fields must
    render as a JSON object whose top-level keys include all of them.
    """
    formatter = _JsonFormatter()
    record = logging.LogRecord(
        name="app.main", level=logging.ERROR, pathname="x", lineno=1,
        msg="api.unhandled_exception", args=(), exc_info=None,
    )
    record.route = "/library/preview"
    record.method = "GET"
    record.error_class = "RuntimeError"
    record.error_message = "db down"
    record.stack_hash = "abcdef0123456789"
    record.client_host = "192.168.1.1"

    payload = json.loads(formatter.format(record))
    assert payload["route"] == "/library/preview"
    assert payload["method"] == "GET"
    assert payload["error_class"] == "RuntimeError"
    assert payload["error_message"] == "db down"
    assert payload["stack_hash"] == "abcdef0123456789"
    assert payload["client_host"] == "192.168.1.1"


# ---------------------------------------------------------------------------
# ASGI integration test — uses a tiny FastAPI app that wires only the
# KAN-190 handlers. Avoids the production app's DB-bound lifespan so the
# test runs without Postgres.
# ---------------------------------------------------------------------------


def _isolated_app() -> FastAPI:
    """Build a FastAPI app with only the KAN-190 handlers registered + a
    couple of routes that intentionally raise.
    """
    from fastapi.exceptions import RequestValidationError
    from app.main import _passthrough_validation_handler

    iso = FastAPI()
    iso.add_exception_handler(StarletteHTTPException, _passthrough_http_exception_handler)
    iso.add_exception_handler(RequestValidationError, _passthrough_validation_handler)
    iso.add_exception_handler(Exception, structured_exception_handler)

    @iso.get("/boom")
    async def boom():
        raise RuntimeError("kaboom")

    @iso.get("/notfound")
    async def notfound():
        raise HTTPException(status_code=404, detail="nope")

    return iso


@pytest.mark.asyncio
async def test_unhandled_exception_returns_500_with_generic_body(caplog):
    caplog.set_level(logging.ERROR, logger="app.main")
    iso = _isolated_app()
    async with AsyncClient(transport=ASGITransport(app=iso, raise_app_exceptions=False), base_url="http://test") as ac:
        resp = await ac.get("/boom")
    assert resp.status_code == 500
    assert resp.json() == {"detail": "Internal Server Error"}

    # Structured log should have been emitted with the route + error class.
    records = [r for r in caplog.records if r.name == "app.main" and r.levelno == logging.ERROR]
    assert any(
        r.message == "api.unhandled_exception"
        and getattr(r, "route", None) == "/boom"
        and getattr(r, "error_class", None) == "RuntimeError"
        for r in records
    ), "structured exception log was not emitted for /boom"


@pytest.mark.asyncio
async def test_http_exception_4xx_does_not_hit_unhandled_handler(caplog):
    """HTTPException(404) must be passed through to FastAPI's default 4xx
    response shape and MUST NOT log api.unhandled_exception (otherwise every
    404 from a typo'd URL would page on-call).
    """
    caplog.set_level(logging.ERROR, logger="app.main")
    iso = _isolated_app()
    async with AsyncClient(transport=ASGITransport(app=iso, raise_app_exceptions=False), base_url="http://test") as ac:
        resp = await ac.get("/notfound")
    assert resp.status_code == 404
    assert resp.json() == {"detail": "nope"}

    unhandled_logs = [
        r for r in caplog.records
        if r.name == "app.main" and getattr(r, "message", "") == "api.unhandled_exception"
    ]
    assert unhandled_logs == [], (
        "HTTPException(404) leaked into the unhandled-exception handler — "
        "every 4xx would log as a 5xx"
    )
