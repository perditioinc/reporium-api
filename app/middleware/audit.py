"""Audit middleware for the API gateway (KAN-governance).

Intercepts every request and, when auditing is warranted, persists a row to
the ``audit_logs`` table via a fire-and-forget ``asyncio.create_task``.

Auditing is triggered when:
  1. The request targets any ``/intelligence/*`` endpoint, OR
  2. The request carries the ``X-Sandbox: true`` header.

The middleware is feature-flagged behind ``AUDIT_ENABLED=1``.  When the env
var is absent or ``0`` the middleware is a transparent pass-through.

Privacy: request bodies are truncated to 500 chars and run through
``privacy.redact_pii`` before storage.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)

# Maximum chars of the request body stored in request_summary.
_MAX_SUMMARY_LEN = 500


def _is_audit_enabled() -> bool:
    return os.environ.get("AUDIT_ENABLED", "0") == "1"


def _hash_key(raw: str | None) -> str | None:
    if not raw:
        return None
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class AuditMiddleware(BaseHTTPMiddleware):
    """Starlette middleware that writes audit log entries asynchronously."""

    async def dispatch(self, request: Request, call_next) -> Response:
        if not _is_audit_enabled():
            return await call_next(request)

        path = request.url.path
        is_intelligence = path.startswith("/intelligence")
        is_sandbox = (request.headers.get("x-sandbox") or "").lower() == "true"

        if not is_intelligence and not is_sandbox:
            return await call_next(request)

        # Capture timing
        start = time.perf_counter()
        response: Response = await call_next(request)
        latency_ms = round((time.perf_counter() - start) * 1000)

        # Build summary (truncated + redacted)
        try:
            body_bytes = await request.body()
            summary = body_bytes.decode("utf-8", errors="replace")[:_MAX_SUMMARY_LEN]
        except Exception:
            summary = ""

        try:
            from app.privacy import redact_pii
            summary = redact_pii(summary)
        except Exception:
            pass

        # Derive api_key_hash from any of the auth headers
        raw_key = (
            request.headers.get("x-app-token")
            or request.headers.get("x-admin-key")
            or request.headers.get("authorization", "").removeprefix("Bearer ").strip()
            or None
        )

        entry = {
            "api_key_hash": _hash_key(raw_key),
            "endpoint": path,
            "method": request.method,
            "request_summary": summary if is_sandbox else summary[:200],
            "response_status": response.status_code,
            "latency_ms": latency_ms,
            "sandbox": is_sandbox,
        }

        asyncio.create_task(_persist_audit(entry))
        return response


async def _persist_audit(entry: dict) -> None:
    """Fire-and-forget: insert one row into audit_logs."""
    try:
        from app.database import async_session_factory
        from app.models.audit_log import AuditLog

        async with async_session_factory() as session:
            log = AuditLog(**entry)
            session.add(log)
            await session.commit()
    except Exception:
        logger.warning("Failed to persist audit log entry", exc_info=True)
