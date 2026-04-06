"""Audit query helpers for the /admin/audit endpoint (KAN-governance).

Provides the database query logic used by the admin router to list and filter
audit log entries.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.audit_log import AuditLog


async def list_audit_logs(
    session: AsyncSession,
    *,
    api_key_hash: str | None = None,
    endpoint: str | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    sandbox_only: bool = False,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """Return audit log entries matching the given filters."""
    stmt = select(AuditLog).order_by(AuditLog.timestamp.desc())

    if api_key_hash:
        stmt = stmt.where(AuditLog.api_key_hash == api_key_hash)
    if endpoint:
        stmt = stmt.where(AuditLog.endpoint.ilike(f"%{endpoint}%"))
    if date_from:
        stmt = stmt.where(
            AuditLog.timestamp >= datetime(date_from.year, date_from.month, date_from.day, tzinfo=timezone.utc)
        )
    if date_to:
        dt_to = datetime(date_to.year, date_to.month, date_to.day, 23, 59, 59, tzinfo=timezone.utc)
        stmt = stmt.where(AuditLog.timestamp <= dt_to)
    if sandbox_only:
        stmt = stmt.where(AuditLog.sandbox.is_(True))

    stmt = stmt.offset(offset).limit(limit)
    result = await session.execute(stmt)
    rows = result.scalars().all()

    return [
        {
            "id": r.id,
            "timestamp": r.timestamp.isoformat() if r.timestamp else None,
            "api_key_hash": r.api_key_hash,
            "endpoint": r.endpoint,
            "method": r.method,
            "response_status": r.response_status,
            "model_used": r.model_used,
            "tokens_input": r.tokens_input,
            "tokens_output": r.tokens_output,
            "cost_usd": r.cost_usd,
            "latency_ms": r.latency_ms,
            "sandbox": r.sandbox,
        }
        for r in rows
    ]
