"""Audit log ORM model (KAN-governance).

Stores a row for every auditable API request -- all /intelligence/* calls, and
optionally any request with the X-Sandbox header.
"""

from datetime import datetime

from sqlalchemy import Boolean, Float, Integer, String, Text, TIMESTAMP
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.database import Base


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now(), index=True,
    )
    api_key_hash: Mapped[str | None] = mapped_column(String(64), index=True)  # SHA-256 of key
    endpoint: Mapped[str] = mapped_column(String(100), nullable=False)
    method: Mapped[str] = mapped_column(String(10), nullable=False)
    request_summary: Mapped[str | None] = mapped_column(Text)  # Truncated/redacted
    response_status: Mapped[int] = mapped_column(Integer, nullable=False)
    model_used: Mapped[str | None] = mapped_column(String(50), nullable=True)
    tokens_input: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tokens_output: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sandbox: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
