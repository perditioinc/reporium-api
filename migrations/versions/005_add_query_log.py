"""Add query_log table for /intelligence/ask logging

Revision ID: 005
Revises: 004
Create Date: 2026-03-23

Stores one row per intelligence query. Prerequisite for semantic caching,
cost tracking, and abuse detection.

Refs: https://github.com/perditioinc/reporium-api/issues/36
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "005"
down_revision: Union[str, None] = "004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS query_log (
            id                BIGSERIAL PRIMARY KEY,
            timestamp         TIMESTAMPTZ NOT NULL DEFAULT now(),
            question          TEXT NOT NULL,
            answer_truncated  TEXT,
            sources           JSONB,
            tokens_prompt     INTEGER,
            tokens_completion INTEGER,
            cost_usd          NUMERIC(10, 6),
            hashed_ip         TEXT,
            latency_ms        INTEGER,
            model             TEXT,
            cache_hit         BOOLEAN NOT NULL DEFAULT false
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS ix_query_log_timestamp ON query_log (timestamp)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_query_log_hashed_ip ON query_log (hashed_ip)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS query_log;")
