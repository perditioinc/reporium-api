"""Create audit_logs table with indexes.

Stores per-request audit entries for governance, cost tracking, and sandbox
replay.  Indexes on (timestamp) and (api_key_hash, timestamp) support the
/admin/audit query patterns.

Revision ID: 025
Revises: 024
"""

import sqlalchemy as sa
from alembic import op


revision = "025"
down_revision = "024"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "audit_logs",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "timestamp",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("api_key_hash", sa.String(64), nullable=True),
        sa.Column("endpoint", sa.String(100), nullable=False),
        sa.Column("method", sa.String(10), nullable=False),
        sa.Column("request_summary", sa.Text, nullable=True),
        sa.Column("response_status", sa.Integer, nullable=False),
        sa.Column("model_used", sa.String(50), nullable=True),
        sa.Column("tokens_input", sa.Integer, nullable=True),
        sa.Column("tokens_output", sa.Integer, nullable=True),
        sa.Column("cost_usd", sa.Float, nullable=True),
        sa.Column("latency_ms", sa.Integer, nullable=True),
        sa.Column(
            "sandbox",
            sa.Boolean,
            nullable=False,
            server_default=sa.text("false"),
        ),
    )

    op.create_index("idx_audit_logs_timestamp", "audit_logs", ["timestamp"])
    op.create_index(
        "idx_audit_logs_key_ts",
        "audit_logs",
        ["api_key_hash", "timestamp"],
    )


def downgrade() -> None:
    op.drop_index("idx_audit_logs_key_ts", table_name="audit_logs")
    op.drop_index("idx_audit_logs_timestamp", table_name="audit_logs")
    op.drop_table("audit_logs")
