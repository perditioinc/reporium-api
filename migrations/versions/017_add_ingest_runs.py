"""Add ingest_runs table for pipeline run history.

Revision ID: 017
Revises: 016
Create Date: 2026-03-24
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "017"
down_revision = "016"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "ingest_runs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("run_mode", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False, server_default="running"),
        sa.Column("repos_upserted", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("repos_processed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("errors", JSONB(), nullable=True),
        sa.Column(
            "started_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("finished_at", sa.TIMESTAMP(timezone=True), nullable=True),
    )
    op.create_index("ix_ingest_runs_started_at", "ingest_runs", ["started_at"])
    op.create_index("ix_ingest_runs_status", "ingest_runs", ["status"])


def downgrade() -> None:
    op.drop_index("ix_ingest_runs_status", "ingest_runs")
    op.drop_index("ix_ingest_runs_started_at", "ingest_runs")
    op.drop_table("ingest_runs")
