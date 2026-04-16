"""Extend ingest_runs with graph build provenance columns.

Adds four optional columns to ingest_runs (migration 017):
  - checkpoint_data JSONB   — phase-level resume state for crash recovery
  - prev_edge_counts JSONB  — edge counts from the previous graph build run,
                               used by reporium-audit regression checks
  - git_sha TEXT            — git SHA of reporium-ingestion at run time
  - triggered_by TEXT       — 'schedule' | 'workflow_dispatch' | 'manual' | etc.

All columns are nullable; existing rows are unaffected.

Revision ID: 032
Revises: 031
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


revision = "032"
down_revision = "031"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("ingest_runs", sa.Column("checkpoint_data", JSONB(), nullable=True))
    op.add_column("ingest_runs", sa.Column("prev_edge_counts", JSONB(), nullable=True))
    op.add_column("ingest_runs", sa.Column("git_sha", sa.Text(), nullable=True))
    op.add_column("ingest_runs", sa.Column("triggered_by", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("ingest_runs", "triggered_by")
    op.drop_column("ingest_runs", "git_sha")
    op.drop_column("ingest_runs", "prev_edge_counts")
    op.drop_column("ingest_runs", "checkpoint_data")
