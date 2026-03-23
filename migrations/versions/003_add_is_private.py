"""Add is_private to repos and enforce NOT NULL

Revision ID: 003
Revises: 002
Create Date: 2026-03-23

The is_private column was added to the live DB manually before this migration
existed. This migration makes the schema authoritative and enforces NOT NULL.

All forks are public by default (you cannot fork a private repo on GitHub).
perditioinc's own private repos are excluded by the ingestion pipeline before
they ever reach the API — the pipeline calls r.get('private') from the GitHub
API and skips any private repo.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Column may already exist (added manually). Use IF NOT EXISTS guard.
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'repos' AND column_name = 'is_private'
            ) THEN
                ALTER TABLE repos ADD COLUMN is_private BOOLEAN NOT NULL DEFAULT false;
            ELSE
                -- Backfill any NULLs (all existing repos are public forks)
                UPDATE repos SET is_private = false WHERE is_private IS NULL;
                ALTER TABLE repos ALTER COLUMN is_private SET NOT NULL;
                ALTER TABLE repos ALTER COLUMN is_private SET DEFAULT false;
            END IF;
        END $$;
    """)


def downgrade() -> None:
    op.drop_column("repos", "is_private")
