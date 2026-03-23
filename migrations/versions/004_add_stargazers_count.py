"""Add stargazers_count to repos for non-fork (built) repos

Revision ID: 004
Revises: 003
Create Date: 2026-03-23

parent_stars stores the upstream star count for forks.
For built (non-fork) repos parent_stars is NULL, so their own star count
was never stored — they showed 0 on reporium.com.

This column holds the repo's own GitHub stargazers count and is populated
by the ingestion pipeline for non-fork repos.

Fixes: https://github.com/perditioinc/reporium-api/issues/13
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'repos' AND column_name = 'stargazers_count'
            ) THEN
                ALTER TABLE repos ADD COLUMN stargazers_count INTEGER;
            END IF;
        END $$;
    """)


def downgrade() -> None:
    op.drop_column("repos", "stargazers_count")
