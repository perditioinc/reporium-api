"""Add indexes on repos for common query patterns

Revision ID: 008
Revises: 007
Create Date: 2026-03-24

Adds indexes on columns used in ORDER BY and WHERE clauses across
/library/full, /repos, and /stats endpoints. At 10K repos these
become table scans without indexes.

Refs: https://github.com/perditioinc/reporium-api/issues/32
"""
from typing import Sequence, Union

from alembic import op

revision: str = "008"
down_revision: Union[str, None] = "007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ORDER BY COALESCE(parent_stars, stargazers_count) DESC — /library/full default sort
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_repos_stars
        ON repos (COALESCE(parent_stars, stargazers_count, 0) DESC)
    """)
    # WHERE is_fork = true/false — /repos filter, /stats fork counts
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_repos_is_fork ON repos (is_fork)
    """)
    # WHERE is_private = false — every public query
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_repos_is_private ON repos (is_private)
    """)
    # ORDER BY updated_at DESC — /repos default sort
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_repos_updated_at ON repos (updated_at DESC)
    """)
    # activity_score — future sort option
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_repos_activity_score ON repos (activity_score DESC)
    """)
    # full_name — exact and prefix lookups (btree is enough without pg_trgm)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_repos_full_name ON repos (full_name)
        WHERE full_name IS NOT NULL
    """)


def downgrade() -> None:
    for idx in [
        "ix_repos_stars", "ix_repos_is_fork", "ix_repos_is_private",
        "ix_repos_updated_at", "ix_repos_activity_score", "ix_repos_full_name",
    ]:
        op.execute(f"DROP INDEX IF EXISTS {idx}")
