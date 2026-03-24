"""Add full_name (owner/name) column to repos for stable identity at scale

Revision ID: 006
Revises: 005
Create Date: 2026-03-23

bare repos.name collides at scale (e.g. two orgs each have a repo called "autogen").
full_name stores the canonical "owner/name" string (e.g. "microsoft/autogen") and
has a UNIQUE constraint so collisions are caught at insert time.

For forks: full_name = forked_from (already "owner/name" format).
For owned repos: full_name = "perditioinc/" + name.

Refs: https://github.com/perditioinc/reporium-api/issues/39
"""
from typing import Sequence, Union

from alembic import op

revision: str = "006"
down_revision: Union[str, None] = "005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE repos ADD COLUMN IF NOT EXISTS full_name TEXT
    """)
    op.execute("""
        UPDATE repos
        SET full_name = forked_from
        WHERE forked_from IS NOT NULL AND full_name IS NULL
    """)
    op.execute("""
        UPDATE repos
        SET full_name = 'perditioinc/' || name
        WHERE forked_from IS NULL AND full_name IS NULL
    """)
    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS uq_repos_full_name ON repos (full_name)
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_repos_full_name")
    op.execute("ALTER TABLE repos DROP COLUMN IF EXISTS full_name")
