"""Create repo_industries junction table.

Revision ID: 030
Revises: 029
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID


revision = "030"
down_revision = "029"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
        CREATE TABLE IF NOT EXISTS repo_industries (
            repo_id UUID REFERENCES repos(id) ON DELETE CASCADE,
            industry TEXT NOT NULL,
            PRIMARY KEY (repo_id, industry)
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS ix_repo_industries_repo_id ON repo_industries(repo_id)")


def downgrade():
    op.drop_index("ix_repo_industries_repo_id", table_name="repo_industries")
    op.drop_table("repo_industries")
