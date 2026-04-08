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
    op.create_table(
        "repo_industries",
        sa.Column("repo_id", UUID(as_uuid=True), sa.ForeignKey("repos.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("industry", sa.Text(), primary_key=True),
    )
    op.create_index("ix_repo_industries_repo_id", "repo_industries", ["repo_id"])


def downgrade():
    op.drop_index("ix_repo_industries_repo_id", table_name="repo_industries")
    op.drop_table("repo_industries")
