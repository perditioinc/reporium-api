"""Add community health signal columns to repos table.

Stores contributor count, release info, issue close rate, PR merge rate,
GitHub Discussions flag, and community health percentage from the free
GitHub REST API.

Revision ID: 026
Revises: 025
"""

import sqlalchemy as sa
from alembic import op


revision = "026"
down_revision = "025"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("repos", sa.Column("contributors_count", sa.Integer, nullable=True))
    op.add_column("repos", sa.Column("release_count", sa.Integer, nullable=True))
    op.add_column(
        "repos",
        sa.Column("latest_release_date", sa.TIMESTAMP(timezone=True), nullable=True),
    )
    op.add_column("repos", sa.Column("issue_close_rate", sa.Float, nullable=True))
    op.add_column("repos", sa.Column("pr_merge_rate", sa.Float, nullable=True))
    op.add_column("repos", sa.Column("has_discussions", sa.Boolean, nullable=True))
    op.add_column("repos", sa.Column("community_health_pct", sa.Integer, nullable=True))


def downgrade() -> None:
    op.drop_column("repos", "community_health_pct")
    op.drop_column("repos", "has_discussions")
    op.drop_column("repos", "pr_merge_rate")
    op.drop_column("repos", "issue_close_rate")
    op.drop_column("repos", "latest_release_date")
    op.drop_column("repos", "release_count")
    op.drop_column("repos", "contributors_count")
