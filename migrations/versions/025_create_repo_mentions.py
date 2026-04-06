"""Create repo_mentions table for tracking HN / social mentions.

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
        "repo_mentions",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "repo_id",
            sa.dialects.postgresql.UUID(as_uuid=True),
            sa.ForeignKey("repos.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("source", sa.String, nullable=False),
        sa.Column("external_id", sa.String, nullable=False),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("url", sa.Text, nullable=True),
        sa.Column("score", sa.Integer, nullable=True),
        sa.Column("comment_count", sa.Integer, nullable=True),
        sa.Column("author", sa.String, nullable=True),
        sa.Column("published_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column(
            "fetched_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    op.create_index("idx_repo_mentions_repo_id", "repo_mentions", ["repo_id"])
    op.create_unique_constraint(
        "uq_repo_mentions_repo_source_ext",
        "repo_mentions",
        ["repo_id", "source", "external_id"],
    )


def downgrade() -> None:
    op.drop_constraint("uq_repo_mentions_repo_source_ext", "repo_mentions", type_="unique")
    op.drop_index("idx_repo_mentions_repo_id", table_name="repo_mentions")
    op.drop_table("repo_mentions")
