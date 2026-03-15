"""Initial schema

Revision ID: 001
Revises:
Create Date: 2026-01-01 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "repos",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("name", sa.Text, nullable=False, unique=True),
        sa.Column("owner", sa.Text, nullable=False),
        sa.Column("description", sa.Text),
        sa.Column("is_fork", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("forked_from", sa.Text),
        sa.Column("primary_language", sa.Text),
        sa.Column("github_url", sa.Text, nullable=False),
        sa.Column("fork_sync_state", sa.Text),
        sa.Column("behind_by", sa.Integer, server_default="0"),
        sa.Column("ahead_by", sa.Integer, server_default="0"),
        sa.Column("upstream_created_at", sa.TIMESTAMP(timezone=True)),
        sa.Column("forked_at", sa.TIMESTAMP(timezone=True)),
        sa.Column("your_last_push_at", sa.TIMESTAMP(timezone=True)),
        sa.Column("upstream_last_push_at", sa.TIMESTAMP(timezone=True)),
        sa.Column("parent_stars", sa.Integer),
        sa.Column("parent_forks", sa.Integer),
        sa.Column("parent_is_archived", sa.Boolean, server_default="false"),
        sa.Column("commits_last_7_days", sa.Integer, server_default="0"),
        sa.Column("commits_last_30_days", sa.Integer, server_default="0"),
        sa.Column("commits_last_90_days", sa.Integer, server_default="0"),
        sa.Column("readme_summary", sa.Text),
        sa.Column("activity_score", sa.Integer, server_default="0"),
        sa.Column("ingested_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("github_updated_at", sa.TIMESTAMP(timezone=True)),
    )

    op.create_table(
        "repo_tags",
        sa.Column("repo_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("repos.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("tag", sa.Text, primary_key=True),
    )

    op.create_table(
        "repo_categories",
        sa.Column("repo_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("repos.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("category_id", sa.Text, primary_key=True),
        sa.Column("category_name", sa.Text, nullable=False),
        sa.Column("is_primary", sa.Boolean, server_default="false"),
    )

    op.create_table(
        "repo_builders",
        sa.Column("repo_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("repos.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("login", sa.Text, primary_key=True),
        sa.Column("display_name", sa.Text),
        sa.Column("org_category", sa.Text),
        sa.Column("is_known_org", sa.Boolean, server_default="false"),
    )

    op.create_table(
        "repo_ai_dev_skills",
        sa.Column("repo_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("repos.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("skill", sa.Text, primary_key=True),
    )

    op.create_table(
        "repo_pm_skills",
        sa.Column("repo_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("repos.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("skill", sa.Text, primary_key=True),
    )

    op.create_table(
        "repo_languages",
        sa.Column("repo_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("repos.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("language", sa.Text, primary_key=True),
        sa.Column("bytes", sa.Integer, nullable=False),
        sa.Column("percentage", sa.Float, nullable=False),
    )

    op.create_table(
        "repo_commits",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("repo_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("repos.id", ondelete="CASCADE"), nullable=False),
        sa.Column("sha", sa.Text, nullable=False),
        sa.Column("message", sa.Text, nullable=False),
        sa.Column("author", sa.Text),
        sa.Column("committed_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("url", sa.Text),
    )
    op.create_index("ix_repo_commits_repo_id_committed_at", "repo_commits", ["repo_id", sa.text("committed_at DESC")])

    op.create_table(
        "trend_snapshots",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("snapshotted_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("tag", sa.Text, nullable=False),
        sa.Column("category", sa.Text),
        sa.Column("repo_count", sa.Integer, nullable=False),
        sa.Column("commit_count_7d", sa.Integer, nullable=False, server_default="0"),
    )
    op.create_index("ix_trend_snapshots_snapshotted_at", "trend_snapshots", [sa.text("snapshotted_at DESC")])

    op.create_table(
        "gap_analysis",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("generated_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("skill", sa.Text, nullable=False),
        sa.Column("severity", sa.Text, nullable=False),
        sa.Column("repo_count", sa.Integer, nullable=False),
        sa.Column("why", sa.Text),
        sa.Column("trend", sa.Text),
        sa.Column("essential_repos", postgresql.JSONB),
    )

    op.create_table(
        "repo_embeddings",
        sa.Column("repo_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("repos.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("embedding", sa.Text),  # stored as JSON; use pgvector type directly if available
        sa.Column("model", sa.Text, nullable=False, server_default=sa.text("'nomic-embed-text'")),
        sa.Column("generated_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("NOW()")),
    )

    op.create_table(
        "ingestion_log",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("started_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("completed_at", sa.TIMESTAMP(timezone=True)),
        sa.Column("mode", sa.Text, nullable=False),
        sa.Column("repos_fetched", sa.Integer, server_default="0"),
        sa.Column("repos_updated", sa.Integer, server_default="0"),
        sa.Column("api_calls_made", sa.Integer, server_default="0"),
        sa.Column("errors", postgresql.JSONB, server_default=sa.text("'[]'")),
        sa.Column("status", sa.Text, nullable=False, server_default=sa.text("'running'")),
    )


def downgrade() -> None:
    op.drop_table("ingestion_log")
    op.drop_table("repo_embeddings")
    op.drop_table("gap_analysis")
    op.drop_table("trend_snapshots")
    op.drop_table("repo_commits")
    op.drop_table("repo_languages")
    op.drop_table("repo_pm_skills")
    op.drop_table("repo_ai_dev_skills")
    op.drop_table("repo_builders")
    op.drop_table("repo_categories")
    op.drop_table("repo_tags")
    op.drop_table("repos")
