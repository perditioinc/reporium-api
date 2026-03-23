"""Add enrichment columns for ingestion pipeline

Revision ID: 002
Revises: 001
Create Date: 2026-03-21 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("repos", sa.Column("dependencies", postgresql.JSONB, server_default="[]"))
    op.add_column("repos", sa.Column("problem_solved", sa.Text))
    op.add_column("repos", sa.Column("integration_tags", postgresql.JSONB, server_default="[]"))
    op.add_column("repos", sa.Column("quality_signals", postgresql.JSONB, server_default="{}"))


def downgrade() -> None:
    op.drop_column("repos", "quality_signals")
    op.drop_column("repos", "integration_tags")
    op.drop_column("repos", "problem_solved")
    op.drop_column("repos", "dependencies")
