"""Add pros_cons and pros_cons_generated_at columns to repos table.

Stores AI-generated developer-focused evaluation: pros, cons, best_for,
avoid_if, community_verdict, comparable_to.

Revision ID: 028
Revises: 027
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB


revision = "028"
down_revision = "027"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("repos", sa.Column("pros_cons", JSONB, nullable=True))
    op.add_column(
        "repos",
        sa.Column("pros_cons_generated_at", sa.TIMESTAMP(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("repos", "pros_cons_generated_at")
    op.drop_column("repos", "pros_cons")
