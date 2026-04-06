"""Add btree index on repo_embeddings(repo_id).

The repo_embeddings table is joined by repo_id in every graph and similarity
query.  The primary key already covers exact lookups, but an explicit btree
index ensures the planner can use it for range scans, ANY() filters, and
lateral joins without ambiguity.

Revision ID: 024
Revises: 023
"""

import sqlalchemy as sa
from alembic import op


revision = "024"
down_revision = "023"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        "idx_repo_embeddings_repo_id",
        "repo_embeddings",
        ["repo_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "idx_repo_embeddings_repo_id",
        table_name="repo_embeddings",
    )
