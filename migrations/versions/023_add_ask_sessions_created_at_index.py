"""Add created_at index to ask_sessions for retention purge query.

The retention purge deletes expired sessions with
``DELETE FROM ask_sessions WHERE created_at < :cutoff``.
Without an index on ``created_at`` this becomes a sequential scan on
every scheduled run.  The B-tree index makes the purge O(log n).

Revision ID: 023
Revises: 022
"""

from alembic import op


revision = "023"
down_revision = "022"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        "idx_ask_sessions_created_at",
        "ask_sessions",
        ["created_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "idx_ask_sessions_created_at",
        table_name="ask_sessions",
    )
