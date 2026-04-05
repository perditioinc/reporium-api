"""Add token_hash column to ask_sessions for ownership binding (Issue #235).

Security hardening: the ask_sessions table previously had no ownership
binding. Any caller presenting a valid X-App-Token could fetch the last N
conversation turns of any session by guessing/knowing the session UUID.
This migration adds a SHA-256 hash of the X-App-Token that first created
the session. Loads filter by matching hash so one app token cannot read
conversations stored under another app token.

Existing rows (pre-migration) carry NULL token_hash — they are treated as
"legacy / unbound" by the application code and any token may read them.
We deliberately do NOT backfill.

Revision ID: 022
Revises: 021
"""

import sqlalchemy as sa
from alembic import op


revision = "022"
down_revision = "021"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "ask_sessions",
        sa.Column("token_hash", sa.String(length=64), nullable=True),
    )
    op.create_index(
        "idx_ask_sessions_session_id_token_hash",
        "ask_sessions",
        ["session_id", "token_hash"],
    )


def downgrade() -> None:
    op.drop_index(
        "idx_ask_sessions_session_id_token_hash",
        table_name="ask_sessions",
    )
    op.drop_column("ask_sessions", "token_hash")
