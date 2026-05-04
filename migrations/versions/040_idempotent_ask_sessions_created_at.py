"""Idempotently ensure ask_sessions.created_at column exists (KAN-223).

Production DB started failing with `column "created_at" does not exist`
when /intelligence/ask called _load_session_turns() per intelligence.py:2140.
The function catches the error broadly and returns [] so /ask still 200's,
but conversational memory loads as empty — silent feature degradation
since whenever the column went missing.

Migration 021 originally creates ask_sessions WITH `created_at TIMESTAMPTZ
DEFAULT NOW()`. The exact reason this column is absent from prod is
unclear — possibilities: (a) table was created manually before migrations
were wired, then 021 was skipped to avoid 'table exists' error, leaving
the column-less manual table in place; (b) snapshot restored from a
pre-021 backup; (c) the column was dropped by an out-of-band operation.

Fix: idempotent ALTER TABLE — `ADD COLUMN IF NOT EXISTS` so this is
safe whether the column is missing or already present. NOT NULL with a
DEFAULT NOW() so any existing rows get a sensible timestamp (these rows
were unaddressable by the 24h cutoff anyway, so backfilling to NOW() is
fine — they'll fall outside the window after 24h naturally).

Revision ID: 040
Revises: 039
"""

from alembic import op


revision = "040"
down_revision = "039"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ADD COLUMN IF NOT EXISTS is idempotent — safe whether the column
    # missing (KAN-223 scenario) or present (most envs).
    op.execute("""
        ALTER TABLE ask_sessions
        ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    """)


def downgrade() -> None:
    # Don't drop on downgrade — losing this column re-introduces the
    # silent failure described in KAN-223. Down-migration is for schema
    # rollback testing; if needed, run the SQL manually.
    pass
