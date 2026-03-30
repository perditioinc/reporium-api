"""Add ask_sessions table for conversational memory (KAN-158).

Stores the last N question/answer turns per session so /ask and /query
can prepend prior context to Claude's messages array.

Each session is identified by a client-provided UUID.  Rows are lightweight
(question + answer text only — no embeddings) and indexed on (session_id, turn_number)
for fast last-N-turns retrieval.

Revision ID: 021
Revises: 020
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID


revision = "021"
down_revision = "020"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "ask_sessions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("session_id", UUID(as_uuid=True), nullable=False),
        sa.Column("turn_number", sa.Integer, nullable=False, server_default="0"),
        sa.Column("question", sa.Text, nullable=False),
        sa.Column("answer", sa.Text, nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index(
        "idx_ask_sessions_session_id_turn",
        "ask_sessions",
        ["session_id", "turn_number"],
    )


def downgrade() -> None:
    op.drop_index("idx_ask_sessions_session_id_turn", table_name="ask_sessions")
    op.drop_table("ask_sessions")
