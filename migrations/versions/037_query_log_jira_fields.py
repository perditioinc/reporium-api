"""Add JIRA integration fields to query_log (admin endpoint write-back).

Changes:
  1. jira_ticket_key VARCHAR(32) NULL  — JIRA ticket created from the ask
  2. jira_status     VARCHAR(32) NULL  — "open" | "in_progress" | "done" | "rejected"
  3. action_taken    TEXT NULL         — freeform note of resolution
  4. cost_cents      INTEGER NULL      — aggregated spend (input+output tokens × model rate)
  5. sentiment       VARCHAR(16) NULL  — "positive" | "negative" | "neutral"

  Indexes:
  - idx_query_log_created_at_desc  (created_at DESC) — dashboard time-range queries
  - idx_query_log_jira_ticket_key  partial on non-null jira_ticket_key — ticket-key lookups

Revision ID: 037
Revises: 036
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "037"
down_revision = "036"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("query_log", sa.Column("jira_ticket_key", sa.String(32), nullable=True))
    op.add_column("query_log", sa.Column("jira_status", sa.String(32), nullable=True))
    op.add_column("query_log", sa.Column("action_taken", sa.Text(), nullable=True))
    op.add_column("query_log", sa.Column("cost_cents", sa.Integer(), nullable=True))
    op.add_column("query_log", sa.Column("sentiment", sa.String(16), nullable=True))

    # Index for time-range dashboard queries (DESC matches typical ORDER BY)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_query_log_created_at_desc
        ON query_log (timestamp DESC)
    """)

    # Sparse index — only rows where a JIRA ticket has been assigned
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_query_log_jira_ticket_key
        ON query_log (jira_ticket_key)
        WHERE jira_ticket_key IS NOT NULL
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_query_log_jira_ticket_key")
    op.execute("DROP INDEX IF EXISTS idx_query_log_created_at_desc")
    op.drop_column("query_log", "sentiment")
    op.drop_column("query_log", "cost_cents")
    op.drop_column("query_log", "action_taken")
    op.drop_column("query_log", "jira_status")
    op.drop_column("query_log", "jira_ticket_key")
