"""Create repo_edges_history for tracking edge counts per rebuild run.

Enables velocity views and regression detection (e.g. >50% edge drop
between runs triggers an abort in the graph builder).

Revision ID: 032
Revises: 031
"""

from alembic import op

revision = "032"
down_revision = "031"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS repo_edges_history (
            id SERIAL PRIMARY KEY,
            run_id INTEGER,
            edge_type TEXT NOT NULL,
            edge_count INTEGER NOT NULL DEFAULT 0,
            sample_edges JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)

    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_repo_edges_history_created_at "
        "ON repo_edges_history(created_at DESC)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS repo_edges_history")
