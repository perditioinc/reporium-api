"""Formalize repo_edges table under Alembic control.

The table may already exist from build_knowledge_graph.py's ensure_table().
Uses IF NOT EXISTS guards throughout. Adds new columns (confidence,
updated_at) for the data trust initiative.

Revision ID: 031
Revises: 030
"""

from alembic import op

revision = "031"
down_revision = "030"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create table if it doesn't exist (may have been created by script)
    op.execute("""
        CREATE TABLE IF NOT EXISTS repo_edges (
            id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
            source_repo_id UUID NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
            target_repo_id UUID NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
            edge_type TEXT NOT NULL,
            weight FLOAT DEFAULT 1.0,
            confidence FLOAT DEFAULT 0.5,
            evidence JSONB DEFAULT '{}',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE (source_repo_id, target_repo_id, edge_type)
        )
    """)

    # Add new columns to existing tables that lack them
    op.execute(
        "ALTER TABLE repo_edges ADD COLUMN IF NOT EXISTS confidence FLOAT DEFAULT 0.5"
    )
    op.execute(
        "ALTER TABLE repo_edges ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW()"
    )

    # Indexes
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_repo_edges_source ON repo_edges(source_repo_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_repo_edges_target ON repo_edges(target_repo_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_repo_edges_type ON repo_edges(edge_type)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS repo_edges CASCADE")
