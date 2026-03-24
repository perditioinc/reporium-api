"""Add quality_signals JSONB column to repos

Revision ID: 015
Revises: 014
Create Date: 2026-03-24
"""
from alembic import op

revision = '015'
down_revision = '014'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE repos
        ADD COLUMN IF NOT EXISTS quality_signals JSONB
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_repos_quality_signals
        ON repos USING gin (quality_signals)
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_repos_quality_signals")
    op.execute("ALTER TABLE repos DROP COLUMN IF EXISTS quality_signals")
