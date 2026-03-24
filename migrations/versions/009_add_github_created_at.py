"""add github_created_at to repos

Revision ID: 009
Revises: 008
Create Date: 2026-03-24
"""
from alembic import op

revision = '009'
down_revision = '008'
branch_labels = None
depends_on = None

def upgrade():
    op.execute("ALTER TABLE repos ADD COLUMN IF NOT EXISTS github_created_at TIMESTAMPTZ")
    op.execute("CREATE INDEX IF NOT EXISTS ix_repos_github_created_at ON repos (github_created_at)")

def downgrade():
    op.execute("DROP INDEX IF EXISTS ix_repos_github_created_at")
    op.execute("ALTER TABLE repos DROP COLUMN IF EXISTS github_created_at")
