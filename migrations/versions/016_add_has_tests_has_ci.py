"""Add has_tests and has_ci boolean columns to repos

Revision ID: 016
Revises: 015
Create Date: 2026-03-24
"""
from alembic import op

revision = '016'
down_revision = '015'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE repos ADD COLUMN IF NOT EXISTS has_tests BOOLEAN")
    op.execute("ALTER TABLE repos ADD COLUMN IF NOT EXISTS has_ci BOOLEAN")


def downgrade() -> None:
    op.execute("ALTER TABLE repos DROP COLUMN IF EXISTS has_tests")
    op.execute("ALTER TABLE repos DROP COLUMN IF EXISTS has_ci")
