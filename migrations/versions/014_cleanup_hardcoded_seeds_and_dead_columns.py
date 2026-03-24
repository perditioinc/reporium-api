"""Clean up hardcoded seed data, drop dead columns, and add license_spdx.

Revision ID: 014
Revises: 013
Create Date: 2026-03-24
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "014"
down_revision = "013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Remove the 28 hardcoded seed rows from skill_areas.
    # The taxonomy rebuild pipeline will repopulate from real data.
    op.execute("DELETE FROM skill_areas WHERE id > 0")

    # Drop dead columns defined in migration 002 but never populated or used.
    op.execute("ALTER TABLE repos DROP COLUMN IF EXISTS dependencies")
    op.execute("ALTER TABLE repos DROP COLUMN IF EXISTS quality_signals")

    # Add license_spdx column to repos table.
    op.execute("ALTER TABLE repos ADD COLUMN IF NOT EXISTS license_spdx TEXT")


def downgrade() -> None:
    # Seed data cannot be restored — drop the column additions only.
    op.execute("ALTER TABLE repos DROP COLUMN IF EXISTS license_spdx")
    op.execute("ALTER TABLE repos ADD COLUMN IF NOT EXISTS dependencies JSONB")
    op.execute("ALTER TABLE repos ADD COLUMN IF NOT EXISTS quality_signals JSONB")
