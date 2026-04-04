"""Add security_signals JSONB column to repos.

Stores manually-curated and auto-detected security risk metadata per repo:
  - risk_level: 'critical' | 'high' | 'medium' | 'low' | null
  - incident_reported: bool (publicly disclosed security incident)
  - incident_date: ISO date string
  - incident_url: link to CVE / blog post / GitHub advisory
  - incident_summary: one-sentence description

Revision ID: 020
Revises: 019
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB


revision = "020"
down_revision = "019"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("repos", sa.Column("security_signals", JSONB, nullable=True))


def downgrade() -> None:
    op.drop_column("repos", "security_signals")
