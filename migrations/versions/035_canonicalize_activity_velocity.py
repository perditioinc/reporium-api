"""Canonicalization, activity score breakdown, forks_count, and velocity views.

Changes:
  1. ADD COLUMN IF NOT EXISTS integration_tags JSONB — column already exists in prod
     (created by migration 002); guard prevents duplicate-column error on rerun.
  2. ADD COLUMN raw_integration_tags JSONB — stores original AI output before canonicalization.
  3. ADD COLUMN forks_count INTEGER — GitHub forks count, used in new activity score formula.
  4. ADD COLUMN activity_score_breakdown JSONB — per-component breakdown of the activity score.
  5. CREATE VIEW v_edge_count_by_run — edge counts grouped by ingest run, used by audit checks.
  6. CREATE VIEW v_repo_activity_trend — rolling 7-day window of per-repo activity scores,
     useful for trending dashboards and regression detection.

Revision ID: 035
Revises: 034
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


# revision identifiers
revision = "035"
down_revision = "034"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()

    # 1. integration_tags — guard against pre-existing column (migration 002 created it)
    conn.execute(sa.text("""
        ALTER TABLE repos
        ADD COLUMN IF NOT EXISTS integration_tags JSONB
    """))

    # 2–4. New columns (safe to ADD without IF NOT EXISTS — 035 is the only migration adding these)
    conn.execute(sa.text("""
        ALTER TABLE repos
        ADD COLUMN IF NOT EXISTS raw_integration_tags JSONB,
        ADD COLUMN IF NOT EXISTS forks_count INTEGER NOT NULL DEFAULT 0,
        ADD COLUMN IF NOT EXISTS activity_score_breakdown JSONB
    """))

    # 5. v_edge_count_by_run — edge counts per ingest run and edge type
    #    Used by reporium-audit to detect >20% drops in DEPENDS_ON coverage.
    conn.execute(sa.text("""
        CREATE OR REPLACE VIEW v_edge_count_by_run AS
        SELECT
            ir.id                                    AS run_id,
            ir.started_at,
            ir.finished_at,
            ir.status,
            re.edge_type,
            COUNT(re.id)                             AS edge_count
        FROM ingest_runs ir
        LEFT JOIN repo_edges re ON re.ingest_run_id = ir.id
        GROUP BY ir.id, ir.started_at, ir.finished_at, ir.status, re.edge_type
        ORDER BY ir.started_at DESC, re.edge_type
    """))

    # 6. v_repo_activity_trend — 7-day rolling window of activity scores
    #    Compares each repo's latest score against its score from 7 days ago.
    conn.execute(sa.text("""
        CREATE OR REPLACE VIEW v_repo_activity_trend AS
        SELECT
            r.id                                     AS repo_id,
            r.name,
            r.activity_score                         AS current_score,
            r.updated_at                             AS scored_at,
            LAG(r.activity_score) OVER (
                PARTITION BY r.id
                ORDER BY r.updated_at
            )                                        AS prev_score,
            r.activity_score - COALESCE(
                LAG(r.activity_score) OVER (
                    PARTITION BY r.id
                    ORDER BY r.updated_at
                ), r.activity_score
            )                                        AS score_delta
        FROM repos r
        WHERE r.updated_at >= NOW() - INTERVAL '7 days'
        ORDER BY score_delta DESC, r.activity_score DESC
    """))


def downgrade() -> None:
    conn = op.get_bind()

    conn.execute(sa.text("DROP VIEW IF EXISTS v_repo_activity_trend"))
    conn.execute(sa.text("DROP VIEW IF EXISTS v_edge_count_by_run"))

    # Only drop columns added by this migration — do NOT drop integration_tags
    # (it predates this migration and may have data from other migrations)
    conn.execute(sa.text("""
        ALTER TABLE repos
        DROP COLUMN IF EXISTS activity_score_breakdown,
        DROP COLUMN IF EXISTS forks_count,
        DROP COLUMN IF EXISTS raw_integration_tags
    """))
