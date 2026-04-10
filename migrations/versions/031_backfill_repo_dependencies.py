"""Backfill repo_dependencies from repo_taxonomy (dimension='dependency').

Migration 014 dropped the repos.dependencies JSONB column.
Migration 029 created the proper repo_dependencies table but it was never written to
because the ingestion writer still targeted the dropped column.

Historical dependency data survived in repo_taxonomy (dimension='dependency') because
the ingest router's _TAXONOMY_DIMENSION_MAP routed the field there as a fallback.

This migration:
  1. Copies those rows into repo_dependencies (package_ecosystem='unknown' since
     the source file name was lost — the taxonomy only stored the package name).
  2. Removes the stale dimension='dependency' rows from repo_taxonomy to avoid
     double-counting once the ingestion pipeline is rewired.

Schema-only — no external API calls. Idempotent via ON CONFLICT DO NOTHING.

Revision ID: 031
Revises: 030
"""

from alembic import op


revision = "031"
down_revision = "030"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Backfill from repo_taxonomy rows where dimension='dependency'.
    # repo_taxonomy.raw_value holds the package name; ecosystem is unknown (historical).
    op.execute("""
        INSERT INTO repo_dependencies
            (id, repo_id, package_name, package_ecosystem, is_direct, fetched_at)
        SELECT
            gen_random_uuid(),
            repo_id,
            raw_value AS package_name,
            'unknown' AS package_ecosystem,
            true AS is_direct,
            NOW() AS fetched_at
        FROM (SELECT DISTINCT repo_id, raw_value FROM repo_taxonomy WHERE dimension = 'dependency') dedup
        ON CONFLICT (repo_id, package_name, package_ecosystem) DO NOTHING
    """)

    # Clean up the stale taxonomy rows now that they live in the proper table.
    op.execute("DELETE FROM repo_taxonomy WHERE dimension = 'dependency'")


def downgrade() -> None:
    # Cannot safely reverse: we'd need to know which repo_dependencies rows
    # were backfilled vs. written by the new ingestion pipeline. Leave as-is.
    pass
