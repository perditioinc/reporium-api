"""Extend repo_embeddings for append-only history tracking.

Previously, repo_embeddings had repo_id as primary key, enforcing one row per repo.
This migration converts the table to an append-only history table:

  1. Drop the repo_id-as-primary-key constraint (repo_id becomes a plain FK column).
  2. Add a UUID primary key so multiple history rows can coexist per repo.
  3. Add is_current BOOLEAN (TRUE for the latest embedding, FALSE for prior versions).
  4. Add ingest_run_id FK → ingest_runs(id) ON DELETE SET NULL for provenance.
  5. Create a partial unique index: UNIQUE (repo_id) WHERE is_current = TRUE
     — guarantees at most one current embedding per repo, enforced by the DB.
  6. Backfill existing rows: set is_current = TRUE (all existing rows are current).

Also extends repo_edges_history from the count-log schema (migration 033) to a
full per-edge temporal archive needed by the Wave 3 atomic-swap rebuild:

  - Adds source_repo_id, target_repo_id, weight, confidence, metadata,
    ingest_run_id, valid_from, valid_until as nullable columns.
  - Existing count-log rows (run_id, edge_type, edge_count, sample_edges) are
    preserved; those columns remain for backward compatibility.
  - The new per-edge columns are nullable so the ALTER succeeds with existing rows.

The HNSW vector index (migration 007) is on embedding_vec; it will now include
historical rows. Since ANN similarity queries must filter WHERE is_current = TRUE,
a separate index on (repo_id, is_current) is added for those queries.

Revision ID: 034
Revises: 033
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID, JSONB


revision = "034"
down_revision = "033"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── repo_embeddings: convert to append-only ──────────────────────────────

    # 1. Drop the repo_id primary key constraint.
    #    In PostgreSQL, PK constraints have auto-generated names; use raw SQL.
    op.execute("""
        ALTER TABLE repo_embeddings
        DROP CONSTRAINT IF EXISTS repo_embeddings_pkey
    """)

    # 2. Add UUID primary key column.
    op.execute("""
        ALTER TABLE repo_embeddings
        ADD COLUMN IF NOT EXISTS id UUID DEFAULT gen_random_uuid()
    """)
    # Backfill any rows that were added before the column (shouldn't happen, but safe)
    op.execute("""
        UPDATE repo_embeddings SET id = gen_random_uuid() WHERE id IS NULL
    """)
    op.execute("""
        ALTER TABLE repo_embeddings ADD PRIMARY KEY (id)
    """)

    # 3. Add is_current column (default TRUE so all existing rows stay current).
    op.add_column(
        "repo_embeddings",
        sa.Column("is_current", sa.Boolean(), nullable=False, server_default="TRUE"),
    )

    # 4. Add ingest_run_id FK.
    op.add_column(
        "repo_embeddings",
        sa.Column(
            "ingest_run_id",
            sa.Integer(),
            sa.ForeignKey("ingest_runs.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )

    # 5. Partial unique index: only one current embedding per repo.
    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS uq_repo_embeddings_current
        ON repo_embeddings (repo_id)
        WHERE is_current = TRUE
    """)

    # 6. Index for fast "latest embedding for this repo" queries.
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_repo_embeddings_repo_current
        ON repo_embeddings (repo_id, is_current)
    """)

    # 7. Index for ingest run provenance lookups.
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_repo_embeddings_run
        ON repo_embeddings (ingest_run_id)
        WHERE ingest_run_id IS NOT NULL
    """)

    # ── repo_edges_history: extend from count-log to full per-edge archive ───
    #
    # Migration 033 created repo_edges_history as a lightweight count-log.
    # Wave 3's atomic-swap rebuild archives full edge records here before each
    # rebuild.  These columns are nullable so existing count-log rows survive.

    op.add_column(
        "repo_edges_history",
        sa.Column("source_repo_id", UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "repo_edges_history",
        sa.Column("target_repo_id", UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "repo_edges_history",
        sa.Column("weight", sa.Float(), nullable=True),
    )
    op.add_column(
        "repo_edges_history",
        sa.Column("confidence", sa.Float(), nullable=True),
    )
    op.add_column(
        "repo_edges_history",
        sa.Column("metadata", JSONB(), nullable=True),
    )
    op.add_column(
        "repo_edges_history",
        sa.Column(
            "ingest_run_id",
            sa.Integer(),
            sa.ForeignKey("ingest_runs.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.add_column(
        "repo_edges_history",
        sa.Column("valid_from", sa.TIMESTAMP(timezone=True), nullable=True),
    )
    op.add_column(
        "repo_edges_history",
        sa.Column("valid_until", sa.TIMESTAMP(timezone=True), nullable=True),
    )

    # Index for temporal queries: "what did the graph look like on date X"
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_repo_edges_history_valid_until "
        "ON repo_edges_history(edge_type, valid_until) "
        "WHERE valid_until IS NOT NULL"
    )
    # Index for per-source lookups in the archive
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_repo_edges_history_source "
        "ON repo_edges_history(source_repo_id) "
        "WHERE source_repo_id IS NOT NULL"
    )


def downgrade() -> None:
    # repo_edges_history extensions
    op.execute("DROP INDEX IF EXISTS idx_repo_edges_history_source")
    op.execute("DROP INDEX IF EXISTS idx_repo_edges_history_valid_until")
    op.drop_column("repo_edges_history", "valid_until")
    op.drop_column("repo_edges_history", "valid_from")
    op.drop_column("repo_edges_history", "ingest_run_id")
    op.drop_column("repo_edges_history", "metadata")
    op.drop_column("repo_edges_history", "confidence")
    op.drop_column("repo_edges_history", "weight")
    op.drop_column("repo_edges_history", "target_repo_id")
    op.drop_column("repo_edges_history", "source_repo_id")

    # repo_embeddings revert
    op.execute("DROP INDEX IF EXISTS idx_repo_embeddings_run")
    op.execute("DROP INDEX IF EXISTS idx_repo_embeddings_repo_current")
    op.execute("DROP INDEX IF EXISTS uq_repo_embeddings_current")
    op.drop_column("repo_embeddings", "ingest_run_id")
    op.drop_column("repo_embeddings", "is_current")
    # Restore repo_id as primary key (only valid if there is at most one row per repo)
    op.execute("DELETE FROM repo_embeddings WHERE id NOT IN (SELECT MIN(id) FROM repo_embeddings GROUP BY repo_id)")
    op.execute("ALTER TABLE repo_embeddings DROP COLUMN IF EXISTS id")
    op.execute("ALTER TABLE repo_embeddings ADD PRIMARY KEY (repo_id)")
