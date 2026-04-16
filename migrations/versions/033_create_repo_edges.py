"""Create repo_edges and repo_edges_history tables.

repo_edges — live knowledge graph edge set, rebuilt nightly by
scripts/build_knowledge_graph.py in reporium-ingestion.

Schema design decisions:
  - confidence FLOAT (0-1): separate from weight; DEPENDS_ON=0.95,
    COMPATIBLE_WITH=overlap_ratio (floor 0.3), ALTERNATIVE_TO=0.70/0.40
  - metadata JSONB: per-edge evidence (shared tags, category, dep name)
  - ingest_run_id INTEGER FK → ingest_runs(id) ON DELETE SET NULL:
    every edge is traceable to the pipeline run that created it
  - UNIQUE (source_repo_id, target_repo_id, edge_type): upsert-safe

repo_edges_history — append-only temporal archive.
  Before each nightly rebuild, the live edges are copied here with
  valid_until = NOW(). Enables "what did the graph look like on date X"
  queries and post-mortem auditing.

Handles pre-existing repo_edges table: if build_knowledge_graph.py was
run before this migration (creating a differently-structured table), the
migration renames the old table to repo_edges_legacy and creates fresh.

Revision ID: 033
Revises: 032
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB


revision = "033"
down_revision = "032"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Handle pre-existing repo_edges table from old build_knowledge_graph.py
    # (it used `evidence` instead of `metadata` and had no ingest_run_id).
    # Guard: only rename if repo_edges exists AND repo_edges_legacy does not —
    # prevents a retry from renaming the freshly-created table over a prior backup.
    op.execute("""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_name = 'repo_edges'
            ) AND NOT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_name = 'repo_edges_legacy'
            ) THEN
                ALTER TABLE repo_edges RENAME TO repo_edges_legacy;
                RAISE NOTICE 'Renamed pre-existing repo_edges to repo_edges_legacy';
            END IF;
        END;
        $$
    """)

    # ── repo_edges ─────────────────────────────────────────────────────────────
    op.create_table(
        "repo_edges",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "source_repo_id",
            UUID(as_uuid=True),
            sa.ForeignKey("repos.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "target_repo_id",
            UUID(as_uuid=True),
            sa.ForeignKey("repos.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("edge_type", sa.String(32), nullable=False),
        sa.Column("weight", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="0.5"),
        sa.Column("metadata", JSONB(), nullable=True, server_default="{}"),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column(
            "ingest_run_id",
            sa.Integer(),
            sa.ForeignKey("ingest_runs.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.UniqueConstraint(
            "source_repo_id", "target_repo_id", "edge_type",
            name="uq_repo_edges_src_tgt_type",
        ),
    )

    # Use IF NOT EXISTS: the pre-existing repo_edges (renamed to repo_edges_legacy)
    # carried these index names along with it, so a plain CREATE INDEX would fail.
    op.execute("CREATE INDEX IF NOT EXISTS idx_repo_edges_source ON repo_edges(source_repo_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_repo_edges_target ON repo_edges(target_repo_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_repo_edges_type ON repo_edges(edge_type)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_repo_edges_ingest_run ON repo_edges(ingest_run_id)")
    # Partial index for high-confidence edges used by the frontend/API
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_repo_edges_high_confidence "
        "ON repo_edges(edge_type, confidence) WHERE confidence >= 0.7"
    )

    # ── repo_edges_history ────────────────────────────────────────────────────
    # Lightweight count-log: one row per edge_type per rebuild run.
    # Records aggregate edge counts for velocity tracking and monitoring.
    # (A full per-edge temporal archive is out of scope for Wave 2 and will be
    # added in a future migration once the atomic-swap rebuild is in place.)
    op.create_table(
        "repo_edges_history",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.Integer(), nullable=True),
        sa.Column("edge_type", sa.String(32), nullable=False),
        sa.Column("edge_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("sample_edges", JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
    )

    op.create_index(
        "idx_repo_edges_history_created_at",
        "repo_edges_history",
        ["created_at"],
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_repo_edges_high_confidence")
    op.drop_index("idx_repo_edges_ingest_run", table_name="repo_edges")
    op.drop_index("idx_repo_edges_type", table_name="repo_edges")
    op.drop_index("idx_repo_edges_target", table_name="repo_edges")
    op.drop_index("idx_repo_edges_source", table_name="repo_edges")
    op.drop_index("idx_repo_edges_history_created_at", table_name="repo_edges_history")
    op.drop_table("repo_edges_history")
    op.drop_table("repo_edges")

    # Restore pre-existing table if it was renamed during upgrade
    op.execute("""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_name = 'repo_edges_legacy'
            ) THEN
                ALTER TABLE repo_edges_legacy RENAME TO repo_edges;
            END IF;
        END;
        $$
    """)
