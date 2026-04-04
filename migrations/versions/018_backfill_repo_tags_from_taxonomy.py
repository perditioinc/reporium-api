"""Backfill repo_tags from repo_taxonomy and primary_language.

repo_tags was wiped to ~3 rows by a sparse ingestion run that predated
the skip-empty guard (PR #126).  This migration restores a meaningful
tag set from already-populated sources so the tag cloud and repo-card
tags are non-empty without requiring a full re-ingestion run.

Sources (in priority order):
  1. repo_taxonomy.raw_value — all dimensions (skill_area, ai_trend,
     modality, use_case, industry, deployment_context) — these were
     written by the AI enricher and represent rich semantic labels.
  2. repos.primary_language — a single-word tag that is present for
     93 % of repos and is immediately useful for filtering.

Both inserts use ON CONFLICT DO NOTHING so they are idempotent and
safe to re-run.

Revision ID: 018
Revises: 017
"""

from alembic import op

revision = "018"
down_revision = "017"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── 1. Tags from taxonomy ─────────────────────────────────────────────────
    op.execute("""
        INSERT INTO repo_tags (repo_id, tag)
        SELECT DISTINCT rt.repo_id, rt.raw_value
        FROM   repo_taxonomy rt
        WHERE  rt.raw_value IS NOT NULL
          AND  trim(rt.raw_value) != ''
        ON CONFLICT DO NOTHING
    """)

    # ── 2. Primary language as a tag ──────────────────────────────────────────
    op.execute("""
        INSERT INTO repo_tags (repo_id, tag)
        SELECT r.id, r.primary_language
        FROM   repos r
        WHERE  r.primary_language IS NOT NULL
          AND  trim(r.primary_language) != ''
        ON CONFLICT DO NOTHING
    """)


def downgrade() -> None:
    # Downgrade is intentionally a no-op: we cannot reliably distinguish
    # tags that existed before this migration from tags inserted by it.
    pass
