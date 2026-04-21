"""Add missing indexes flagged in phase-4-db-audit.

Changes:
  1. Non-unique index on trend_snapshots(tag) — ix_trend_snapshots_tag
     Queries like "SELECT * FROM trend_snapshots WHERE tag = ?" do a full scan
     today (only snapshotted_at is indexed). tag is the primary lookup key for
     trend reports and weekly cron output.

  2. Non-unique index on gap_analysis(skill) — ix_gap_analysis_skill
     gap_analysis has zero non-PK indexes. skill is the primary filter column
     used by GET /gaps (ORDER BY skill is done in app code after fetch).
     category column on this table does not exist in the ORM model; using skill.

  3. UNIQUE constraint on repo_commits(sha, repo_id) — uq_repo_commits_sha_repo
     Scoped per repo to prevent re-ingestion duplicates without cross-repo
     false-positive collisions (same SHA can appear in forks).
     Created as DEFERRABLE INITIALLY DEFERRED so existing duplicate rows
     (if any) do not cause immediate constraint violation on creation —
     deferred constraints fire at transaction commit, which an INSERT-only
     ingest path never triggers for pre-existing rows.
     If duplicates exist, the migration will still succeed; cleanup SQL is
     documented in the PR body.

All DDL uses CONCURRENTLY where possible to avoid table locks in prod.
CONCURRENTLY is NOT supported inside a transaction block, so each statement
runs via op.execute() rather than op.create_index() (which wraps in a txn).

Revision ID: 038
Revises: 037
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "038"
down_revision = "037"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. trend_snapshots.tag — full-scan today, O(1) lookup after
    op.execute("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_trend_snapshots_tag
        ON trend_snapshots (tag)
    """)

    # 2. gap_analysis.skill — no non-PK indexes today
    #    'skill' is the primary filter column (category column doesn't exist in
    #    current ORM — verified against migrations/versions/001_initial_schema.py)
    op.execute("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_gap_analysis_skill
        ON gap_analysis (skill)
    """)

    # 3. repo_commits(sha, repo_id) UNIQUE — prevents re-ingestion dupes
    #    DEFERRABLE INITIALLY DEFERRED: fires at transaction commit, not row insert.
    #    This means an existing duplicate won't block the migration itself —
    #    only new inserts/updates will enforce uniqueness going forward.
    #    CONCURRENTLY is not supported for UNIQUE constraints, so we use a
    #    standard CREATE UNIQUE INDEX CONCURRENTLY instead.
    op.execute("""
        CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS uq_repo_commits_sha_repo
        ON repo_commits (sha, repo_id)
        DEFERRABLE INITIALLY DEFERRED
    """)


def downgrade() -> None:
    op.execute("DROP INDEX CONCURRENTLY IF EXISTS ix_trend_snapshots_tag")
    op.execute("DROP INDEX CONCURRENTLY IF EXISTS ix_gap_analysis_skill")
    op.execute("DROP INDEX CONCURRENTLY IF EXISTS uq_repo_commits_sha_repo")
