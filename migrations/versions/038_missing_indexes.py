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

  3. UNIQUE index on repo_commits(sha, repo_id) — uq_repo_commits_sha_repo
     Scoped per repo to prevent re-ingestion duplicates without cross-repo
     false-positive collisions (same SHA can appear in forks).

     NOTE: Creating a UNIQUE index (concurrent or not) validates existing rows
     at build time and WILL fail if duplicates exist. A DEFERRABLE constraint
     was considered but DEFERRABLE is a CONSTRAINT attribute, not an INDEX
     attribute — CREATE UNIQUE INDEX ... DEFERRABLE is invalid Postgres DDL.
     Instead, this migration runs a preflight duplicate check and raises
     clearly if any (sha, repo_id) duplicates exist so operators can clean
     them up before re-running.

All DDL uses CONCURRENTLY where possible to avoid table locks in prod.
CONCURRENTLY is NOT supported inside a transaction block, so each statement
runs inside op.get_context().autocommit_block() which temporarily exits
Alembic's per-migration transaction.

Revision ID: 038
Revises: 037
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "038"
down_revision = "037"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Preflight: CREATE UNIQUE INDEX CONCURRENTLY still validates existing rows
    # and will fail if duplicates exist. Fail loudly up-front with a clear
    # message rather than letting Postgres emit a generic unique-violation.
    conn = op.get_bind()
    dupe = conn.execute(
        sa.text(
            """
            SELECT sha, repo_id, COUNT(*) AS n
            FROM repo_commits
            GROUP BY sha, repo_id
            HAVING COUNT(*) > 1
            LIMIT 1
            """
        )
    ).first()
    if dupe is not None:
        raise RuntimeError(
            "repo_commits has duplicate (sha, repo_id) rows — clean up before "
            f"running migration 038. Example: sha={dupe[0]!r} repo_id={dupe[1]!r} "
            f"count={dupe[2]}. See PR body for cleanup SQL."
        )

    # CREATE INDEX CONCURRENTLY cannot run inside a transaction block.
    # autocommit_block() temporarily exits Alembic's wrapping transaction.
    with op.get_context().autocommit_block():
        # 1. trend_snapshots.tag — full-scan today, O(1) lookup after
        op.execute(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_trend_snapshots_tag "
            "ON trend_snapshots (tag)"
        )

        # 2. gap_analysis.skill — no non-PK indexes today
        #    'skill' is the primary filter column (category column doesn't exist
        #    in current ORM — verified against 001_initial_schema.py)
        op.execute(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_gap_analysis_skill "
            "ON gap_analysis (skill)"
        )

        # 3. repo_commits(sha, repo_id) UNIQUE — prevents re-ingestion dupes.
        #    Plain concurrent unique index; deferrability dropped (see header).
        op.execute(
            "CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS uq_repo_commits_sha_repo "
            "ON repo_commits (sha, repo_id)"
        )


def downgrade() -> None:
    # DROP INDEX CONCURRENTLY also cannot run inside a transaction block.
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS ix_trend_snapshots_tag")
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS ix_gap_analysis_skill")
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS uq_repo_commits_sha_repo")
