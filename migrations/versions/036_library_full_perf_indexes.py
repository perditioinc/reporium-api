"""Add indexes to speed up /library/full pagination query.

Changes:
  1. Expression index on COALESCE(parent_stars, stargazers_count, 0) DESC for is_private=false
     repos — eliminates full-table sort for ORDER BY in _fetch_page_repos.
  2. Partial index on repos (is_private) for the COUNT(*) query.
  3. Composite indexes on junction tables using native UUID to avoid ::text casts
     (repo_tags, repo_languages, repo_categories, repo_ai_dev_skills,
      repo_pm_skills, repo_builders, repo_taxonomy, repo_industries).

Revision ID: 036
Revises: 035
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "036"
down_revision = "035"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Expression index for ORDER BY COALESCE(parent_stars, stargazers_count, 0) DESC
    # WHERE is_private = false — matches the exact WHERE clause in _fetch_page_repos
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_repos_stars_sort_public
        ON repos (COALESCE(parent_stars, stargazers_count, 0) DESC)
        WHERE is_private = false
    """)

    # Partial index for COUNT(*) WHERE is_private = false
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_repos_is_private_partial
        ON repos (id)
        WHERE is_private = false
    """)

    # Junction table indexes — ensures repo_id UUID lookups are index-only scans
    # (These may already exist as PKs; IF NOT EXISTS guards against duplicates)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_repo_tags_repo_id
        ON repo_tags (repo_id)
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_repo_languages_repo_id
        ON repo_languages (repo_id)
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_repo_categories_repo_id
        ON repo_categories (repo_id)
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_repo_pm_skills_repo_id
        ON repo_pm_skills (repo_id)
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_repo_builders_repo_id
        ON repo_builders (repo_id)
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_repo_taxonomy_repo_id
        ON repo_taxonomy (repo_id)
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_repo_industries_repo_id
        ON repo_industries (repo_id)
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_repos_stars_sort_public")
    op.execute("DROP INDEX IF EXISTS idx_repos_is_private_partial")
    op.execute("DROP INDEX IF EXISTS idx_repo_tags_repo_id")
    op.execute("DROP INDEX IF EXISTS idx_repo_languages_repo_id")
    op.execute("DROP INDEX IF EXISTS idx_repo_categories_repo_id")
    op.execute("DROP INDEX IF EXISTS idx_repo_pm_skills_repo_id")
    op.execute("DROP INDEX IF EXISTS idx_repo_builders_repo_id")
    op.execute("DROP INDEX IF EXISTS idx_repo_taxonomy_repo_id")
    op.execute("DROP INDEX IF EXISTS idx_repo_industries_repo_id")
