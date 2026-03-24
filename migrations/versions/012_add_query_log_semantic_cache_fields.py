"""Add query_log fields needed for semantic caching

Revision ID: 012
Revises: 011
Create Date: 2026-03-24

Adds:
1. answer_full TEXT so cache hits can return the full prior answer
2. question_embedding_vec vector(384) for nearest-neighbor cache lookup
3. HNSW index for semantic cache distance queries
"""
from typing import Sequence, Union

from alembic import op

revision: str = "012"
down_revision: Union[str, None] = "011"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("""
        ALTER TABLE query_log
        ADD COLUMN IF NOT EXISTS answer_full TEXT
    """)
    op.execute("""
        ALTER TABLE query_log
        ADD COLUMN IF NOT EXISTS question_embedding_vec vector(384)
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_query_log_question_embedding_hnsw
        ON query_log
        USING hnsw (question_embedding_vec vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_query_log_question_embedding_hnsw")
    op.execute("ALTER TABLE query_log DROP COLUMN IF EXISTS question_embedding_vec")
    op.execute("ALTER TABLE query_log DROP COLUMN IF EXISTS answer_full")
