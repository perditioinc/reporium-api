"""Add embedding_vec vector(384) column with HNSW index for fast similarity search

Revision ID: 007
Revises: 006
Create Date: 2026-03-23

The existing repo_embeddings.embedding column stores 384-dim vectors as a JSON
TEXT array. This works but requires fetching all ~1400 rows into Python and computing
cosine similarity in a loop — O(N) memory and time, won't survive 10K repos.

This migration:
1. Adds embedding_vec vector(384) column
2. Backfills from the existing TEXT column via ::vector cast
3. Creates an HNSW index with cosine distance ops for sub-millisecond ANN search

At 10K repos the HNSW index makes similarity search ~1000x faster than the Python loop.

Refs: https://github.com/perditioinc/reporium-api/issues/41
"""
from typing import Sequence, Union

from alembic import op

revision: str = "007"
down_revision: Union[str, None] = "006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Ensure extension is enabled (idempotent)
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    # Add native vector column
    op.execute("""
        ALTER TABLE repo_embeddings
        ADD COLUMN IF NOT EXISTS embedding_vec vector(384)
    """)
    # Backfill from the JSON text column — cast directly via pgvector's text input
    op.execute("""
        UPDATE repo_embeddings
        SET embedding_vec = embedding::vector
        WHERE embedding IS NOT NULL AND embedding_vec IS NULL
    """)
    # HNSW index: cosine distance (1 - cosine_similarity), m=16 ef_construction=64
    # CONCURRENTLY not allowed inside a transaction; use CREATE INDEX without it here
    # (alembic runs in a transaction by default; acceptable for current scale)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_repo_embeddings_vec_hnsw
        ON repo_embeddings
        USING hnsw (embedding_vec vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_repo_embeddings_vec_hnsw")
    op.execute("ALTER TABLE repo_embeddings DROP COLUMN IF EXISTS embedding_vec")
