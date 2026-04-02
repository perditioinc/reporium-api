"""
KAN-83 / KAN-124: Knowledge graph edges endpoint.

Returns repo-to-repo similarity edges computed on the fly from pgvector
embeddings (HNSW cosine similarity). Each repo's top-K nearest neighbours
become graph edges, giving full coverage across the entire library.

Previously read from a static `repo_edges` table populated by a naive
category-matching script.  The new approach uses the existing 384-dim
nomic-embed-text embeddings and the HNSW index for fast ANN queries.
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db

router = APIRouter(tags=["Graph"])


@router.get("/graph/edges")
async def get_graph_edges(
    limit: int = Query(default=500, ge=1, le=5000),
    min_similarity: float = Query(default=0.55, ge=0.0, le=1.0,
                                  description="Minimum cosine similarity threshold"),
    neighbours: int = Query(default=8, ge=1, le=30,
                            description="Max neighbours per repo"),
    db: AsyncSession = Depends(get_db),
):
    """
    Returns knowledge graph edges based on pgvector embedding similarity.
    Each repo is connected to its top-K nearest neighbours above the
    similarity threshold.  Edges are SIMILAR_TO with weight = similarity.
    """
    # Use a CTE to find top-K neighbours per repo via HNSW index.
    # The <=> operator returns cosine distance; 1 - distance = similarity.
    # We lateral-join to get the K nearest neighbours per repo efficiently.
    sql = text("""
        WITH ranked AS (
            SELECT
                e1.repo_id   AS source_id,
                e2.repo_id   AS target_id,
                1 - (e1.embedding_vec <=> e2.embedding_vec) AS similarity
            FROM repo_embeddings e1
            CROSS JOIN LATERAL (
                SELECT e2_inner.repo_id,
                       e2_inner.embedding_vec
                FROM repo_embeddings e2_inner
                WHERE e2_inner.repo_id != e1.repo_id
                ORDER BY e1.embedding_vec <=> e2_inner.embedding_vec
                LIMIT :neighbours
            ) e2
            WHERE 1 - (e1.embedding_vec <=> e2.embedding_vec) >= :min_sim
        ),
        deduped AS (
            SELECT DISTINCT ON (LEAST(source_id, target_id), GREATEST(source_id, target_id))
                source_id, target_id, similarity
            FROM ranked
            ORDER BY LEAST(source_id, target_id), GREATEST(source_id, target_id),
                     similarity DESC
        )
        SELECT
            d.similarity,
            r1.name        AS source_name,
            r1.description AS source_description,
            r1.primary_category AS source_category,
            r1.owner       AS source_owner,
            r2.name        AS target_name,
            r2.description AS target_description,
            r2.primary_category AS target_category,
            r2.owner       AS target_owner
        FROM deduped d
        JOIN repos r1 ON r1.id = d.source_id AND r1.is_private = false
        JOIN repos r2 ON r2.id = d.target_id AND r2.is_private = false
        ORDER BY d.similarity DESC
        LIMIT :limit
    """)

    result = await db.execute(sql, {
        "neighbours": neighbours,
        "min_sim": min_similarity,
        "limit": limit,
    })
    rows = result.fetchall()

    edges = [
        {
            "edgeType": "SIMILAR_TO",
            "weight": round(float(row.similarity), 4),
            "evidence": None,
            "source": {
                "name": row.source_name,
                "owner": row.source_owner,
                "description": row.source_description,
                "category": row.source_category,
            },
            "target": {
                "name": row.target_name,
                "owner": row.target_owner,
                "description": row.target_description,
                "category": row.target_category,
            },
        }
        for row in rows
    ]

    # Count distinct repos in result set
    repo_ids = set()
    for row in rows:
        repo_ids.add(row.source_name)
        repo_ids.add(row.target_name)

    return {
        "total": len(edges),
        "total_repos": len(repo_ids),
        "edgeTypes": ["SIMILAR_TO"],
        "edges": edges,
    }
