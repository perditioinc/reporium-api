"""
KAN-83 / KAN-124: Knowledge graph edges endpoint.

Returns repo-to-repo similarity edges computed on the fly from pgvector
embeddings (HNSW cosine similarity). Each repo's top-K nearest neighbours
become graph edges, giving full coverage across the entire library.

Previously read from a static `repo_edges` table populated by a naive
category-matching script.  The new approach uses the existing 384-dim
all-MiniLM-L6-v2 embeddings and the HNSW index for fast ANN queries.
"""

import hashlib
import logging

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.cache import cache
from app.database import get_db
from app.embeddings import get_embedding_model
from app.rate_limit import rate_limit_storage
from app.utils import vec_to_pg

logger = logging.getLogger(__name__)

CACHE_TTL_GRAPH_EDGES = 3600  # 1 hr
CACHE_TTL_GRAPH_SEARCH = 1800  # 30 min

_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
_EMBEDDING_DIMENSION = 384

CACHE_TTL_GRAPH_EDGES = 3600  # 1 hr

router = APIRouter(tags=["Graph"])
_limiter = Limiter(key_func=get_remote_address, storage_uri=rate_limit_storage)


@router.get("/graph/edges")
@_limiter.limit("20/minute")
async def get_graph_edges(
    request: Request,
    limit: int = Query(default=500, ge=1, le=10000),
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
    # --- Redis cache check ---
    cache_key = f"graph_edges:{limit}:{min_similarity}:{neighbours}"
    cached = await cache.get(cache_key)
    if cached is not None:
        response = JSONResponse(content=cached)
        response.headers["Cache-Control"] = "public, max-age=3600"
        return response

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
        ),
        -- Orphan rescue: repos with embeddings but no edges above threshold
        orphan_edges AS (
            SELECT DISTINCT ON (LEAST(e1.repo_id, e2.repo_id), GREATEST(e1.repo_id, e2.repo_id))
                e1.repo_id AS source_id,
                e2.repo_id AS target_id,
                1 - (e1.embedding_vec <=> e2.embedding_vec) AS similarity
            FROM repo_embeddings e1
            CROSS JOIN LATERAL (
                SELECT e2_inner.repo_id,
                       e2_inner.embedding_vec
                FROM repo_embeddings e2_inner
                WHERE e2_inner.repo_id != e1.repo_id
                ORDER BY e1.embedding_vec <=> e2_inner.embedding_vec
                LIMIT 1
            ) e2
            WHERE NOT EXISTS (
                SELECT 1 FROM deduped d
                WHERE d.source_id = e1.repo_id OR d.target_id = e1.repo_id
            )
            ORDER BY LEAST(e1.repo_id, e2.repo_id), GREATEST(e1.repo_id, e2.repo_id),
                     similarity DESC
        ),
        all_edges AS (
            SELECT source_id, target_id, similarity FROM deduped
            UNION
            SELECT source_id, target_id, similarity FROM orphan_edges
        )
        SELECT
            ae.similarity,
            r1.name        AS source_name,
            r1.description AS source_description,
            r1.primary_category AS source_category,
            r1.owner       AS source_owner,
            r2.name        AS target_name,
            r2.description AS target_description,
            r2.primary_category AS target_category,
            r2.owner       AS target_owner
        FROM all_edges ae
        JOIN repos r1 ON r1.id = ae.source_id AND r1.is_private = false
        JOIN repos r2 ON r2.id = ae.target_id AND r2.is_private = false
        ORDER BY ae.similarity DESC
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

    # Count repos with and without embeddings for diagnostics
    counts = await db.execute(text("""
        SELECT
            (SELECT COUNT(*) FROM repos WHERE is_private = false) AS total_public,
            (SELECT COUNT(DISTINCT re.repo_id)
             FROM repo_embeddings re
             JOIN repos r ON r.id = re.repo_id
             WHERE r.is_private = false
               AND re.embedding_vec IS NOT NULL) AS with_embeddings
    """))
    count_row = counts.fetchone()

    result_payload = {
        "total": len(edges),
        "total_repos": len(repo_ids),
        "total_public_repos": count_row.total_public if count_row else 0,
        "repos_with_embeddings": count_row.with_embeddings if count_row else 0,
        "edgeTypes": ["SIMILAR_TO"],
        "edges": edges,
    }

    # Store in Redis cache
    await cache.set(cache_key, result_payload, ttl=CACHE_TTL_GRAPH_EDGES)

    response = JSONResponse(content=result_payload)
    response.headers["Cache-Control"] = "public, max-age=3600"
    return response


# ---------------------------------------------------------------------------
# Helper: build edge dicts from rows (shared by /graph/edges and search)
# ---------------------------------------------------------------------------

def _rows_to_edges(rows) -> list[dict]:
    """Convert DB rows (with source_*/target_* columns) to edge dicts."""
    return [
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


# ---------------------------------------------------------------------------
# GET /graph/edges/search — semantic graph search
# ---------------------------------------------------------------------------

@router.get("/graph/edges/search")
@_limiter.limit("10/minute")
async def search_graph_edges(
    request: Request,
    query: str = Query(..., min_length=1, description="Natural-language search query"),
    top_k: int = Query(default=10, ge=1, le=50,
                       description="Number of most-similar seed repos to find"),
    neighbours: int = Query(default=3, ge=1, le=20,
                            description="Neighbours to expand per seed repo"),
    min_similarity: float = Query(default=0.5, ge=0.0, le=1.0,
                                  description="Minimum cosine similarity threshold"),
    db: AsyncSession = Depends(get_db),
):
    """
    Semantic graph search: embed a free-text query, find the top-K most
    similar repos via pgvector, then expand each seed's nearest neighbours
    to build a subgraph of edges.  Results are cached in Redis (TTL 30 min).
    """
    # --- Redis cache ---
    query_hash = hashlib.sha256(query.lower().strip().encode()).hexdigest()[:16]
    cache_key = f"graph_search:{query_hash}:{top_k}:{neighbours}:{min_similarity}"
    cached = await cache.get(cache_key)
    if cached is not None:
        return cached

    # 1. Embed the query
    model = get_embedding_model()
    query_vec = model.encode(query)
    vec_str = vec_to_pg(query_vec)

    # 2. Find top_k most similar repos to the query embedding
    # 3. For each seed, find its `neighbours` nearest neighbours
    # 4. Build edges between seeds and their neighbours, plus inter-neighbour
    sql = text("""
        WITH seeds AS (
            SELECT re.repo_id,
                   1 - (re.embedding_vec <=> CAST(:vec AS vector)) AS query_sim
            FROM repo_embeddings re
            JOIN repos r ON r.id = re.repo_id
            WHERE r.is_private = false
              AND re.embedding_vec IS NOT NULL
            ORDER BY re.embedding_vec <=> CAST(:vec AS vector)
            LIMIT :top_k
        ),
        -- Expand each seed's neighbourhood
        expanded AS (
            SELECT
                s.repo_id      AS source_id,
                e2.repo_id     AS target_id,
                1 - (e1.embedding_vec <=> e2.embedding_vec) AS similarity
            FROM seeds s
            JOIN repo_embeddings e1 ON e1.repo_id = s.repo_id
            CROSS JOIN LATERAL (
                SELECT e2_inner.repo_id,
                       e2_inner.embedding_vec
                FROM repo_embeddings e2_inner
                WHERE e2_inner.repo_id != s.repo_id
                ORDER BY e1.embedding_vec <=> e2_inner.embedding_vec
                LIMIT :neighbours
            ) e2
            WHERE 1 - (e1.embedding_vec <=> e2.embedding_vec) >= :min_sim
        ),
        deduped AS (
            SELECT DISTINCT ON (LEAST(source_id, target_id), GREATEST(source_id, target_id))
                source_id, target_id, similarity
            FROM expanded
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
    """)

    result = await db.execute(sql, {
        "vec": vec_str,
        "top_k": top_k,
        "neighbours": neighbours,
        "min_sim": min_similarity,
    })
    rows = result.fetchall()
    edges = _rows_to_edges(rows)

    # Count distinct repos in the subgraph
    repo_names: set[str] = set()
    for row in rows:
        repo_names.add(row.source_name)
        repo_names.add(row.target_name)

    payload = {
        "query": query,
        "total": len(edges),
        "total_repos": len(repo_names),
        "edgeTypes": ["SIMILAR_TO"],
        "edges": edges,
    }

    await cache.set(cache_key, payload, ttl=CACHE_TTL_GRAPH_SEARCH)
    return payload


# ---------------------------------------------------------------------------
# GET /metrics/embeddings — embedding coverage diagnostics
# ---------------------------------------------------------------------------

@router.get("/metrics/embeddings")
@_limiter.limit("30/minute")
async def get_embedding_metrics(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Returns embedding coverage metrics: how many public repos have
    embeddings vs total, the model name, and vector dimension.
    """
    counts = await db.execute(text("""
        SELECT
            (SELECT COUNT(*) FROM repos WHERE is_private = false) AS total_public,
            (SELECT COUNT(DISTINCT re.repo_id)
             FROM repo_embeddings re
             JOIN repos r ON r.id = re.repo_id
             WHERE r.is_private = false
               AND re.embedding_vec IS NOT NULL) AS with_embeddings
    """))
    row = counts.fetchone()
    total = row.total_public if row else 0
    with_emb = row.with_embeddings if row else 0
    coverage = round((with_emb / total * 100) if total > 0 else 0.0, 2)

    return {
        "total_public_repos": total,
        "repos_with_embeddings": with_emb,
        "coverage_percent": coverage,
        "model": _EMBEDDING_MODEL_NAME,
        "dimension": _EMBEDDING_DIMENSION,
    }
