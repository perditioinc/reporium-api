"""
KAN-156: GET /repos/{name}/similar  — pure pgvector cosine similarity, no LLM.
         GET /intelligence/recommended — top-N similar repos across a list of seeds.

Uses existing repo_embeddings (384-dim all-MiniLM-L6-v2 vectors).
No Anthropic API calls — $0.00 per request.

/similar       — find repos semantically close to a given repo
/recommended   — given a comma-separated list of recently-viewed repo names,
                 return a deduplicated ranked list of recommendations
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.cache import CACHE_TTL_STATS, cache
from app.database import get_db

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Repos"])

_DEFAULT_SIMILAR_LIMIT = 8
_DEFAULT_REC_LIMIT = 12
_MIN_SIMILARITY = 0.55  # cosine similarity floor — below this, repos are unrelated


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class SimilarRepo(BaseModel):
    name: str
    owner: str
    description: str | None
    primary_language: str | None
    primary_category: str | None
    stars: int | None
    similarity: float
    readme_summary: str | None


class SimilarReposResponse(BaseModel):
    source_repo: str
    similar: list[SimilarRepo]
    total: int


class RecommendedReposResponse(BaseModel):
    seeds: list[str]
    recommended: list[SimilarRepo]
    total: int


# ---------------------------------------------------------------------------
# Shared SQL — single HNSW index scan per seed repo
# ---------------------------------------------------------------------------

_SIMILAR_SQL = text("""
    SELECT
        r.name,
        r.owner,
        r.description,
        r.primary_language,
        r.primary_category,
        r.parent_stars         AS stars,
        r.readme_summary,
        1 - (e2.embedding_vec <=> e1.embedding_vec) AS similarity
    FROM repo_embeddings e1
    JOIN repos seed_r ON seed_r.id = e1.repo_id
    JOIN repo_embeddings e2 ON e2.repo_id != e1.repo_id
    JOIN repos r ON r.id = e2.repo_id
    WHERE seed_r.name = :name
      AND r.is_private = false
      AND r.parent_is_archived = false
      AND e1.embedding_vec IS NOT NULL
      AND e2.embedding_vec IS NOT NULL
      AND 1 - (e2.embedding_vec <=> e1.embedding_vec) >= :min_similarity
    ORDER BY e2.embedding_vec <=> e1.embedding_vec
    LIMIT :limit
""")


def _row_to_similar(row) -> SimilarRepo:
    return SimilarRepo(
        name=row.name,
        owner=row.owner,
        description=row.description,
        primary_language=row.primary_language,
        primary_category=row.primary_category,
        stars=row.stars,
        similarity=round(float(row.similarity), 4),
        readme_summary=row.readme_summary,
    )


# ---------------------------------------------------------------------------
# GET /repos/{name}/similar
# ---------------------------------------------------------------------------

@router.get("/intelligence/similar/{name}", response_model=SimilarReposResponse)
async def similar_repos(
    name: str,
    limit: int = Query(default=_DEFAULT_SIMILAR_LIMIT, ge=1, le=24),
    min_similarity: float = Query(default=_MIN_SIMILARITY, ge=0.0, le=1.0),
    db: AsyncSession = Depends(get_db),
) -> SimilarReposResponse:
    """
    Return repos semantically similar to {name} using pgvector cosine similarity.
    Pure vector search — no LLM, no API credits consumed.

    Results cached for 1 hour. Excludes archived repos and the source repo itself.
    """
    cache_key = f"similar:{name}:{limit}:{min_similarity}"
    cached = await cache.get(cache_key)
    if cached:
        return SimilarReposResponse(**cached)

    result = await db.execute(
        _SIMILAR_SQL,
        {"name": name, "limit": limit, "min_similarity": min_similarity},
    )
    rows = result.fetchall()

    if not rows:
        # Check whether the source repo exists at all
        exists = await db.execute(
            text("SELECT 1 FROM repos WHERE name = :name AND is_private = false LIMIT 1"),
            {"name": name},
        )
        if not exists.fetchone():
            raise HTTPException(status_code=404, detail=f"Repo '{name}' not found")

        # Repo exists but has no embedding or no similar neighbours above threshold
        response = SimilarReposResponse(source_repo=name, similar=[], total=0)
        await cache.set(cache_key, response.model_dump(), ttl=CACHE_TTL_STATS)
        return response

    similar = [_row_to_similar(r) for r in rows]
    response = SimilarReposResponse(source_repo=name, similar=similar, total=len(similar))
    await cache.set(cache_key, response.model_dump(), ttl=CACHE_TTL_STATS)
    return response


# ---------------------------------------------------------------------------
# GET /intelligence/recommended?seeds=repo1,repo2,...
# ---------------------------------------------------------------------------

@router.get("/intelligence/recommended", response_model=RecommendedReposResponse)
async def recommended_repos(
    seeds: str = Query(
        ...,
        description="Comma-separated list of recently-viewed repo names (max 5)",
        min_length=1,
    ),
    limit: int = Query(default=_DEFAULT_REC_LIMIT, ge=1, le=24),
    db: AsyncSession = Depends(get_db),
) -> RecommendedReposResponse:
    """
    Given recently-viewed repos, return a deduplicated ranked recommendation list.
    Each seed contributes similar repos; results are merged and re-ranked by similarity.
    Pure pgvector — no LLM, no API credits.
    """
    seed_names = [s.strip() for s in seeds.split(",") if s.strip()][:5]
    if not seed_names:
        raise HTTPException(status_code=422, detail="At least one seed repo name is required")

    cache_key = f"recommended:{','.join(sorted(seed_names))}:{limit}"
    cached = await cache.get(cache_key)
    if cached:
        return RecommendedReposResponse(**cached)

    # Collect similar repos for all seeds; merge by best similarity score
    merged: dict[str, SimilarRepo] = {}
    for seed in seed_names:
        result = await db.execute(
            _SIMILAR_SQL,
            {"name": seed, "limit": limit * 2, "min_similarity": _MIN_SIMILARITY},
        )
        for row in result.fetchall():
            if row.name in seed_names:
                continue  # don't recommend a repo the user already viewed
            repo = _row_to_similar(row)
            if row.name not in merged or repo.similarity > merged[row.name].similarity:
                merged[row.name] = repo

    ranked = sorted(merged.values(), key=lambda r: r.similarity, reverse=True)[:limit]
    response = RecommendedReposResponse(
        seeds=seed_names,
        recommended=ranked,
        total=len(ranked),
    )
    await cache.set(cache_key, response.model_dump(), ttl=CACHE_TTL_STATS)
    return response
