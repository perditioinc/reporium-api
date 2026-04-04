from fastapi import APIRouter, Depends, Query, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy import or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.embeddings import get_embedding_model
from app.models.repo import Repo
from app.routers.library import _repo_to_summary
from app.schemas.repo import RepoSemanticResult, RepoSummary
from app.utils import vec_to_pg

router = APIRouter(tags=["Search"])
limiter = Limiter(key_func=get_remote_address)

MAX_RESULTS = 20
MAX_SEMANTIC_RESULTS = 50


@router.get("/search", response_model=list[RepoSummary])
@limiter.limit("30/minute")
async def search_repos(
    request: Request,
    q: str = Query(..., min_length=1),
    db: AsyncSession = Depends(get_db),
) -> list[RepoSummary]:
    """
    Text search on name and description.
    Falls back gracefully if embeddings are not available.
    """
    search = f"%{q}%"
    stmt = (
        select(Repo)
        .where(
            Repo.is_private == False,  # noqa: E712 — SQLAlchemy requires == not `is`
            or_(
                Repo.name.ilike(search),
                Repo.description.ilike(search),
                Repo.readme_summary.ilike(search),
            ),
        )
        .options(
            selectinload(Repo.tags),
            selectinload(Repo.categories),
            selectinload(Repo.builders),
            selectinload(Repo.ai_dev_skills),
            selectinload(Repo.pm_skills),
            selectinload(Repo.languages),
            selectinload(Repo.taxonomy),
        )
        .order_by(Repo.activity_score.desc())
        .limit(MAX_RESULTS)
    )
    result = await db.execute(stmt)
    repos = result.scalars().all()
    return [_repo_to_summary(r) for r in repos]


async def _semantic_candidate_rows(
    db: AsyncSession,
    *,
    query_embedding,
    limit: int,
):
    vec_str = vec_to_pg(query_embedding)
    result = await db.execute(
        text("""
            SELECT r.id AS repo_id,
                   1 - (re.embedding_vec <=> CAST(:vec AS vector)) AS similarity
            FROM repo_embeddings re
            JOIN repos r ON r.id = re.repo_id
            WHERE re.embedding_vec IS NOT NULL
              AND r.is_private = false
            ORDER BY re.embedding_vec <=> CAST(:vec AS vector)
            LIMIT :limit
        """),
        {"vec": vec_str, "limit": limit},
    )
    return result.fetchall()


async def _hydrate_semantic_results(
    db: AsyncSession,
    *,
    candidate_rows,
) -> list[RepoSemanticResult]:
    if not candidate_rows:
        return []

    repo_ids = [row.repo_id for row in candidate_rows]
    similarity_by_id = {row.repo_id: float(row.similarity) for row in candidate_rows}

    stmt = (
        select(Repo)
        .where(Repo.id.in_(repo_ids))
        .options(
            selectinload(Repo.tags),
            selectinload(Repo.categories),
            selectinload(Repo.builders),
            selectinload(Repo.ai_dev_skills),
            selectinload(Repo.pm_skills),
            selectinload(Repo.languages),
            selectinload(Repo.taxonomy),
        )
    )
    result = await db.execute(stmt)
    repo_map = {repo.id: repo for repo in result.scalars().all()}

    ordered_results: list[RepoSemanticResult] = []
    for repo_id in repo_ids:
        repo = repo_map.get(repo_id)
        if repo is None:
            continue
        ordered_results.append(
            RepoSemanticResult(
                **_repo_to_summary(repo).model_dump(),
                similarity=round(similarity_by_id[repo_id], 6),
            )
        )

    return ordered_results


async def _full_text_fallback(
    db: AsyncSession,
    *,
    query: str,
    limit: int,
) -> list[RepoSemanticResult]:
    """ILIKE fallback used when repo_embeddings table has no vectors yet."""
    search = f"%{query}%"
    stmt = (
        select(Repo)
        .where(
            or_(
                Repo.name.ilike(search),
                Repo.description.ilike(search),
                Repo.readme_summary.ilike(search),
            )
        )
        .options(
            selectinload(Repo.tags),
            selectinload(Repo.categories),
            selectinload(Repo.builders),
            selectinload(Repo.ai_dev_skills),
            selectinload(Repo.pm_skills),
            selectinload(Repo.languages),
            selectinload(Repo.taxonomy),
        )
        .order_by(Repo.activity_score.desc())
        .limit(limit)
    )
    result = await db.execute(stmt)
    repos = result.scalars().all()
    return [
        RepoSemanticResult(**_repo_to_summary(r).model_dump(), similarity=0.0)
        for r in repos
    ]


async def _semantic_search(
    db: AsyncSession,
    *,
    query: str,
    limit: int,
) -> list[RepoSemanticResult]:
    model = get_embedding_model()
    query_embedding = model.encode(query)
    candidate_rows = await _semantic_candidate_rows(
        db,
        query_embedding=query_embedding,
        limit=limit,
    )
    if not candidate_rows:
        # No embeddings exist yet — fall back to full-text search so the
        # endpoint is useful before the embedding pipeline has run.
        return await _full_text_fallback(db, query=query, limit=limit)
    return await _hydrate_semantic_results(db, candidate_rows=candidate_rows)


@router.get("/search/semantic", response_model=list[RepoSemanticResult])
@limiter.limit("10/minute")
async def semantic_search_repos(
    request: Request,
    q: str = Query(..., min_length=1),
    limit: int = Query(default=10, ge=1, le=MAX_SEMANTIC_RESULTS),
    db: AsyncSession = Depends(get_db),
) -> list[RepoSemanticResult]:
    return await _semantic_search(db, query=q, limit=limit)
