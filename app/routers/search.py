from fastapi import APIRouter, Depends, Query
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.models.repo import Repo
from app.routers.library import _repo_to_summary
from app.schemas.repo import RepoSummary

router = APIRouter()

MAX_RESULTS = 20


@router.get("/search", response_model=list[RepoSummary])
async def search_repos(
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
        )
        .order_by(Repo.activity_score.desc())
        .limit(MAX_RESULTS)
    )
    result = await db.execute(stmt)
    repos = result.scalars().all()
    return [_repo_to_summary(r) for r in repos]
