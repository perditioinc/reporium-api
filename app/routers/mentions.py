import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.mention import RepoMention
from app.models.repo import Repo

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Mentions"])


@router.get("/repos/{repo_id}/mentions", response_model=list[dict])
async def get_repo_mentions(
    repo_id: UUID,
    source: str | None = Query(default=None, description="Filter by source (hackernews, reddit, youtube)"),
    db: AsyncSession = Depends(get_db),
):
    """Return all social mentions for a repo, ordered by score descending."""

    # Verify repo exists
    repo = await db.get(Repo, repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="Repo not found")

    stmt = (
        select(RepoMention)
        .where(RepoMention.repo_id == repo_id)
        .order_by(RepoMention.score.desc().nullslast())
    )
    if source:
        stmt = stmt.where(RepoMention.source == source)

    result = await db.execute(stmt)
    mentions = result.scalars().all()

    return [
        {
            "id": str(m.id),
            "source": m.source,
            "external_id": m.external_id,
            "title": m.title,
            "url": m.url,
            "score": m.score,
            "comment_count": m.comment_count,
            "author": m.author,
            "published_at": m.published_at.isoformat() if m.published_at else None,
            "fetched_at": m.fetched_at.isoformat() if m.fetched_at else None,
        }
        for m in mentions
    ]
