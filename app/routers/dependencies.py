"""Dependency graph endpoints -- reads from repo_dependencies (populated by SBOM backfill)."""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.db_filters import public_repo_filter
from app.models.dependency import RepoDependency
from app.models.repo import Repo

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Dependencies"])


@router.get("/repos/{repo_id}/dependencies", response_model=list[dict])
async def get_repo_dependencies(
    repo_id: UUID,
    ecosystem: str | None = Query(default=None, description="Filter by ecosystem (pypi, npm, etc.)"),
    db: AsyncSession = Depends(get_db),
):
    """Return all dependencies for a repo, ordered by package_name.

    SECURITY: returns 404 for private repos so they can't be enumerated by
    UUID. Uses the centralized `public_repo_filter()` predicate — see
    app/db_filters.py.
    """

    # Verify repo exists AND is public
    stmt_repo = select(Repo).where(Repo.id == repo_id, public_repo_filter())
    repo = (await db.execute(stmt_repo)).scalar_one_or_none()
    if repo is None:
        raise HTTPException(status_code=404, detail="Repo not found")

    stmt = (
        select(RepoDependency)
        .where(RepoDependency.repo_id == repo_id)
        .order_by(RepoDependency.package_name)
    )
    if ecosystem:
        stmt = stmt.where(RepoDependency.package_ecosystem == ecosystem)

    result = await db.execute(stmt)
    deps = result.scalars().all()

    return [
        {
            "id": str(d.id),
            "package_name": d.package_name,
            "package_ecosystem": d.package_ecosystem,
            "version_constraint": d.version_constraint,
            "is_direct": d.is_direct,
            "fetched_at": d.fetched_at.isoformat() if d.fetched_at else None,
        }
        for d in deps
    ]


@router.get("/dependencies/dependents", response_model=list[dict])
async def get_dependents(
    package: str = Query(..., description="Package name to search for (e.g. 'pytorch', 'langchain')"),
    ecosystem: str | None = Query(default=None, description="Filter by ecosystem"),
    db: AsyncSession = Depends(get_db),
):
    """Return all repos that depend on a given package.

    Enables queries like 'show me all repos using PyTorch'.
    """

    stmt = (
        select(
            Repo.id,
            Repo.name,
            Repo.owner,
            Repo.description,
            Repo.github_url,
            Repo.primary_language,
            RepoDependency.package_ecosystem,
            RepoDependency.version_constraint,
            RepoDependency.is_direct,
        )
        .join(RepoDependency, RepoDependency.repo_id == Repo.id)
        .where(
            func.lower(RepoDependency.package_name) == package.lower(),
            public_repo_filter(),  # SECURITY: never expose private repos
        )
    )
    if ecosystem:
        stmt = stmt.where(RepoDependency.package_ecosystem == ecosystem)

    stmt = stmt.order_by(Repo.name)

    result = await db.execute(stmt)
    rows = result.fetchall()

    return [
        {
            "repo_id": str(r.id),
            "name": r.name,
            "owner": r.owner,
            "description": r.description,
            "github_url": r.github_url,
            "primary_language": r.primary_language,
            "package_ecosystem": r.package_ecosystem,
            "version_constraint": r.version_constraint,
            "is_direct": r.is_direct,
        }
        for r in rows
    ]
