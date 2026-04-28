"""Admin endpoint to mark a single repo as ``is_private = true``, with a
dry-run preview and post-mutation cache invalidation.

Built for incident response (2026-04-27 hippo-harvest-assignment leak).
Cloud SQL is private-IP only, so direct ``UPDATE`` from an operator host
isn't possible — this endpoint is the in-VPC corrective path.

Contract — see ``tests/test_admin_mark_private.py``:
  - X-Admin-Key required.
  - 404 when no row matches owner+name.
  - dry_run=true: returns the match info, does not mutate or invalidate.
  - dry_run=false: sets ``is_private = true``, invalidates documented cache
    prefixes, writes an AuditLog row, and returns ``applied=true``.
  - Idempotent: applying to an already-private row is a no-op success.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import require_admin_key
from app.cache_redis import redis_cache
from app.database import get_db
from app.models.audit_log import AuditLog
from app.models.repo import Repo

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Admin"])


# Cache key prefixes that may include the affected repo. Conservative — the
# bleed-stop value of clearing too much beats serving a private repo from a
# stale list. Each prefix is matched via Redis SCAN (``clear_prefix``), so
# this is bounded I/O even on production-sized key spaces.
INVALIDATION_PREFIXES: tuple[str, ...] = (
    "library:",       # /library and /library/full pages
    "repos:",         # /repos list + /repos/{name} detail
    "graph_",         # graph_edges, graph_subgraph, graph_clusters, graph_search
    "trending:",      # /intelligence/trending leaderboards
    "ecosystem:",     # /intelligence/ecosystem/{name}
    "intelligence:",  # portfolio insights, category momentum, similar
    "signals:",       # taxonomy gaps, stale repos, velocity leaders
    "compare:",       # /compare results
    "similar:",       # /intelligence/similar/{name}
    "smart_route:",   # LLM router cache (may cite the repo)
    "llm_response:",  # LLM answer cache (same)
)


class MarkPrivateRequest(BaseModel):
    owner: str = Field(..., min_length=1, max_length=200)
    name: str = Field(..., min_length=1, max_length=200)
    dry_run: bool = Field(
        default=True,
        description=(
            "If true (default), returns match info without mutating. "
            "Operators MUST run dry_run=true first and confirm match_count=1 "
            "before submitting dry_run=false."
        ),
    )


class MarkPrivateMatch(BaseModel):
    id: str
    owner: str
    name: str
    current_is_private: bool
    ingested_at: str | None


class MarkPrivateResponse(BaseModel):
    match: MarkPrivateMatch
    match_count: int
    applied: bool
    would_invalidate_prefixes: list[str]
    invalidated_prefixes: list[str] | None = None


@router.post(
    "/admin/repos/mark-private",
    response_model=MarkPrivateResponse,
    dependencies=[Depends(require_admin_key)],
)
async def mark_private(
    body: MarkPrivateRequest,
    db: AsyncSession = Depends(get_db),
) -> MarkPrivateResponse:
    """Flip a single repo to ``is_private = true``. Dry-run first by default."""

    stmt = select(Repo).where(Repo.owner == body.owner, Repo.name == body.name)
    matches = (await db.execute(stmt)).scalars().all()

    if len(matches) == 0:
        raise HTTPException(status_code=404, detail="Repo not found")
    if len(matches) > 1:
        # `repos.name` is UNIQUE in the schema; this is a defensive guard for
        # an impossible state that, if it ever happens, must NOT mutate.
        raise HTTPException(
            status_code=409,
            detail=f"Expected exactly 1 match, found {len(matches)} — refusing to mutate",
        )

    repo = matches[0]
    match_payload = MarkPrivateMatch(
        id=str(repo.id),
        owner=repo.owner,
        name=repo.name,
        current_is_private=bool(repo.is_private),
        ingested_at=repo.ingested_at.isoformat() if repo.ingested_at else None,
    )

    if body.dry_run:
        return MarkPrivateResponse(
            match=match_payload,
            match_count=1,
            applied=False,
            would_invalidate_prefixes=list(INVALIDATION_PREFIXES),
        )

    # Apply path — mutate first, then invalidate caches, then audit-log.
    if not repo.is_private:
        repo.is_private = True
        await db.flush()

    invalidated: list[str] = []
    for prefix in INVALIDATION_PREFIXES:
        try:
            await redis_cache.clear_prefix(prefix)
            invalidated.append(prefix)
        except Exception:  # noqa: BLE001 — invalidation is best-effort
            logger.warning(
                "mark_private: clear_prefix(%s) failed — continuing", prefix,
                exc_info=True,
            )

    audit = AuditLog(
        endpoint="admin.mark_private",
        method="POST",
        request_summary=(
            f"owner={body.owner} name={body.name} dry_run=False "
            f"prior_is_private={match_payload.current_is_private}"
        ),
        response_status=200,
    )
    db.add(audit)
    await db.commit()
    await db.refresh(repo)

    return MarkPrivateResponse(
        match=MarkPrivateMatch(
            id=str(repo.id),
            owner=repo.owner,
            name=repo.name,
            current_is_private=bool(repo.is_private),
            ingested_at=repo.ingested_at.isoformat() if repo.ingested_at else None,
        ),
        match_count=1,
        applied=True,
        would_invalidate_prefixes=list(INVALIDATION_PREFIXES),
        invalidated_prefixes=invalidated,
    )
