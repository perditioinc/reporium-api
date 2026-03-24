import logging
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import require_ingest_key, verify_api_key
from app.cache import cache
from app.database import get_db
from app.models.repo import (
    Repo,
    RepoAIDevSkill,
    RepoBuilder,
    RepoCategory,
    RepoCommit,
    RepoLanguage,
    RepoPMSkill,
    RepoTag,
    RepoTaxonomy,
)
from app.models.trend import GapAnalysis, IngestionLog, TrendSnapshot
from app.routers.library_full import invalidate_library_cache
from app.schemas.repo import IngestResponse, RepoEnrichItem, RepoIngestItem
from app.schemas.trend import GapAnalysisIn, GapAnalysisOut, IngestionLogIn, IngestionLogOut, TrendSnapshotIn, TrendSnapshotOut

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", dependencies=[Depends(verify_api_key), Depends(require_ingest_key)])
limiter = Limiter(key_func=get_remote_address)

MAX_BATCH = 100

# Mapping from ingest payload field → taxonomy dimension string
_TAXONOMY_DIMENSION_MAP = {
    "skill_areas": "skill_area",
    "industries": "industry",
    "use_cases": "use_case",
    "modalities": "modality",
    "ai_trends": "ai_trend",
    "deployment_context": "deployment_context",
}


async def _upsert_repo_taxonomy(db: AsyncSession, repo_id, item_dict: dict) -> None:
    """Write taxonomy dimension values to repo_taxonomy using INSERT ... ON CONFLICT DO NOTHING."""
    from sqlalchemy import text as _text
    for field, dimension in _TAXONOMY_DIMENSION_MAP.items():
        values = item_dict.get(field) or []
        for raw_value in values:
            if not raw_value:
                continue
            await db.execute(
                _text(
                    "INSERT INTO repo_taxonomy (repo_id, dimension, raw_value, assigned_by) "
                    "VALUES (:repo_id, :dimension, :raw_value, 'enrichment') "
                    "ON CONFLICT (repo_id, dimension, raw_value) DO NOTHING"
                ),
                {"repo_id": str(repo_id), "dimension": dimension, "raw_value": raw_value},
            )


async def _upsert_repo(db: AsyncSession, item: RepoIngestItem) -> Repo:
    stmt = select(Repo).where(Repo.name == item.name)
    result = await db.execute(stmt)
    repo = result.scalar_one_or_none()

    repo_fields = item.model_dump(
        exclude={"tags", "categories", "builders", "ai_dev_skills", "pm_skills", "languages", "commits"}
    )

    if repo is None:
        repo = Repo(**repo_fields)
        db.add(repo)
    else:
        for key, val in repo_fields.items():
            if val is not None or key in {"description", "forked_from", "primary_language",
                                           "fork_sync_state", "readme_summary", "github_updated_at",
                                           "github_created_at"}:
                setattr(repo, key, val)
        repo.updated_at = datetime.now(timezone.utc)

    await db.flush()

    # Replace child rows
    await db.execute(RepoTag.__table__.delete().where(RepoTag.repo_id == repo.id))
    await db.execute(RepoCategory.__table__.delete().where(RepoCategory.repo_id == repo.id))
    await db.execute(RepoBuilder.__table__.delete().where(RepoBuilder.repo_id == repo.id))
    await db.execute(RepoAIDevSkill.__table__.delete().where(RepoAIDevSkill.repo_id == repo.id))
    await db.execute(RepoPMSkill.__table__.delete().where(RepoPMSkill.repo_id == repo.id))
    await db.execute(RepoLanguage.__table__.delete().where(RepoLanguage.repo_id == repo.id))

    for tag in item.tags:
        db.add(RepoTag(repo_id=repo.id, tag=tag))
    for cat in item.categories:
        db.add(RepoCategory(repo_id=repo.id, **cat.model_dump()))
    for builder in item.builders:
        db.add(RepoBuilder(repo_id=repo.id, **builder.model_dump()))
    for skill in item.ai_dev_skills:
        db.add(RepoAIDevSkill(repo_id=repo.id, skill=skill))
    for skill in item.pm_skills:
        db.add(RepoPMSkill(repo_id=repo.id, skill=skill))
    for lang in item.languages:
        db.add(RepoLanguage(repo_id=repo.id, **lang.model_dump()))

    if item.commits:
        await db.execute(RepoCommit.__table__.delete().where(RepoCommit.repo_id == repo.id))
        for commit in item.commits:
            db.add(RepoCommit(repo_id=repo.id, **commit.model_dump()))

    # Write dynamic taxonomy dimensions (ON CONFLICT DO NOTHING — safe to re-run)
    await _upsert_repo_taxonomy(db, repo.id, item.model_dump())

    return repo


@router.post("/repos", response_model=IngestResponse)
@limiter.limit("200/minute")
async def ingest_repos(
    request: Request,
    items: list[RepoIngestItem],
    db: AsyncSession = Depends(get_db),
) -> IngestResponse:
    if len(items) > MAX_BATCH:
        raise HTTPException(status_code=400, detail=f"Max {MAX_BATCH} repos per request")

    upserted = 0
    errors: list[str] = []

    for item in items:
        try:
            await _upsert_repo(db, item)
            upserted += 1
        except Exception as e:
            logger.error(f"Failed to upsert repo '{item.name}': {e}")
            errors.append(f"{item.name}: {str(e)}")

    await db.commit()

    # Invalidate both Redis cache keys and the in-memory /library/full cache
    await cache.invalidate("library:full*")
    await cache.invalidate("repos:list:*")
    await cache.invalidate("stats:overview")
    invalidate_library_cache()

    return IngestResponse(upserted=upserted, errors=errors)


@router.post("/repos/{name}/enrich", response_model=dict)
async def enrich_repo(
    name: str,
    item: RepoEnrichItem,
    db: AsyncSession = Depends(get_db),
) -> dict:
    stmt = select(Repo).where(Repo.name == name)
    result = await db.execute(stmt)
    repo = result.scalar_one_or_none()
    if not repo:
        raise HTTPException(status_code=404, detail=f"Repo '{name}' not found")

    if item.readme_summary is not None:
        repo.readme_summary = item.readme_summary
    if item.activity_score is not None:
        repo.activity_score = item.activity_score
    repo.updated_at = datetime.now(timezone.utc)

    if item.tags is not None:
        await db.execute(RepoTag.__table__.delete().where(RepoTag.repo_id == repo.id))
        for tag in item.tags:
            db.add(RepoTag(repo_id=repo.id, tag=tag))

    if item.ai_dev_skills is not None:
        await db.execute(RepoAIDevSkill.__table__.delete().where(RepoAIDevSkill.repo_id == repo.id))
        for skill in item.ai_dev_skills:
            db.add(RepoAIDevSkill(repo_id=repo.id, skill=skill))

    if item.pm_skills is not None:
        await db.execute(RepoPMSkill.__table__.delete().where(RepoPMSkill.repo_id == repo.id))
        for skill in item.pm_skills:
            db.add(RepoPMSkill(repo_id=repo.id, skill=skill))

    # Write any new taxonomy dimension values (ON CONFLICT DO NOTHING)
    await _upsert_repo_taxonomy(db, repo.id, item.model_dump())

    await db.commit()

    await cache.invalidate(f"repos:detail:{name}")
    await cache.invalidate("library:full*")

    return {"status": "ok", "name": name}


@router.post("/trends/snapshot", response_model=list[TrendSnapshotOut])
async def ingest_trend_snapshot(
    items: list[TrendSnapshotIn],
    db: AsyncSession = Depends(get_db),
) -> list[TrendSnapshotOut]:
    rows = []
    for item in items:
        snap = TrendSnapshot(**item.model_dump())
        db.add(snap)
        rows.append(snap)

    await db.commit()
    for row in rows:
        await db.refresh(row)

    await cache.invalidate("trends:latest")
    return [TrendSnapshotOut.model_validate(r, from_attributes=True) for r in rows]


@router.post("/gaps", response_model=list[GapAnalysisOut])
async def ingest_gaps(
    items: list[GapAnalysisIn],
    db: AsyncSession = Depends(get_db),
) -> list[GapAnalysisOut]:
    rows = []
    for item in items:
        gap = GapAnalysis(**item.model_dump())
        db.add(gap)
        rows.append(gap)

    await db.commit()
    for row in rows:
        await db.refresh(row)

    await cache.invalidate("gaps:latest")
    return [GapAnalysisOut.model_validate(r, from_attributes=True) for r in rows]


@router.post("/log", response_model=IngestionLogOut)
async def ingest_log(
    item: IngestionLogIn,
    db: AsyncSession = Depends(get_db),
) -> IngestionLogOut:
    # Find the latest running log for this mode, or create a new one
    stmt = (
        select(IngestionLog)
        .where(IngestionLog.mode == item.mode, IngestionLog.status == "running")
        .order_by(IngestionLog.started_at.desc())
        .limit(1)
    )
    result = await db.execute(stmt)
    log = result.scalar_one_or_none()

    if log is None:
        log = IngestionLog(mode=item.mode, status=item.status or "running")
        db.add(log)
    else:
        if item.repos_fetched is not None:
            log.repos_fetched = item.repos_fetched
        if item.repos_updated is not None:
            log.repos_updated = item.repos_updated
        if item.api_calls_made is not None:
            log.api_calls_made = item.api_calls_made
        if item.errors is not None:
            log.errors = item.errors
        if item.status is not None:
            log.status = item.status
        if item.completed_at is not None:
            log.completed_at = item.completed_at

    await db.commit()
    await db.refresh(log)
    return IngestionLogOut.model_validate(log, from_attributes=True)
