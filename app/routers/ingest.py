import asyncio
import base64
import json
import logging
import os
from datetime import datetime, timezone
from uuid import uuid4

import anthropic as _anthropic_lib
from fastapi import APIRouter, Depends, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy import select

from app.circuit_breaker import anthropic_breaker
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import require_ingest_key, require_pubsub_push, verify_api_key
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
from app.routers.intelligence import _portfolio_insights
from app.routers.library_full import invalidate_library_cache
from app.routers.taxonomy import AssignBody, RebuildBody, assign_taxonomy, embed_taxonomy, rebuild_taxonomy
from app.schemas.repo import IngestResponse, RepoEnrichItem, RepoIngestItem
from app.schemas.trend import GapAnalysisIn, GapAnalysisOut, IngestionLogIn, IngestionLogOut, TrendSnapshotIn, TrendSnapshotOut

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/ingest",
    tags=["Ingest"],
    dependencies=[Depends(verify_api_key), Depends(require_ingest_key)],
)
events_router = APIRouter(
    prefix="/ingest/events",
    tags=["Ingest"],
    dependencies=[Depends(require_ingest_key), Depends(require_pubsub_push)],
)
from app.rate_limit import rate_limit_storage

limiter = Limiter(key_func=get_remote_address, storage_uri=rate_limit_storage)

MAX_BATCH = 100

# Mapping from ingest payload field → taxonomy dimension string
_TAXONOMY_DIMENSION_MAP = {
    "skill_areas": "skill_area",
    "industries": "industry",
    "use_cases": "use_case",
    "modalities": "modality",
    "ai_trends": "ai_trend",
    "deployment_context": "deployment_context",
    "dependencies": "dependency",
}


# ---------------------------------------------------------------------------
# KAN-157: Auto-enrichment helpers (Haiku, ~$0.0004/repo)
# ---------------------------------------------------------------------------

_AUTO_ENRICH_PROMPT = """Analyze this AI/ML GitHub repository and return a JSON object.

Repository information:
{repo_context}

{{
  "readme_summary": "2-3 sentence plain language description of what this repo does and who uses it",
  "problem_solved": "1 sentence: what specific problem does this solve",
  "quality_assessment": "high|medium|low — based on documentation quality, activity, and stars",
  "maturity_level": "research|prototype|beta|production",
  "skill_areas": ["AI/ML expertise domains — e.g. 'Retrieval-Augmented Generation', 'LoRA Fine-tuning'"],
  "industries": ["industry verticals — e.g. 'Healthcare', 'FinTech' — omit if general-purpose"],
  "use_cases": ["specific problems solved — e.g. 'Document Question Answering', 'Code Review Automation'"],
  "modalities": ["data types — e.g. 'Text', 'Code', 'Image', 'Audio'"],
  "ai_trends": ["AI paradigms — e.g. 'Agentic AI', 'Small Language Models', 'AI Safety'"],
  "deployment_context": ["where it runs — e.g. 'Cloud API', 'Self-hosted', 'Edge/Mobile'"],
  "integration_tags": ["specific frameworks/tools — lowercase — e.g. 'langchain', 'pytorch', 'fastapi'"]
}}

Rules:
- Base all values on the description/context — no speculation
- integration_tags: lowercase, specific library names only
- Return ONLY valid JSON, no markdown, no explanation"""

_ENRICH_TAXONOMY_DIMENSIONS = {
    "skill_areas":        "skill_area",
    "industries":         "industry",
    "use_cases":          "use_case",
    "modalities":         "modality",
    "ai_trends":          "ai_trend",
    "deployment_context": "deployment_context",
}


def _get_anthropic_key() -> str:
    """Retrieve Anthropic API key from env or GCP Secret Manager."""
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if key:
        return key
    try:
        from google.cloud import secretmanager
        client = secretmanager.SecretManagerServiceClient()
        project = os.getenv("GCP_PROJECT", "perditio-platform")
        name = f"projects/{project}/secrets/anthropic-api-key/versions/latest"
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8").strip()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"ANTHROPIC_API_KEY not configured: {exc}")


async def _enrich_repo_with_haiku(repo: Repo, db: AsyncSession) -> dict:
    """
    Call Claude Haiku to enrich a single repo with readme_summary, quality_signals,
    and taxonomy dimensions. Returns a dict with enrichment outcome stats.

    Uses asyncio.to_thread so the synchronous Anthropic SDK call doesn't block the loop.
    """
    parts = [
        f"Name: {repo.owner}/{repo.name}",
        f"Description: {repo.description or 'None'}",
        f"Primary Language: {repo.primary_language or 'Unknown'}",
    ]
    if repo.forked_from:
        parts.append(f"Forked from: {repo.forked_from}")
    repo_context = "\n".join(parts)
    prompt = _AUTO_ENRICH_PROMPT.format(repo_context=repo_context)

    api_key = _get_anthropic_key()

    def _call_haiku():
        client = _anthropic_lib.Anthropic(api_key=api_key)
        return client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )

    with anthropic_breaker:
        response = await asyncio.to_thread(_call_haiku)

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    data = json.loads(raw)

    # Validate quality/maturity
    quality = data.get("quality_assessment", "medium")
    if quality not in ("high", "medium", "low"):
        quality = "medium"
    maturity = data.get("maturity_level", "")
    if maturity not in ("research", "prototype", "beta", "production"):
        maturity = None

    # Write enrichment columns
    repo.readme_summary = data.get("readme_summary") or repo.readme_summary
    repo.problem_solved = data.get("problem_solved") or repo.problem_solved
    repo.integration_tags = [
        t.lower().strip() for t in data.get("integration_tags", []) if t and isinstance(t, str)
    ] or repo.integration_tags
    repo.quality_signals = {"quality": quality, "maturity": maturity}
    repo.updated_at = datetime.now(timezone.utc)

    # Write taxonomy dimensions via the shared helper
    taxonomy_dict = {k: data.get(k, []) for k in _ENRICH_TAXONOMY_DIMENSIONS}
    await _upsert_repo_taxonomy(db, repo.id, taxonomy_dict)

    await db.commit()

    return {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "quality": quality,
        "maturity": maturity,
    }


def _severity_for_repo_count(repo_count: int) -> str:
    if repo_count == 0:
        return "missing"
    if repo_count <= 2:
        return "weak"
    if repo_count <= 5:
        return "moderate"
    return "strong"


def _trend_label(trending_score: float) -> str:
    if trending_score >= 5:
        return "rising"
    if trending_score <= -1:
        return "cooling"
    return "stable"


async def _rebuild_gap_analysis(db: AsyncSession) -> dict[str, int]:
    await db.execute(GapAnalysis.__table__.delete())

    rows = (
        await db.execute(
            text(
                """
                SELECT tv.name AS skill,
                       COALESCE(tv.repo_count, 0) AS repo_count,
                       COALESCE(tv.trending_score, 0) AS trending_score
                FROM taxonomy_values tv
                WHERE tv.dimension = 'skill_area'
                ORDER BY tv.repo_count ASC, tv.name ASC
                """
            )
        )
    ).all()

    inserted = 0
    for row in rows:
        repo_count = int(row.repo_count or 0)
        gap = GapAnalysis(
            skill=row.skill,
            severity=_severity_for_repo_count(repo_count),
            repo_count=repo_count,
            why=f"{repo_count} repos currently cover {row.skill}.",
            trend=_trend_label(float(row.trending_score or 0.0)),
            essential_repos=None,
        )
        db.add(gap)
        inserted += 1

    await db.commit()
    await cache.invalidate("gaps:latest")
    return {"gap_rows": inserted}


def _parse_pubsub_payload(payload: dict) -> dict:
    message = payload.get("message")
    if not isinstance(message, dict):
        return payload

    raw_data = message.get("data")
    if not raw_data:
        return payload

    decoded = base64.b64decode(raw_data).decode("utf-8")
    try:
        parsed = json.loads(decoded)
        return parsed if isinstance(parsed, dict) else {"decoded": parsed}
    except json.JSONDecodeError:
        return {"decoded": decoded}


async def _refresh_portfolio_intelligence(db: AsyncSession) -> dict[str, int]:
    await cache.invalidate("intelligence:portfolio-insights")
    response = await _portfolio_insights(db)
    return {
        "taxonomy_gap_count": len(response.taxonomy_gaps),
        "stale_repo_count": len(response.stale_repos),
        "velocity_leader_count": len(response.velocity_leaders),
        "near_duplicate_cluster_count": len(response.near_duplicate_clusters),
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
        exclude={"tags", "categories", "builders", "ai_dev_skills", "pm_skills", "languages", "commits",
                 "skill_areas", "industries", "use_cases", "modalities", "ai_trends", "deployment_context",
                 "dependencies"}
    )

    if repo is None:
        repo = Repo(**repo_fields)
        db.add(repo)
    else:
        for key, val in repo_fields.items():
            # readme_summary and quality_signals are written by the AI enricher —
            # never overwrite them with NULL during a bulk ingest upsert.
            if key in {"readme_summary", "quality_signals", "problem_solved", "integration_tags"}:
                if val is not None:
                    setattr(repo, key, val)
            elif val is not None or key in {"description", "forked_from", "primary_language",
                                             "fork_sync_state", "github_updated_at",
                                             "github_created_at"}:
                setattr(repo, key, val)
        repo.updated_at = datetime.now(timezone.utc)

    await db.flush()

    # Replace child rows — skip-empty guard: only replace if incoming array is non-empty
    # to prevent accidental data loss when payload omits or sends empty lists.
    if item.tags:
        await db.execute(RepoTag.__table__.delete().where(RepoTag.repo_id == repo.id))
        for tag in item.tags:
            db.add(RepoTag(repo_id=repo.id, tag=tag))
    if item.categories:
        await db.execute(RepoCategory.__table__.delete().where(RepoCategory.repo_id == repo.id))
        for cat in item.categories:
            db.add(RepoCategory(repo_id=repo.id, **cat.model_dump()))
    if item.builders:
        await db.execute(RepoBuilder.__table__.delete().where(RepoBuilder.repo_id == repo.id))
        for builder in item.builders:
            db.add(RepoBuilder(repo_id=repo.id, **builder.model_dump()))
    if item.ai_dev_skills:
        await db.execute(RepoAIDevSkill.__table__.delete().where(RepoAIDevSkill.repo_id == repo.id))
        for skill in item.ai_dev_skills:
            db.add(RepoAIDevSkill(repo_id=repo.id, skill=skill))
    if item.pm_skills:
        await db.execute(RepoPMSkill.__table__.delete().where(RepoPMSkill.repo_id == repo.id))
        for skill in item.pm_skills:
            db.add(RepoPMSkill(repo_id=repo.id, skill=skill))
    if item.languages:
        await db.execute(RepoLanguage.__table__.delete().where(RepoLanguage.repo_id == repo.id))
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
    """Upsert a bounded batch of repos from ingestion. Requires API and ingest keys."""
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
    """Apply enrichment-only fields to an existing repo. Requires API and ingest keys."""
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
    """Persist trend snapshots produced by ingestion. Requires API and ingest keys."""
    rows = []
    for item in items:
        snap = TrendSnapshot(**item.model_dump())
        db.add(snap)
        rows.append(snap)

    await db.commit()
    for row in rows:
        await db.refresh(row)

    await cache.invalidate("trends:latest")
    await cache.invalidate("trends:report")
    return [TrendSnapshotOut.model_validate(r, from_attributes=True) for r in rows]


@router.post("/gaps", response_model=list[GapAnalysisOut])
async def ingest_gaps(
    items: list[GapAnalysisIn],
    db: AsyncSession = Depends(get_db),
) -> list[GapAnalysisOut]:
    """Persist gap-analysis rows produced by ingestion. Requires API and ingest keys."""
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
    """Persist or update an ingestion run log record. Requires API and ingest keys."""
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


@events_router.post("/repo-ingested", response_model=dict)
async def repo_ingested_event(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Handle Pub/Sub repo-ingested pushes and refresh taxonomy, gaps, and insights."""
    payload = _parse_pubsub_payload(await request.json())

    repo_name = payload.get("repo_name") or payload.get("name") or "unknown"
    logger.info(f"Repo ingested: {repo_name}, enrichment pending")

    log = logging.getLogger(__name__)

    rebuild_result = await rebuild_taxonomy(RebuildBody(), db)
    try:
        embed_result = await embed_taxonomy(db)
    except Exception as exc:
        log.exception("embed_taxonomy failed during repo-ingested refresh")
        embed_result = {"status": "skipped", "error": str(exc), "embedded": 0}
    assign_result = await assign_taxonomy(AssignBody(), db)
    gap_result = await _rebuild_gap_analysis(db)

    await cache.invalidate("library:full*")
    await cache.invalidate("repos:list:*")
    await cache.invalidate("stats:overview")
    invalidate_library_cache()

    insights_result = await _refresh_portfolio_intelligence(db)

    return {
        "status": "ok",
        "received": payload,
        "taxonomy_rebuild": rebuild_result,
        "taxonomy_embed": embed_result,
        "taxonomy_assign": assign_result,
        "gap_rebuild": gap_result,
        "portfolio_insights": insights_result,
    }


# ---------------------------------------------------------------------------
# KAN-157: POST /ingest/events/repo-added
# Triggered by Pub/Sub REPO_ADDED events when a new repo enters the platform.
# ---------------------------------------------------------------------------

@events_router.post("/repo-added", response_model=dict)
async def repo_added_event(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Handle Pub/Sub REPO_ADDED push notifications.

    When forksync or the ingestion pipeline adds a new repo, it publishes a
    ``repo.added`` event with ``name_with_owner`` in the payload.  This handler:

    1. Resolves the repo by name from the DB.
    2. Skips if already enriched (idempotent).
    3. Calls Claude Haiku to generate readme_summary, quality_signals, and
       taxonomy dimensions — ~$0.0004 per repo.
    4. Invalidates relevant caches.

    Always returns HTTP 200 so Pub/Sub does not retry on transient enrichment
    failures (errors are logged instead).
    """
    payload = _parse_pubsub_payload(await request.json())

    # REPO_ADDED schema: {"name_with_owner": "owner/name", "stars": N, "language": "..."}
    name_with_owner = (
        payload.get("name_with_owner")
        or payload.get("repo_name")
        or payload.get("name")
        or ""
    )
    # Extract bare repo name (the "name" column in repos table uses owner/name or just name)
    # The repos table uses bare name as the unique key; strip owner prefix if present.
    repo_name = name_with_owner.split("/")[-1] if name_with_owner else ""
    if not repo_name:
        logger.warning("repo-added event: no repo name in payload — skipping")
        return {"status": "skipped", "reason": "no repo name in payload"}

    result = await db.execute(select(Repo).where(Repo.name == repo_name))
    repo = result.scalar_one_or_none()

    if repo is None:
        logger.warning("repo-added event: repo '%s' not found in DB — may not be ingested yet", repo_name)
        return {"status": "skipped", "reason": f"repo '{repo_name}' not found in DB"}

    if repo.quality_signals is not None:
        logger.info("repo-added event: '%s' already enriched — skipping", repo_name)
        return {"status": "skipped", "reason": "already enriched", "repo": repo_name}

    logger.info("repo-added event: enriching '%s' with Haiku", repo_name)
    try:
        enrich_result = await _enrich_repo_with_haiku(repo, db)
    except json.JSONDecodeError as exc:
        logger.error("repo-added event: Haiku JSON parse error for '%s': %s", repo_name, exc)
        return {"status": "error", "repo": repo_name, "reason": "json_parse_error"}
    except HTTPException as exc:
        logger.error("repo-added event: HTTP error for '%s': %s", repo_name, exc.detail)
        return {"status": "error", "repo": repo_name, "reason": exc.detail}
    except Exception as exc:
        logger.error("repo-added event: unexpected error for '%s': %s", repo_name, exc)
        return {"status": "error", "repo": repo_name, "reason": str(exc)}

    # Invalidate caches so the enriched data appears immediately
    await cache.invalidate(f"repos:detail:{repo_name}")
    await cache.invalidate("library:full*")
    await cache.invalidate(f"similar:{repo_name}:*")
    invalidate_library_cache()

    logger.info(
        "repo-added event: '%s' enriched — quality=%s maturity=%s tokens=%d+%d",
        repo_name,
        enrich_result.get("quality"),
        enrich_result.get("maturity"),
        enrich_result.get("input_tokens", 0),
        enrich_result.get("output_tokens", 0),
    )
    return {
        "status": "ok",
        "repo": repo_name,
        "enrichment": enrich_result,
    }
