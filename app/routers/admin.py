import json
import logging

from fastapi import APIRouter, Depends, Query
from sqlalchemy import delete, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import require_admin_key, verify_api_key
from app.cache import cache
from app.database import get_db
from app.models.repo import IngestRun, Repo, RepoEmbedding, RepoTag
from app.routers.library_full import invalidate_library_cache

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Admin"])

# Canonical noise-tag list from Reporium taxonomy Phase 4 cleanup rules.
NOISE_TAGS = frozenset({
    "python", "javascript", "typescript", "rust", "go", "java", "c++",
    "react", "nextjs", "nodejs", "express", "fastapi", "flask", "django",
    "postgresql", "mysql", "mongodb", "redis", "docker", "kubernetes",
    "aws", "gcp", "azure", "terraform", "nginx", "linux", "macos", "windows",
    "git", "github", "api", "rest", "graphql", "grpc", "websocket", "cli",
    "sdk", "library", "framework", "tutorial", "example", "demo", "template",
    "boilerplate", "starter", "awesome", "list", "collection", "open-source",
    "free", "fast", "simple", "easy", "lightweight", "minimal",
})


async def _prune_noise_tags(db: AsyncSession, *, dry_run: bool) -> dict:
    tag_counts_result = await db.execute(
        select(
            func.lower(RepoTag.tag).label("tag"),
            func.count().label("count"),
        )
        .where(func.lower(RepoTag.tag).in_(NOISE_TAGS))
        .group_by(func.lower(RepoTag.tag))
        .order_by(func.count().desc(), func.lower(RepoTag.tag))
    )
    matched_tags = {row.tag: row.count for row in tag_counts_result.fetchall()}
    matched_rows = sum(matched_tags.values())

    deleted_rows = 0
    if not dry_run and matched_rows:
        delete_result = await db.execute(
            delete(RepoTag).where(func.lower(RepoTag.tag).in_(NOISE_TAGS))
        )
        deleted_rows = delete_result.rowcount or 0
        await db.commit()

        await cache.invalidate("library:full*")
        await cache.invalidate("repos:list:*")
        invalidate_library_cache()

    return {
        "dry_run": dry_run,
        "matched_rows": matched_rows,
        "matched_tag_count": len(matched_tags),
        "deleted_rows": 0 if dry_run else deleted_rows,
        "matched_tags": matched_tags,
    }


@router.get("/admin/data-quality", response_model=dict)
async def data_quality(
    db: AsyncSession = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
    _admin_key: None = Depends(require_admin_key),
):
    """Return aggregate admin-only data quality metrics for the current repo corpus."""

    # Query counts
    total = (await db.execute(text("SELECT COUNT(*) FROM repos;"))).scalar()
    owned = (await db.execute(text("SELECT COUNT(*) FROM repos WHERE is_fork = false;"))).scalar()
    forks = (await db.execute(text("SELECT COUNT(*) FROM repos WHERE is_fork = true;"))).scalar()
    missing_summary = (await db.execute(text("SELECT COUNT(*) FROM repos WHERE readme_summary IS NULL OR readme_summary = '';"))).scalar()
    missing_desc = (await db.execute(text("SELECT COUNT(*) FROM repos WHERE description IS NULL OR description = '';"))).scalar()
    missing_cats = (await db.execute(text("SELECT COUNT(*) FROM repos r WHERE r.id NOT IN (SELECT DISTINCT repo_id FROM repo_categories);"))).scalar()
    missing_builders = (await db.execute(text("SELECT COUNT(*) FROM repos r WHERE r.id NOT IN (SELECT DISTINCT repo_id FROM repo_builders);"))).scalar()
    missing_embeddings = (await db.execute(text("SELECT COUNT(*) FROM repos r WHERE r.id NOT IN (SELECT DISTINCT repo_id FROM repo_embeddings);"))).scalar()

    # Category distribution
    cat_result = await db.execute(text("""
        SELECT category_name, COUNT(DISTINCT repo_id) as cnt
        FROM repo_categories WHERE is_primary = true
        GROUP BY category_name ORDER BY cnt DESC;
    """))
    cat_dist = {r[0]: r[1] for r in cat_result.fetchall()}
    max_cat_pct = (max(cat_dist.values()) / total * 100) if cat_dist and total > 0 else 0

    # Quality score: 100 minus penalties
    score = 100
    if missing_summary > 0:
        score -= min(20, missing_summary)
    if missing_desc > 0:
        score -= min(10, missing_desc)
    if missing_cats > 0:
        score -= min(10, missing_cats)
    if missing_builders > 0:
        score -= min(10, missing_builders)
    if missing_embeddings > 0:
        score -= min(15, missing_embeddings)
    if max_cat_pct > 25:
        score -= min(10, int(max_cat_pct - 25))

    return {
        "total_repos": total,
        "owned_repos": owned,
        "fork_repos": forks,
        "missing_summary": missing_summary,
        "missing_description": missing_desc,
        "missing_categories": missing_cats,
        "missing_builders": missing_builders,
        "missing_embeddings": missing_embeddings,
        "category_distribution": cat_dist,
        "max_category_percent": round(max_cat_pct, 1),
        "quality_score": max(0, score),
    }


@router.post("/admin/tags/prune", response_model=dict)
async def prune_tags(
    dry_run: bool = Query(default=False),
    db: AsyncSession = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
    _admin_key: None = Depends(require_admin_key),
):
    """Count or delete noise tags from repo_tags. Requires API and admin keys."""
    return await _prune_noise_tags(db, dry_run=dry_run)


@router.post("/admin/quality/compute", response_model=dict)
async def compute_quality_signals(
    db: AsyncSession = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
    _admin_key: None = Depends(require_admin_key),
):
    """Compute quality_signals for all repos from existing data (no GitHub API calls)."""
    BATCH_SIZE = 100
    offset = 0
    computed = 0
    skipped = 0

    while True:
        stmt = select(Repo).offset(offset).limit(BATCH_SIZE)
        result = await db.execute(stmt)
        repos = result.scalars().all()

        if not repos:
            break

        for repo in repos:
            try:
                commit_velocity_30d = repo.commits_last_30_days / 30.0
                commit_velocity_7d = repo.commits_last_7_days / 7.0
                is_active = repo.commits_last_30_days > 0
                has_open_issues = repo.open_issues_count > 0

                activity = repo.activity_score  # 0-100

                weekly_score = (min(repo.commits_last_7_days, 10) / 10.0) * 100
                if repo.open_issues_count < 10:
                    issues_score = 100
                else:
                    issues_score = max(0, 100 - repo.open_issues_count * 2)

                overall_raw = (
                    activity * 0.5
                    + weekly_score * 0.3
                    + issues_score * 0.2
                )
                overall_score = max(0, min(100, round(overall_raw)))

                repo.quality_signals = {
                    "commit_velocity_30d": commit_velocity_30d,
                    "commit_velocity_7d": commit_velocity_7d,
                    "is_active": is_active,
                    "has_open_issues": has_open_issues,
                    "activity_score": activity,
                    "overall_score": overall_score,
                }
                computed += 1
            except Exception:
                skipped += 1

        await db.commit()
        offset += BATCH_SIZE

    return {"computed": computed, "skipped": skipped}


@router.post("/admin/embeddings/backfill", response_model=dict)
async def backfill_embeddings(
    db: AsyncSession = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
    _admin_key: None = Depends(require_admin_key),
):
    """
    Generate embeddings for all repos that have no row in repo_embeddings.
    Uses the sentence-transformers model (all-MiniLM-L6-v2) from app.embeddings.
    Returns the count of embeddings inserted and any per-repo errors.
    """
    from app.embeddings import get_embedding_model

    # Find repos with no embedding (LEFT JOIN → NULL on repo_embeddings side)
    result = await db.execute(text(
        """
        SELECT r.id, r.name, r.description, r.readme_summary,
               r.primary_language
        FROM repos r
        LEFT JOIN repo_embeddings e ON e.repo_id = r.id
        WHERE e.repo_id IS NULL
        ORDER BY r.updated_at DESC
        """
    ))
    rows = result.fetchall()

    if not rows:
        return {"backfilled": 0, "errors": []}

    # Fetch tags for these repos so we can build embed text
    repo_ids = [str(row.id) for row in rows]
    tags_result = await db.execute(text(
        "SELECT repo_id::text, tag FROM repo_tags WHERE repo_id::text = ANY(:ids)"
    ), {"ids": repo_ids})
    tags_by_repo: dict[str, list[str]] = {}
    for t_row in tags_result.fetchall():
        tags_by_repo.setdefault(t_row.repo_id, []).append(t_row.tag)

    # Build embed texts
    embed_texts = []
    for row in rows:
        parts = [row.name or ""]
        if row.description:
            parts.append(row.description)
        if row.readme_summary:
            parts.append(row.readme_summary)
        tags = tags_by_repo.get(str(row.id), [])
        if tags:
            parts.append("tags: " + ", ".join(tags))
        if row.primary_language:
            parts.append("language: " + row.primary_language)
        embed_texts.append(" | ".join(parts))

    # Generate embeddings in batch
    try:
        model = get_embedding_model()
        embeddings = model.encode(embed_texts, normalize_embeddings=True)
    except Exception as exc:
        return {"backfilled": 0, "errors": [f"Model error: {exc}"]}

    backfilled = 0
    errors: list[str] = []
    for row, emb in zip(rows, embeddings):
        try:
            vec_json = json.dumps([float(v) for v in emb])
            await db.execute(text(
                """
                INSERT INTO repo_embeddings (repo_id, embedding, model, generated_at)
                VALUES (:repo_id, :embedding, 'nomic-embed-text', NOW())
                ON CONFLICT (repo_id) DO NOTHING
                """
            ), {"repo_id": str(row.id), "embedding": vec_json})
            backfilled += 1
        except Exception as exc:
            errors.append(f"{row.name}: {exc}")
            logger.warning("Embedding insert failed for %s: %s", row.name, exc)

    await db.commit()
    return {"backfilled": backfilled, "errors": errors}


@router.post("/admin/taxonomy/bootstrap", response_model=dict)
async def bootstrap_taxonomy(
    limit: int = Query(default=100, ge=1, le=500),
    dimension: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
    _admin_key: None = Depends(require_admin_key),
):
    """
    For each repo with no taxonomy entries (optionally filtered to a single dimension),
    run the pgvector similarity-assign pipeline to populate repo_taxonomy.

    Steps:
    1. Find repos with no taxonomy entries for the given dimension (or any dimension).
    2. For each such repo, run assign_taxonomy scoped to that repo's embedding.
    3. Return {processed, assigned, errors}.
    """
    # Find repos that have no taxonomy for the target dimension
    if dimension:
        untagged_result = await db.execute(text(
            """
            SELECT r.id, r.name
            FROM repos r
            WHERE r.id NOT IN (
                SELECT DISTINCT repo_id FROM repo_taxonomy WHERE dimension = :dim
            )
            ORDER BY r.updated_at DESC
            LIMIT :lim
            """
        ), {"dim": dimension, "lim": limit})
    else:
        untagged_result = await db.execute(text(
            """
            SELECT r.id, r.name
            FROM repos r
            WHERE r.id NOT IN (
                SELECT DISTINCT repo_id FROM repo_taxonomy
            )
            ORDER BY r.updated_at DESC
            LIMIT :lim
            """
        ), {"lim": limit})

    untagged_repos = untagged_result.fetchall()
    processed = 0
    assigned = 0
    errors: list[str] = []

    if not untagged_repos:
        return {"processed": 0, "assigned": 0, "errors": []}

    # Fetch taxonomy values that have embeddings (optionally filtered by dimension)
    if dimension:
        tv_result = await db.execute(text(
            "SELECT id, dimension, name FROM taxonomy_values "
            "WHERE embedding_vec IS NOT NULL AND dimension = :dim"
        ), {"dim": dimension})
    else:
        tv_result = await db.execute(text(
            "SELECT id, dimension, name FROM taxonomy_values WHERE embedding_vec IS NOT NULL"
        ))
    taxonomy_values = tv_result.fetchall()

    if not taxonomy_values:
        return {"processed": len(untagged_repos), "assigned": 0, "errors": ["No taxonomy_values with embeddings found"]}

    threshold = 0.65
    repo_ids = [str(row.id) for row in untagged_repos]

    for tv in taxonomy_values:
        try:
            repo_result = await db.execute(text(
                """
                SELECT re.repo_id,
                       1 - (re.embedding_vec <=> tv.embedding_vec) AS similarity
                FROM repo_embeddings re
                JOIN taxonomy_values tv ON tv.id = :tv_id
                WHERE re.embedding_vec IS NOT NULL
                  AND re.repo_id::text = ANY(:repo_ids)
                  AND 1 - (re.embedding_vec <=> tv.embedding_vec) >= :threshold
                """
            ), {"tv_id": tv.id, "threshold": threshold, "repo_ids": repo_ids})
            repo_rows = repo_result.fetchall()

            for rrow in repo_rows:
                await db.execute(text(
                    "INSERT INTO repo_taxonomy "
                    "  (repo_id, dimension, raw_value, taxonomy_value_id, similarity_score, assigned_by) "
                    "VALUES (:repo_id, :dimension, :raw_value, :tv_id, :sim, 'similarity') "
                    "ON CONFLICT (repo_id, dimension, raw_value) DO NOTHING"
                ), {
                    "repo_id": str(rrow.repo_id),
                    "dimension": tv.dimension,
                    "raw_value": tv.name,
                    "tv_id": tv.id,
                    "sim": float(rrow.similarity),
                })
                assigned += 1
        except Exception as exc:
            errors.append(f"tv {tv.name}: {exc}")
            logger.warning("Taxonomy bootstrap failed for value %s: %s", tv.name, exc)

    processed = len(untagged_repos)
    await db.commit()
    return {"processed": processed, "assigned": assigned, "errors": errors}


# ---------------------------------------------------------------------------
# Data integrity health check
# ---------------------------------------------------------------------------

@router.get("/admin/health/data", response_model=dict)
async def data_integrity_health(
    db: AsyncSession = Depends(get_db),
    _admin_key: None = Depends(require_admin_key),
):
    """
    Monitor junction table row counts and coverage ratios to detect data
    regressions immediately after ingestion runs.

    Status thresholds:
    - ``critical``  — repo_tags has < 100 rows total
    - ``degraded``  — repo_tags coverage < 50 % of repos
    - ``healthy``   — all checks pass
    """
    # --- raw counts (fast COUNT queries, no JOINs) ---
    tables = [
        "repos",
        "repo_tags",
        "repo_categories",
        "repo_taxonomy",
        "taxonomy_values",
        "repo_ai_dev_skills",
        "repo_pm_skills",
        "repo_languages",
    ]
    counts: dict[str, int] = {}
    for table in tables:
        row = await db.execute(text(f"SELECT COUNT(*) FROM {table}"))  # noqa: S608
        counts[table] = row.scalar() or 0

    total_repos = counts["repos"]

    # --- coverage: repos that have at least 1 row in each junction table ---
    def _pct(n: int) -> float:
        if total_repos == 0:
            return 0.0
        return round(n / total_repos * 100, 1)

    tags_covered = (
        await db.execute(
            text("SELECT COUNT(DISTINCT repo_id) FROM repo_tags")
        )
    ).scalar() or 0

    cats_covered = (
        await db.execute(
            text("SELECT COUNT(DISTINCT repo_id) FROM repo_categories")
        )
    ).scalar() or 0

    langs_covered = (
        await db.execute(
            text("SELECT COUNT(DISTINCT repo_id) FROM repo_languages")
        )
    ).scalar() or 0

    coverage = {
        "tags_pct": _pct(tags_covered),
        "categories_pct": _pct(cats_covered),
        "languages_pct": _pct(langs_covered),
    }

    # --- alerts & status ---
    alerts: list[str] = []
    status = "healthy"

    tag_total = counts["repo_tags"]
    tags_pct = coverage["tags_pct"]

    if tag_total < 100:
        status = "critical"
        alerts.append(
            f"repo_tags critically low: {tag_total} rows for {total_repos} repos"
        )
    elif tags_pct < 50.0:
        status = "degraded"
        alerts.append(
            f"repo_tags coverage degraded: {tags_pct}% of repos have a tag"
        )

    thresholds = {
        "repo_tags_min_rows": 100,
        "tags_coverage_min_pct": 50.0,
    }

    return {
        "status": status,
        "counts": counts,
        "coverage": coverage,
        "thresholds": thresholds,
        "alerts": alerts,
    }


# ---------------------------------------------------------------------------
# Run history
# ---------------------------------------------------------------------------

@router.get(
    "/admin/runs",
    summary="List recent ingestion runs",
    dependencies=[Depends(require_admin_key)],
)
async def list_runs(
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    """Return the most recent *limit* ingestion run records, newest first."""
    result = await db.execute(
        select(IngestRun).order_by(IngestRun.started_at.desc()).limit(limit)
    )
    runs = result.scalars().all()
    return [
        {
            "id": r.id,
            "run_mode": r.run_mode,
            "status": r.status,
            "repos_upserted": r.repos_upserted,
            "repos_processed": r.repos_processed,
            "errors": r.errors or [],
            "started_at": r.started_at.isoformat() if r.started_at else None,
            "finished_at": r.finished_at.isoformat() if r.finished_at else None,
            "duration_seconds": (
                (r.finished_at - r.started_at).total_seconds()
                if r.finished_at and r.started_at
                else None
            ),
        }
        for r in runs
    ]


@router.post("/admin/enrichment/trigger", response_model=dict)
async def trigger_enrichment(
    db: AsyncSession = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
    _admin_key: None = Depends(require_admin_key),
):
    """Mark unenriched repos and return count.
    A cron/external process picks these up."""
    result = await db.execute(text(
        "SELECT COUNT(*) FROM repos WHERE quality_signals IS NULL"
    ))
    pending = result.scalar()
    return {"pending_enrichment": pending, "message": f"{pending} repos need enrichment"}


@router.post(
    "/admin/runs",
    summary="Record a completed ingestion run",
    dependencies=[Depends(require_admin_key)],
    status_code=201,
)
async def record_run(
    payload: dict,
    db: AsyncSession = Depends(get_db),
):
    """
    Called by the ingestion pipeline after each run to persist run metadata.

    Expected payload::

        {
            "run_mode": "quick",
            "status": "success",
            "repos_upserted": 42,
            "repos_processed": 150,
            "errors": [],
            "started_at": "2026-03-24T05:00:00Z",
            "finished_at": "2026-03-24T05:03:12Z"
        }
    """
    from datetime import datetime, timezone

    def _parse_dt(val):
        if not val:
            return None
        if isinstance(val, datetime):
            return val
        return datetime.fromisoformat(str(val).replace("Z", "+00:00"))

    run = IngestRun(
        run_mode=payload.get("run_mode", "unknown"),
        status=payload.get("status", "unknown"),
        repos_upserted=int(payload.get("repos_upserted", 0)),
        repos_processed=int(payload.get("repos_processed", 0)),
        errors=payload.get("errors") or None,
        started_at=_parse_dt(payload.get("started_at")),
        finished_at=_parse_dt(payload.get("finished_at")),
    )
    db.add(run)
    await db.commit()
    await db.refresh(run)
    return {"id": run.id, "status": "recorded"}
