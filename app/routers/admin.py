import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import delete, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import require_admin_key, verify_api_key
from app.cache import cache
from app.database import get_db
from app.models.repo import IngestRun, Repo, RepoCategory, RepoEmbedding, RepoTag
from app.routers.library_full import invalidate_library_cache

# ── 21-category taxonomy (mirrors ingestion/enrichment/taxonomy.py) ──────────
# Kept in-process so the API can re-derive categories without calling the
# ingestion service.  Tags are case-insensitive prefix/substring matched.
_CATEGORIES: list[dict] = [
    {"id": "foundation-models", "name": "Foundation Models",
     "tags": ["large language model", "transformer", "openai", "anthropic", "claude",
               "google ai", "huggingface", "long context", "multimodal", "quantization",
               "llama", "gguf", "gpt", "llm", "foundational model"]},
    {"id": "ai-agents", "name": "AI Agents",
     "tags": ["ai agent", "multi-agent", "autonomous", "agent memory", "planning",
               "chain-of-thought", "tool use", "langchain", "langgraph", "crewai",
               "autogen", "mcp", "prompt engineering", "context engineering",
               "structured output", "function calling", "agentic"]},
    {"id": "rag-retrieval", "name": "RAG & Retrieval",
     "tags": ["rag", "vector database", "embedding", "knowledge graph",
               "semantic search", "hybrid search", "reranking", "llamaindex",
               "document processing", "chunking", "retrieval"]},
    {"id": "model-training", "name": "Model Training",
     "tags": ["fine-tuning", "reinforcement learning", "lora", "peft", "rlhf",
               "synthetic data", "dataset", "training", "unsloth", "axolotl",
               "trl", "deepspeed", "fsdp", "pytorch", "tensorflow", "keras", "jax"]},
    {"id": "evals-benchmarking", "name": "Evals & Benchmarking",
     "tags": ["eval", "benchmark", "model evaluation", "llm testing", "red teaming",
               "safety evaluation", "mmlu", "humaneval", "code evaluation", "alignment"]},
    {"id": "observability", "name": "Observability & Monitoring",
     "tags": ["observability", "tracing", "monitoring", "llm monitoring", "logging",
               "debugging", "langsmith", "phoenix", "mlflow", "weights & biases",
               "experiment tracking"]},
    {"id": "inference-serving", "name": "Inference & Serving",
     "tags": ["inference", "llm serving", "model optimization", "vllm", "tensorrt",
               "triton", "ollama", "tgi", "batching", "caching", "gpu", "cuda",
               "real-time", "streaming", "deployment"]},
    {"id": "generative-media", "name": "Generative Media",
     "tags": ["image generation", "video generation", "text to speech", "speech to text",
               "music", "audio", "comfyui", "diffusion", "controlnet", "stable diffusion",
               "generative"]},
    {"id": "computer-vision", "name": "Computer Vision",
     "tags": ["computer vision", "point cloud", "3d vision", "object detection",
               "segmentation", "depth estimation", "slam", "optical flow",
               "3d reconstruction", "pose estimation", "vision"]},
    {"id": "robotics", "name": "Robotics",
     "tags": ["robotics", "robot", "humanoid", "simulation", "ros", "motion planning",
               "grasping", "manipulation", "navigation", "control systems"]},
    {"id": "nlp-text", "name": "NLP & Text",
     "tags": ["nlp", "natural language", "text classification", "named entity",
               "sentiment", "summarization", "translation", "question answering",
               "information extraction", "parsing", "tokenization"]},
    {"id": "ml-platform", "name": "ML Platform & Infrastructure",
     "tags": ["ml platform", "mlops", "pipeline", "orchestration", "feature store",
               "data pipeline", "kubeflow", "airflow", "prefect", "infrastructure",
               "platform"]},
    {"id": "safety-alignment", "name": "Safety & Alignment",
     "tags": ["safety", "alignment", "fairness", "bias", "interpretability",
               "explainability", "robustness", "adversarial", "toxicity", "guardrail"]},
    {"id": "coding-devtools", "name": "Coding & Dev Tools",
     "tags": ["code generation", "code completion", "copilot", "devin", "cursor",
               "devtools", "ide", "coding assistant", "code review", "debugging tool",
               "software engineering"]},
    {"id": "data-science", "name": "Data Science & Analytics",
     "tags": ["data science", "analytics", "visualization", "pandas", "numpy",
               "scikit-learn", "sklearn", "statistical", "jupyter", "notebook"]},
    {"id": "healthcare-bio", "name": "Healthcare & Biology",
     "tags": ["healthcare", "medical", "clinical", "biology", "genomics", "protein",
               "drug discovery", "bioinformatics", "radiology", "pathology"]},
    {"id": "finance-legal", "name": "Finance & Legal",
     "tags": ["finance", "trading", "quantitative", "legal", "contract", "compliance",
               "risk", "fraud detection", "fintech"]},
    {"id": "multimodal", "name": "Multimodal AI",
     "tags": ["multimodal", "vision-language", "vlm", "clip", "image-text",
               "audio-visual", "cross-modal"]},
    {"id": "edge-mobile", "name": "Edge & Mobile AI",
     "tags": ["edge", "mobile", "embedded", "iot", "on-device", "tflite",
               "coreml", "onnx", "wasm", "webassembly"]},
    {"id": "search-knowledge", "name": "Search & Knowledge",
     "tags": ["search", "knowledge base", "wiki", "qa system", "question answering",
               "information retrieval", "index", "elasticsearch", "opensearch"]},
    {"id": "other", "name": "Other AI / ML",
     "tags": ["machine learning", "deep learning", "neural network", "ai", "ml",
               "artificial intelligence"]},
]


def _assign_categories_from_tags(tags: list[str]) -> list[dict]:
    """Return list of {category_id, category_name, is_primary} dicts.

    Matching is case-insensitive: a category wins when any keyword
    appears as a substring in a tag (keyword-in-tag direction only).  The category with the most
    keyword hits is marked is_primary.
    """
    tags_lower = [t.lower() for t in tags]
    scores: dict[str, int] = {}
    for cat in _CATEGORIES:
        for kw in cat["tags"]:
            kw_l = kw.lower()
            if any(kw_l in tl for tl in tags_lower):
                scores[cat["id"]] = scores.get(cat["id"], 0) + 1

    if not scores:
        return []

    max_score = max(scores.values())
    result = []
    for cat in _CATEGORIES:
        if cat["id"] in scores:
            result.append({
                "category_id": cat["id"],
                "category_name": cat["name"],
                "is_primary": scores[cat["id"]] == max_score and not any(
                    r["is_primary"] for r in result  # only first max wins
                ),
            })
    return result

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
    # Use explicit SQL strings (not f-strings) to avoid dynamic table name injection.
    _table_count_sql: dict[str, str] = {
        "repos":             "SELECT COUNT(*) FROM repos",
        "repo_tags":         "SELECT COUNT(*) FROM repo_tags",
        "repo_categories":   "SELECT COUNT(*) FROM repo_categories",
        "repo_taxonomy":     "SELECT COUNT(*) FROM repo_taxonomy",
        "taxonomy_values":   "SELECT COUNT(*) FROM taxonomy_values",
        "repo_ai_dev_skills":"SELECT COUNT(*) FROM repo_ai_dev_skills",
        "repo_pm_skills":    "SELECT COUNT(*) FROM repo_pm_skills",
        "repo_languages":    "SELECT COUNT(*) FROM repo_languages",
    }
    counts: dict[str, int] = {}
    for table, sql in _table_count_sql.items():
        row = await db.execute(text(sql))
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


@router.post("/admin/backfill/categories", response_model=dict)
async def backfill_categories(
    batch_size: int = Query(default=200, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
    _admin_key: None = Depends(require_admin_key),
):
    """
    Re-derive repo_categories for all repos from their current repo_tags.

    Idempotent: uses INSERT … ON CONFLICT DO NOTHING so existing rows are
    preserved and the endpoint can be called repeatedly without data loss.

    Returns {processed, assigned, skipped}.
    """
    BATCH = batch_size
    offset = 0
    processed = 0
    assigned = 0
    skipped = 0

    # Build full tag map once (tags are small, fits in memory for ~1 500 repos)
    tags_result = await db.execute(text(
        "SELECT repo_id::text, tag FROM repo_tags"
    ))
    tags_by_repo: dict[str, list[str]] = {}
    for row in tags_result.fetchall():
        tags_by_repo.setdefault(str(row.repo_id), []).append(row.tag)

    while True:
        repo_result = await db.execute(text(
            "SELECT id::text FROM repos ORDER BY updated_at DESC LIMIT :lim OFFSET :off"
        ), {"lim": BATCH, "off": offset})
        repo_ids = [r[0] for r in repo_result.fetchall()]
        if not repo_ids:
            break

        for repo_id in repo_ids:
            processed += 1
            tags = tags_by_repo.get(repo_id, [])
            if not tags:
                skipped += 1
                continue

            cats = _assign_categories_from_tags(tags)
            for cat in cats:
                try:
                    await db.execute(text(
                        """
                        INSERT INTO repo_categories
                            (repo_id, category_id, category_name, is_primary)
                        VALUES
                            (:repo_id, :cat_id, :cat_name, :is_primary)
                        ON CONFLICT (repo_id, category_id) DO UPDATE
                            SET category_name = EXCLUDED.category_name,
                                is_primary     = EXCLUDED.is_primary
                        """
                    ), {
                        "repo_id": repo_id,
                        "cat_id":  cat["category_id"],
                        "cat_name": cat["category_name"],
                        "is_primary": cat["is_primary"],
                    })
                    assigned += 1
                except Exception as exc:
                    logger.warning("Category insert failed for %s / %s: %s",
                                   repo_id, cat["category_id"], exc)

        await db.commit()
        offset += BATCH

    await cache.invalidate("library:full*")
    await cache.invalidate("repos:list:*")
    invalidate_library_cache()

    return {"processed": processed, "assigned": assigned, "skipped": skipped}


# ── Security signal models ──────────────────────────────────────────────────

class SecuritySignalsPatch(BaseModel):
    """Payload for manually setting a repo's security risk signals."""
    risk_level: str | None = None        # 'critical' | 'high' | 'medium' | 'low'
    incident_reported: bool = False
    incident_date: str | None = None     # ISO date, e.g. "2024-05-20"
    incident_url: str | None = None      # link to advisory / blog post / CVE
    incident_summary: str | None = None  # one-sentence human-readable summary


@router.patch(
    "/admin/repos/{repo_name}/security",
    dependencies=[Depends(require_admin_key)],
    summary="Set security risk signals for a repo",
)
async def set_repo_security_signals(
    repo_name: str,
    payload: SecuritySignalsPatch,
    db: AsyncSession = Depends(get_db),
):
    """
    Manually mark a repo with security risk metadata.
    Creates or replaces the security_signals JSONB on the matching repo row.

    Example body for LiteLLM-style supply-chain incident:
        {
          "risk_level": "critical",
          "incident_reported": true,
          "incident_date": "2024-05-20",
          "incident_url": "https://github.com/BerriAI/litellm/issues/3668",
          "incident_summary": "Malicious PyPI package published; credentials at risk"
        }
    """
    result = await db.execute(
        select(Repo).where(Repo.name == repo_name)
    )
    repo = result.scalar_one_or_none()
    if repo is None:
        raise HTTPException(status_code=404, detail=f"Repo '{repo_name}' not found")

    repo.security_signals = {
        "risk_level": payload.risk_level,
        "incident_reported": payload.incident_reported,
        "incident_date": payload.incident_date,
        "incident_url": payload.incident_url,
        "incident_summary": payload.incident_summary,
    }
    await db.commit()

    # Bust all library caches so the next page load reflects the update
    await cache.invalidate("library:full*")
    await cache.invalidate("repos:list:*")
    invalidate_library_cache()

    return {
        "repo": repo_name,
        "security_signals": repo.security_signals,
    }


@router.delete(
    "/admin/repos/{repo_name}/security",
    dependencies=[Depends(require_admin_key)],
    summary="Clear security risk signals for a repo",
)
async def clear_repo_security_signals(
    repo_name: str,
    db: AsyncSession = Depends(get_db),
):
    """Remove all security signals from a repo (set to NULL)."""
    result = await db.execute(select(Repo).where(Repo.name == repo_name))
    repo = result.scalar_one_or_none()
    if repo is None:
        raise HTTPException(status_code=404, detail=f"Repo '{repo_name}' not found")

    repo.security_signals = None
    await db.commit()
    invalidate_library_cache()

    return {"repo": repo_name, "security_signals": None}
