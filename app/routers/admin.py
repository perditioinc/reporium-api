import asyncio
import json
import logging
import os
from dataclasses import dataclass, field as dc_field
from datetime import date

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy import delete, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import require_admin_key, verify_api_key
from app.cache import cache
from app.database import get_db
from app.models.repo import IngestRun, Repo, RepoCategory, RepoEmbedding, RepoTag
from app.rate_limit import rate_limit_storage
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
_limiter = Limiter(key_func=get_remote_address, storage_uri=rate_limit_storage)

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
@_limiter.limit("10/minute")
async def prune_tags(
    request: Request,
    dry_run: bool = Query(default=False),
    db: AsyncSession = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
    _admin_key: None = Depends(require_admin_key),
):
    """Count or delete noise tags from repo_tags. Requires API and admin keys."""
    return await _prune_noise_tags(db, dry_run=dry_run)


@router.post("/admin/quality/compute", response_model=dict)
@_limiter.limit("10/minute")
async def compute_quality_signals(
    request: Request,
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
@_limiter.limit("10/minute")
async def backfill_embeddings(
    request: Request,
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
            async with db.begin_nested():  # savepoint per row
                await db.execute(text(
                    """
                    INSERT INTO repo_embeddings (repo_id, embedding, model, generated_at, embedding_vec)
                    VALUES (:repo_id, :embedding, 'all-MiniLM-L6-v2', NOW(), CAST(:embedding_vec AS vector))
                    ON CONFLICT (repo_id) DO UPDATE
                        SET embedding = EXCLUDED.embedding,
                            embedding_vec = EXCLUDED.embedding_vec,
                            generated_at = NOW()
                    """
                ), {"repo_id": str(row.id), "embedding": vec_json, "embedding_vec": vec_json})
            backfilled += 1
        except Exception as exc:
            errors.append(f"{row.name}: {exc}")
            logger.warning("Embedding insert failed for %s: %s", row.name, exc)

    # Also backfill embedding_vec for any rows that have embedding but no vec
    try:
        async with db.begin_nested():
            sync_result = await db.execute(text(
                """
                UPDATE repo_embeddings
                SET embedding_vec = embedding::vector
                WHERE embedding_vec IS NULL AND embedding IS NOT NULL
                """
            ))
            synced = sync_result.rowcount
    except Exception as exc:
        logger.warning("Vec sync failed: %s", exc)
        synced = 0

    await db.commit()
    return {"backfilled": backfilled, "vec_synced": synced, "errors": errors}


# ── AI agent protocol & framework tagging ────────────────────────────────────
# Detects protocols, AI frameworks, and SDK indicators from repo name,
# description, readme_summary, and tags. Zero LLM cost — pure keyword matching.
#
# Keywords are split into two confidence tiers:
#   "strong"  — Unambiguous identifiers. One match is enough to tag.
#               e.g. "langchain", "crewai", "mcp-server"
#   "weak"    — Generic words that ALSO happen to be library/concept names.
#               e.g. "transformers", "datasets", "eval", "adapter"
#               A weak keyword only fires when CORROBORATED by:
#                 • at least ONE strong keyword for the same tag, OR
#                 • at least TWO weak keywords for the same tag
#
# This eliminates false positives like tagging a generic CSV-parsing repo
# "HuggingFace" just because its description mentions "datasets".

@dataclass
class TagRule:
    """One tagging rule with confidence-tiered keywords."""
    tag: str
    strong: list[str] = dc_field(default_factory=list)  # one match → tag
    weak: list[str] = dc_field(default_factory=list)    # needs corroboration


_TAG_RULES: list[TagRule] = [
    # ── Protocols ──
    TagRule("MCP", strong=[
        "mcp", "model context protocol", "mcp-server", "mcp server",
        "model-context-protocol", "mcp-client", "mcp plugin",
        "mcp-tool", "modelcontextprotocol",
    ]),
    TagRule("CLI", strong=[
        "cli tool", "command-line tool", "command line interface",
        "cli interface", "cli app", "cli utility", "cli framework",
        "cli library",
    ], weak=[
        "command-line", "terminal tool",
    ]),
    TagRule("A2A", strong=[
        "a2a", "agent-to-agent", "agent to agent", "a2a protocol",
        "inter-agent",
    ], weak=[
        "multi-agent communication", "agent communication", "agent protocol",
    ]),
    # ── AI Frameworks & SDKs ──
    TagRule("LangChain", strong=[
        "langchain", "lang-chain", "langchain-community", "langserve",
        "langsmith", "langgraph",
    ]),
    TagRule("LlamaIndex", strong=[
        "llamaindex", "llama-index", "llama_index", "gpt-index",
    ]),
    TagRule("CrewAI", strong=[
        "crewai", "crew-ai", "crew ai",
    ]),
    TagRule("AutoGen", strong=[
        "autogen", "pyautogen",
    ], weak=[
        "auto-gen", "ag2",
    ]),
    TagRule("Ollama", strong=[
        "ollama",
    ]),
    TagRule("vLLM", strong=[
        "vllm",
    ], weak=[
        "v-llm",
    ]),
    TagRule("HuggingFace", strong=[
        "huggingface", "hugging face", "huggingface.co",
        "from_pretrained",  # unmistakable HF API call
    ], weak=[
        "transformers", "diffusers", "datasets", "accelerate",
        "peft", "trl", "tokenizer",
    ]),
    TagRule("OpenAI", strong=[
        "openai", "openai api", "openai-python",
    ], weak=[
        "gpt-4", "gpt-3", "chatgpt", "whisper", "dall-e",
    ]),
    TagRule("Anthropic", strong=[
        "anthropic", "anthropic api", "claude-api",
    ], weak=[
        "claude",
    ]),
    TagRule("Vercel AI SDK", strong=[
        "ai sdk", "vercel ai", "@ai-sdk",
    ]),
    TagRule("Streamlit", strong=[
        "streamlit",
    ]),
    TagRule("Gradio", strong=[
        "gradio",
    ]),
    TagRule("FastAPI", strong=[
        "fastapi",
    ], weak=[
        "fast-api",
    ]),
    TagRule("Docker", strong=[
        "dockerfile", "docker-compose", "docker compose",
    ], weak=[
        "docker", "containerized",
    ]),
    TagRule("Kubernetes", strong=[
        "kubernetes", "k8s",
    ], weak=[
        "helm chart",
    ]),
    # ── AI Patterns ──
    TagRule("RAG", strong=[
        "rag pipeline", "retrieval augmented", "retrieval-augmented",
        "chromadb", "pinecone", "weaviate", "qdrant", "milvus",
    ], weak=[
        "vector search", "vector database", "vector store", "chroma",
    ]),
    TagRule("Fine-Tuning", strong=[
        "fine-tune", "fine-tuning", "finetuning", "finetune",
        "lora", "qlora", "rlhf", "dpo",
    ], weak=[
        "adapter", "sft",
    ]),
    TagRule("Agents", strong=[
        "ai agent", "autonomous agent", "agent framework", "agentic",
        "agent orchestration",
    ], weak=[
        "tool-use", "tool use", "function calling", "tool calling",
        "multi-agent",
    ]),
    TagRule("Evaluation", strong=[
        "llm eval", "llm evaluation", "llm benchmark", "llm judge",
        "model evaluation", "red teaming", "evals framework",
    ], weak=[
        "eval", "evaluation", "benchmark", "leaderboard", "scoring",
        "red team",
    ]),
    TagRule("Prompt Engineering", strong=[
        "prompt engineering", "prompt template", "prompt management",
        "prompt optimization", "dspy",
    ], weak=[
        "prompt chain",
    ]),
]


def _match_tag_rule(rule: TagRule, search_text: str) -> bool:
    """
    Return True if the repo text qualifies for this tag.

    Logic:
      • Any strong keyword match → True immediately
      • Weak keywords alone need corroboration:
        – 2+ weak matches → True
        – 1 weak match alone → False (too noisy)
    """
    strong_hits = sum(1 for kw in rule.strong if kw in search_text)
    if strong_hits > 0:
        return True

    weak_hits = sum(1 for kw in rule.weak if kw in search_text)
    return weak_hits >= 2


@router.post("/admin/tags/protocols", response_model=dict)
@_limiter.limit("10/minute")
async def tag_protocols(
    request: Request,
    dry_run: bool = Query(default=False, description="Preview without writing"),
    db: AsyncSession = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
    _admin_key: None = Depends(require_admin_key),
):
    """
    Scan all public repos and tag with protocol indicators (MCP, CLI, A2A)
    and AI framework/SDK tags (LangChain, LlamaIndex, CrewAI, AutoGen,
    Ollama, vLLM, HuggingFace, OpenAI, Anthropic, etc.) using confidence-
    tiered keyword matching.

    Strong keywords (e.g. "langchain", "crewai") tag on a single match.
    Weak keywords (e.g. "transformers", "datasets", "eval") require 2+
    matches to tag — prevents false positives on generic terms.

    Zero LLM cost — pure keyword matching with noise filtering.
    """
    # Fetch repos with their text fields and existing tags
    result = await db.execute(text("""
        SELECT r.id, r.name, r.description, r.readme_summary,
               COALESCE(
                   (SELECT string_agg(t.tag, ' ') FROM repo_tags t WHERE t.repo_id = r.id),
                   ''
               ) AS all_tags
        FROM repos r
        WHERE r.is_private = false
    """))
    rows = result.fetchall()

    tagged: dict[str, list[str]] = {}  # tag → list of repo names
    matched_by: dict[str, dict[str, str]] = {}  # tag → {repo: match_type}
    inserts = 0
    skipped = 0

    for row in rows:
        # Build searchable text (lowercase)
        search_text = " ".join(filter(None, [
            row.name, row.description, row.readme_summary, row.all_tags,
        ])).lower()

        for rule in _TAG_RULES:
            if _match_tag_rule(rule, search_text):
                # Determine which tier matched for diagnostics
                strong_hit = any(kw in search_text for kw in rule.strong)
                match_type = "strong" if strong_hit else "weak×2+"
                if dry_run:
                    tagged.setdefault(rule.tag, []).append(row.name)
                    matched_by.setdefault(rule.tag, {})[row.name] = match_type
                    continue
                try:
                    async with db.begin_nested():
                        await db.execute(text("""
                            INSERT INTO repo_tags (repo_id, tag)
                            VALUES (:repo_id, :tag)
                            ON CONFLICT (repo_id, tag) DO NOTHING
                        """), {"repo_id": str(row.id), "tag": rule.tag})
                    inserts += 1
                    tagged.setdefault(rule.tag, []).append(row.name)
                    matched_by.setdefault(rule.tag, {})[row.name] = match_type
                except Exception:
                    skipped += 1

    if not dry_run and inserts > 0:
        await db.commit()
        invalidate_library_cache()

    return {
        "dry_run": dry_run,
        "tagged_count": sum(len(v) for v in tagged.values()),
        "inserts": inserts,
        "skipped": skipped,
        "tags": {
            k: {
                "count": len(v),
                "repos": sorted(v),
                "match_breakdown": {
                    "strong": sum(1 for r in v if matched_by.get(k, {}).get(r) == "strong"),
                    "weak_corroborated": sum(1 for r in v if matched_by.get(k, {}).get(r) == "weak×2+"),
                },
            }
            for k, v in tagged.items()
        },
    }


@router.post("/admin/taxonomy/bootstrap", response_model=dict)
@_limiter.limit("10/minute")
async def bootstrap_taxonomy(
    request: Request,
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
    _api_key: str = Depends(verify_api_key),
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

def _format_runs(runs) -> list[dict]:
    """Shared formatter for ingestion run records."""
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


@router.get(
    "/runs",
    summary="List recent ingestion runs (public)",
)
async def list_runs_public(
    limit: int = Query(20, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
):
    """Public read-only view of recent ingestion runs (max 50, no error details)."""
    result = await db.execute(
        select(IngestRun).order_by(IngestRun.started_at.desc()).limit(limit)
    )
    runs = result.scalars().all()
    formatted = _format_runs(runs)
    # Strip error details from public endpoint
    for r in formatted:
        r.pop("errors", None)
    return formatted


@router.get(
    "/admin/runs",
    summary="List recent ingestion runs (admin)",
    dependencies=[Depends(require_admin_key), Depends(verify_api_key)],
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
    return _format_runs(runs)


@router.post("/admin/enrichment/trigger", response_model=dict)
@_limiter.limit("10/minute")
async def trigger_enrichment(
    request: Request,
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
    dependencies=[Depends(require_admin_key), Depends(verify_api_key)],
    status_code=201,
)
@_limiter.limit("10/minute")
async def record_run(
    request: Request,
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
@_limiter.limit("10/minute")
async def backfill_categories(
    request: Request,
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
@_limiter.limit("10/minute")
async def set_repo_security_signals(
    request: Request,
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
@_limiter.limit("10/minute")
async def clear_repo_security_signals(
    request: Request,
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


@router.post(
    "/admin/retention/purge-query-logs",
    summary="Manually purge old query_log rows (PII retention)",
)
async def admin_purge_query_logs(
    days: int = Query(default=90, ge=30, le=3650),
    _admin_key: None = Depends(require_admin_key),
) -> dict:
    """Delete query_log rows older than ``days`` days on demand.

    The retention loop runs this automatically every 24 hours; this endpoint
    is for ops/manual cleanup.
    """
    from app.retention import purge_old_query_logs

    count = await purge_old_query_logs(days=days)
    return {"purged": count, "cutoff_days": days}


# ---------------------------------------------------------------------------
# Issue #238 — ask_sessions retention + right-to-be-forgotten (RTBF)
# ---------------------------------------------------------------------------
#
# ``ask_sessions`` stores the user's question and the assistant's answer so
# /intelligence/ask can surface the last few turns as conversational memory.
# That makes it a PII store; two endpoints keep it compliant:
#
#   1. POST /admin/purge-ask-sessions?days=90 — nightly retention purge.
#      Call externally via any cron (Cloud Scheduler, GitHub Actions, etc.);
#      no new in-process scheduler is started (see $0 infra constraint).
#   2. DELETE /admin/ask-sessions/{session_id} — GDPR RTBF: remove every row
#      tied to a single session_id on demand.
#
# Both endpoints are guarded by ``require_admin_key``.


@router.post(
    "/admin/purge-ask-sessions",
    summary="Purge expired ask_sessions rows (retention, closes #238)",
)
async def admin_purge_ask_sessions(
    days: int = Query(default=90, ge=7, le=365),
    _admin_key: None = Depends(require_admin_key),
) -> dict:
    """Delete ask_sessions rows older than ``days`` days.

    Recommended schedule: invoke daily from Cloud Scheduler / GitHub Actions
    cron. ``days`` is bounded to [7, 365] so a misconfigured caller cannot
    accidentally wipe live sessions or retain data indefinitely.
    """
    from app.retention import purge_expired_ask_sessions

    count = await purge_expired_ask_sessions(max_age_days=days)
    return {"purged": count, "max_age_days": days}


@router.delete(
    "/admin/ask-sessions/{session_id}",
    summary="Delete all ask_sessions rows for a session_id (RTBF)",
)
async def admin_delete_ask_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    _admin_key: None = Depends(require_admin_key),
) -> dict:
    """GDPR right-to-be-forgotten: delete every row for one session_id.

    ``session_id`` is a UUID; an invalid value yields 400. The endpoint is
    intentionally idempotent — deleting a non-existent session returns
    ``{"deleted": 0}`` rather than 404 so repeat RTBF invocations succeed.
    """
    import uuid

    try:
        uuid.UUID(session_id)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail="session_id must be a UUID") from exc

    result = await db.execute(
        text(
            "DELETE FROM ask_sessions WHERE session_id = CAST(:sid AS uuid) RETURNING id"
        ),
        {"sid": session_id},
    )
    deleted = result.fetchall()
    await db.commit()
    count = len(deleted)
    logger.info(
        "ask_sessions RTBF delete: session_id=%s deleted=%d", session_id, count
    )
    return {"deleted": count, "session_id": session_id}


# ---------------------------------------------------------------------------
# Audit log endpoint (KAN-governance)
# ---------------------------------------------------------------------------


@router.get("/admin/audit", dependencies=[Depends(require_admin_key)])
async def get_audit_logs(
    request: Request,
    db: AsyncSession = Depends(get_db),
    api_key_hash: str | None = Query(None, description="Filter by SHA-256 of API key"),
    endpoint: str | None = Query(None, description="Filter by endpoint substring"),
    date_from: date | None = Query(None, description="Start date (inclusive)"),
    date_to: date | None = Query(None, description="End date (inclusive)"),
    sandbox_only: bool = Query(False, description="Only show sandbox entries"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """List recent audit log entries (admin-key required).

    Returns timestamp, endpoint, model, tokens, cost, latency, and sandbox flag.
    """
    from app.audit import list_audit_logs

    entries = await list_audit_logs(
        db,
        api_key_hash=api_key_hash,
        endpoint=endpoint,
        date_from=date_from,
        date_to=date_to,
        sandbox_only=sandbox_only,
        limit=limit,
        offset=offset,
    )
    return {"entries": entries, "count": len(entries), "limit": limit, "offset": offset}


# ---------------------------------------------------------------------------
# License SPDX backfill via GitHub REST API
# ---------------------------------------------------------------------------

async def _fetch_license(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    owner: str,
    name: str,
) -> str | None:
    """Fetch license.spdx_id from GitHub REST API for a single repo.

    Returns the SPDX identifier string, or None if unavailable / on error.
    """
    async with semaphore:
        try:
            resp = await client.get(
                f"https://api.github.com/repos/{owner}/{name}",
                timeout=15.0,
            )
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            data = resp.json()
            license_obj = data.get("license")
            if license_obj and isinstance(license_obj, dict):
                spdx = license_obj.get("spdx_id")
                # GitHub returns "NOASSERTION" when it can't identify the license
                if spdx and spdx != "NOASSERTION":
                    return spdx
            return None
        except Exception:
            return None


@router.post("/admin/backfill-licenses", response_model=dict)
@_limiter.limit("5/minute")
async def backfill_licenses(
    request: Request,
    dry_run: bool = Query(default=False, description="Preview without writing"),
    concurrency: int = Query(default=10, ge=1, le=50, description="Max concurrent GitHub API calls"),
    db: AsyncSession = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
    _admin_key: None = Depends(require_admin_key),
):
    """Backfill license_spdx for repos where the field is NULL or empty.

    Calls the free GitHub REST API ``GET /repos/{owner}/{name}`` which returns
    ``license.spdx_id`` at no cost.  Uses ``GITHUB_TOKEN`` env var for
    authenticated requests (5 000 req/hr) when available.

    Returns ``{total, updated, failed, skipped, dry_run}``.
    """
    # Find repos missing license_spdx
    result = await db.execute(text(
        "SELECT id, owner, name FROM repos "
        "WHERE license_spdx IS NULL OR license_spdx = '' "
        "ORDER BY updated_at DESC"
    ))
    rows = result.fetchall()
    total = len(rows)

    if total == 0:
        return {"total": 0, "updated": 0, "failed": 0, "skipped": 0, "dry_run": dry_run}

    if dry_run:
        return {"total": total, "updated": 0, "failed": 0, "skipped": 0, "dry_run": True}

    # Build HTTP headers — use GITHUB_TOKEN if available for 5k/hr rate limit
    headers: dict[str, str] = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    gh_token = os.getenv("GITHUB_TOKEN")
    if gh_token:
        headers["Authorization"] = f"Bearer {gh_token}"

    semaphore = asyncio.Semaphore(concurrency)
    updated = 0
    failed = 0
    skipped = 0

    async with httpx.AsyncClient(headers=headers) as client:
        # Create tasks for all repos
        tasks = [
            _fetch_license(client, semaphore, row.owner, row.name)
            for row in rows
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    for row, spdx in zip(rows, results):
        if isinstance(spdx, Exception):
            failed += 1
            logger.warning("License fetch exception for %s/%s: %s", row.owner, row.name, spdx)
            continue
        if spdx is None:
            skipped += 1
            continue
        try:
            await db.execute(
                text("UPDATE repos SET license_spdx = :spdx WHERE id = :id"),
                {"spdx": spdx, "id": str(row.id)},
            )
            updated += 1
        except Exception as exc:
            failed += 1
            logger.warning("License update failed for %s/%s: %s", row.owner, row.name, exc)

    if updated > 0:
        await db.commit()

    return {"total": total, "updated": updated, "failed": failed, "skipped": skipped, "dry_run": False}


# ---------------------------------------------------------------------------
# Community health signals backfill
# ---------------------------------------------------------------------------

def _parse_last_page(link_header: str | None) -> int | None:
    """Extract the last page number from a GitHub ``Link`` header.

    Returns ``None`` when the header is absent or has no ``rel="last"`` entry
    (which means the first page already contains all results).
    """
    if not link_header:
        return None
    import re
    match = re.search(r'<[^>]*[?&]page=(\d+)[^>]*>;\s*rel="last"', link_header)
    if match:
        return int(match.group(1))
    return None


async def _fetch_community_signals(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    owner: str,
    name: str,
) -> dict | None:
    """Fetch community health signals for a single repo from the free GitHub API.

    Returns a dict with the signal values, or ``None`` on unrecoverable error
    (404, etc.).
    """
    base = f"https://api.github.com/repos/{owner}/{name}"
    signals: dict = {}

    async with semaphore:
        try:
            # 1. Main repo endpoint → has_discussions, open_issues_count
            resp = await client.get(base, timeout=15.0)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            repo_data = resp.json()
            signals["has_discussions"] = repo_data.get("has_discussions", False)

            # 2. Contributors (per_page=1, parse Link header for total)
            resp_c = await client.get(
                f"{base}/contributors",
                params={"per_page": "1", "anon": "true"},
                timeout=15.0,
            )
            if resp_c.status_code == 200:
                last_page = _parse_last_page(resp_c.headers.get("link"))
                signals["contributors_count"] = last_page if last_page else len(resp_c.json())
            elif resp_c.status_code == 204:
                # Empty repo — no contributors
                signals["contributors_count"] = 0

            # 3. Releases (per_page=1, parse Link header for count + latest date)
            resp_r = await client.get(
                f"{base}/releases",
                params={"per_page": "1"},
                timeout=15.0,
            )
            if resp_r.status_code == 200:
                releases = resp_r.json()
                last_page = _parse_last_page(resp_r.headers.get("link"))
                signals["release_count"] = last_page if last_page else len(releases)
                if releases:
                    signals["latest_release_date"] = releases[0].get("published_at")
            else:
                signals["release_count"] = 0

            # 4. Community profile → health_percentage
            resp_cp = await client.get(
                f"{base}/community/profile",
                timeout=15.0,
            )
            if resp_cp.status_code == 200:
                signals["community_health_pct"] = resp_cp.json().get("health_percentage")

            # 5. Issue close rate
            #    closed issues = total from search, open from repo data
            open_issues = repo_data.get("open_issues_count", 0)
            resp_closed = await client.get(
                "https://api.github.com/search/issues",
                params={
                    "q": f"repo:{owner}/{name} type:issue is:closed",
                    "per_page": "1",
                },
                timeout=15.0,
            )
            if resp_closed.status_code == 200:
                closed_count = resp_closed.json().get("total_count", 0)
                total_issues = open_issues + closed_count
                signals["issue_close_rate"] = round(
                    closed_count / total_issues, 4
                ) if total_issues > 0 else None

            # 6. PR merge rate
            resp_closed_pr = await client.get(
                "https://api.github.com/search/issues",
                params={
                    "q": f"repo:{owner}/{name} type:pr is:merged",
                    "per_page": "1",
                },
                timeout=15.0,
            )
            resp_total_pr = await client.get(
                "https://api.github.com/search/issues",
                params={
                    "q": f"repo:{owner}/{name} type:pr",
                    "per_page": "1",
                },
                timeout=15.0,
            )
            if resp_closed_pr.status_code == 200 and resp_total_pr.status_code == 200:
                merged_count = resp_closed_pr.json().get("total_count", 0)
                total_prs = resp_total_pr.json().get("total_count", 0)
                signals["pr_merge_rate"] = round(
                    merged_count / total_prs, 4
                ) if total_prs > 0 else None

            return signals

        except Exception as exc:
            logger.warning("Community signals fetch failed for %s/%s: %s", owner, name, exc)
            return None


@router.post("/admin/backfill-community-signals", response_model=dict)
@_limiter.limit("5/minute")
async def backfill_community_signals(
    request: Request,
    dry_run: bool = Query(default=False, description="Preview without writing"),
    batch_size: int = Query(default=0, ge=0, le=500, description="Max repos to process (0 = all)"),
    db: AsyncSession = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
    _admin_key: None = Depends(require_admin_key),
):
    """Backfill community health signals for repos missing ``community_health_pct``.

    Calls the free GitHub REST API to populate:
    ``contributors_count``, ``release_count``, ``latest_release_date``,
    ``issue_close_rate``, ``pr_merge_rate``, ``has_discussions``, and
    ``community_health_pct``.

    Uses ``GITHUB_TOKEN`` env var for authenticated requests (5 000 req/hr).

    Returns ``{total, updated, failed, skipped, dry_run}``.
    """
    query = (
        "SELECT id, owner, name FROM repos "
        "WHERE community_health_pct IS NULL "
        "ORDER BY updated_at DESC"
    )
    if batch_size > 0:
        query += f" LIMIT {batch_size}"

    result = await db.execute(text(query))
    rows = result.fetchall()
    total = len(rows)

    if total == 0:
        return {"total": 0, "updated": 0, "failed": 0, "skipped": 0, "dry_run": dry_run}

    if dry_run:
        return {"total": total, "updated": 0, "failed": 0, "skipped": 0, "dry_run": True}

    headers: dict[str, str] = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    gh_token = os.getenv("GITHUB_TOKEN")
    if gh_token:
        headers["Authorization"] = f"Bearer {gh_token}"

    semaphore = asyncio.Semaphore(10)
    updated = 0
    failed = 0
    skipped = 0

    async with httpx.AsyncClient(headers=headers) as client:
        tasks = [
            _fetch_community_signals(client, semaphore, row.owner, row.name)
            for row in rows
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    for row, signals in zip(rows, results):
        if isinstance(signals, Exception):
            failed += 1
            logger.warning(
                "Community signals exception for %s/%s: %s", row.owner, row.name, signals
            )
            continue
        if signals is None:
            skipped += 1
            continue
        if not signals:
            skipped += 1
            continue

        try:
            set_clauses = []
            params: dict = {"id": str(row.id)}

            for col in (
                "contributors_count",
                "release_count",
                "latest_release_date",
                "issue_close_rate",
                "pr_merge_rate",
                "has_discussions",
                "community_health_pct",
            ):
                if col in signals and signals[col] is not None:
                    set_clauses.append(f"{col} = :{col}")
                    params[col] = signals[col]

            if not set_clauses:
                skipped += 1
                continue

            await db.execute(
                text(f"UPDATE repos SET {', '.join(set_clauses)} WHERE id = :id"),
                params,
            )
            updated += 1
        except Exception as exc:
            failed += 1
            logger.warning(
                "Community signals update failed for %s/%s: %s", row.owner, row.name, exc
            )

    if updated > 0:
        await db.commit()

    return {"total": total, "updated": updated, "failed": failed, "skipped": skipped, "dry_run": False}


# ---------------------------------------------------------------------------
# KAN pros-cons enrichment — AI-generated developer-focused evaluation
# ---------------------------------------------------------------------------

_PROS_CONS_PROMPT = """You are evaluating an AI/ML open-source tool for developers.

Repository: {owner}/{name} ({stars}\u2605)
Description: {description}
README Summary: {readme_summary}
Problem Solved: {problem_solved}
Quality: {quality}, Maturity: {maturity}
Category: {primary_category}
Language: {primary_language}
Contributors: {contributors_count}
Issue Close Rate: {issue_close_rate}%
Has Tests: {has_tests}, Has CI: {has_ci}
Community Health: {community_health_pct}%

Generate a developer-focused evaluation:
{{
  "pros": ["3-5 specific strengths based on evidence"],
  "cons": ["2-4 honest weaknesses or gaps"],
  "best_for": "1-sentence ideal use case",
  "avoid_if": "1-sentence when NOT to use this",
  "community_verdict": "1-sentence what developers think",
  "comparable_to": ["3-5 well-known alternatives"]
}}

Rules:
- Be specific and evidence-based, not generic
- Reference actual metrics (stars, contributors, test coverage) in pros/cons
- For comparable_to, list real tools even if not in our database
- If data is sparse, say so in cons rather than speculating"""


async def _generate_pros_cons_for_repo(
    repo: Repo,
    anthropic_client,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Call Claude Haiku to generate pros/cons for a single repo.

    Returns dict with keys: pros_cons, input_tokens, output_tokens, error.
    """
    qs = repo.quality_signals or {}
    # Build context values, falling back to sensible defaults for sparse data
    prompt = _PROS_CONS_PROMPT.format(
        owner=repo.owner or "unknown",
        name=repo.name or "unknown",
        stars=repo.stargazers_count or repo.parent_stars or 0,
        description=repo.description or "No description",
        readme_summary=repo.readme_summary or "Not available",
        problem_solved=repo.problem_solved or "Not available",
        quality=qs.get("quality", "unknown"),
        maturity=qs.get("maturity", "unknown"),
        primary_category=repo.primary_category or "Uncategorized",
        primary_language=repo.primary_language or "Unknown",
        contributors_count=repo.contributors_count or 0,
        issue_close_rate=repo.issue_close_rate or 0,
        has_tests=repo.has_tests if repo.has_tests is not None else "Unknown",
        has_ci=repo.has_ci if repo.has_ci is not None else "Unknown",
        community_health_pct=repo.community_health_pct or 0,
    )

    async with semaphore:
        try:
            response = await asyncio.to_thread(
                anthropic_client.messages.create,
                model="claude-haiku-4-5-20250414",
                max_tokens=512,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:
            return {"pros_cons": None, "input_tokens": 0, "output_tokens": 0, "error": str(exc)}

    raw = response.content[0].text.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {
            "pros_cons": None,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "error": f"JSON parse error: {exc}",
        }

    return {
        "pros_cons": data,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "error": None,
    }


@router.post("/admin/enrich-pros-cons", response_model=dict)
@_limiter.limit("5/minute")
async def enrich_pros_cons(
    request: Request,
    batch_size: int = Query(default=50, ge=1, le=500),
    dry_run: bool = Query(default=False),
    force: bool = Query(default=False, description="Re-generate even if pros_cons already exists"),
    db: AsyncSession = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
    _admin_key: None = Depends(require_admin_key),
):
    """Generate AI-powered pros/cons evaluations for repos using Claude Haiku.

    Queries repos where ``pros_cons IS NULL`` (or all repos if ``force=true``).
    Uses ``claude-haiku-4-5-20250414`` with ``max_tokens=512, temperature=0.3``.
    Estimated cost: ~$0.0015/repo.

    Returns ``{total, enriched, failed, skipped, estimated_cost_usd}``.
    """
    from datetime import datetime as _dt, timezone as _tz

    # Build query
    if force:
        stmt = select(Repo).where(Repo.is_private == False).limit(batch_size)  # noqa: E712
    else:
        stmt = (
            select(Repo)
            .where(Repo.is_private == False)  # noqa: E712
            .where(Repo.pros_cons.is_(None))
            .limit(batch_size)
        )

    result = await db.execute(stmt)
    repos = result.scalars().all()
    total = len(repos)

    if total == 0:
        return {"total": 0, "enriched": 0, "failed": 0, "skipped": 0, "estimated_cost_usd": 0.0}

    if dry_run:
        return {"total": total, "enriched": 0, "failed": 0, "skipped": total, "estimated_cost_usd": 0.0, "dry_run": True}

    # Set up Anthropic client
    from app.routers.ingest import _get_anthropic_key
    import anthropic as _anthropic_lib

    api_key = _get_anthropic_key()
    client = _anthropic_lib.Anthropic(api_key=api_key)

    semaphore = asyncio.Semaphore(5)  # Anthropic rate-limit guard

    enriched = 0
    failed = 0
    skipped = 0
    total_input_tokens = 0
    total_output_tokens = 0

    for repo in repos:
        result_data = await _generate_pros_cons_for_repo(repo, client, semaphore)

        total_input_tokens += result_data["input_tokens"]
        total_output_tokens += result_data["output_tokens"]

        if result_data["error"]:
            failed += 1
            logger.warning(
                "Pros/cons enrichment failed for %s/%s: %s",
                repo.owner, repo.name, result_data["error"],
            )
            continue

        if result_data["pros_cons"] is None:
            skipped += 1
            continue

        repo.pros_cons = result_data["pros_cons"]
        repo.pros_cons_generated_at = _dt.now(_tz.utc)
        enriched += 1

    if enriched > 0:
        await db.commit()

    # Haiku pricing: $0.80/M input, $4.00/M output
    estimated_cost = (total_input_tokens * 0.80 / 1_000_000) + (total_output_tokens * 4.00 / 1_000_000)

    logger.info(
        "Pros/cons enrichment complete: total=%d enriched=%d failed=%d cost=$%.4f",
        total, enriched, failed, estimated_cost,
    )

    return {
        "total": total,
        "enriched": enriched,
        "failed": failed,
        "skipped": skipped,
        "estimated_cost_usd": round(estimated_cost, 6),
    }


# ── SBOM dependency backfill ─────────────────────────────────────────────────

def parse_purl(purl: str) -> tuple[str | None, str | None, str | None]:
    """Parse a Package URL into (ecosystem, name, version).

    purl format: ``pkg:<ecosystem>/<namespace>/<name>@<version>``
    or ``pkg:<ecosystem>/<name>@<version>``

    Returns (ecosystem, name, version_constraint) -- any part may be None
    if the purl cannot be parsed.
    """
    if not purl or not purl.startswith("pkg:"):
        return None, None, None

    body = purl[4:]  # strip "pkg:"
    version: str | None = None
    if "@" in body:
        body, version = body.rsplit("@", 1)

    parts = body.split("/", 2)
    if len(parts) < 2:
        return None, None, None

    ecosystem = parts[0]
    # If there's a namespace (e.g. @scope/name in npm), keep only the final name
    name = parts[-1]
    return ecosystem, name, version


def _extract_deps_from_sbom(sbom: dict) -> list[dict]:
    """Extract dependency info from an SPDX SBOM response.

    Returns a list of dicts with keys: package_name, package_ecosystem,
    version_constraint, is_direct.
    """
    deps: list[dict] = []
    packages = sbom.get("sbom", {}).get("packages", [])

    for pkg in packages:
        # Skip the root package (SPDX document itself)
        spdx_id = pkg.get("SPDXID", "")
        if spdx_id == "SPDXRef-DOCUMENT":
            continue

        # Try to extract from purl in externalRefs
        purl_ref = None
        for ref in pkg.get("externalRefs", []):
            if ref.get("referenceType") == "purl":
                purl_ref = ref.get("referenceLocator")
                break

        if purl_ref:
            ecosystem, name, version = parse_purl(purl_ref)
            if name:
                deps.append({
                    "package_name": name,
                    "package_ecosystem": ecosystem,
                    "version_constraint": version,
                    "is_direct": True,
                })
        else:
            # Fallback: use package name field
            pkg_name = pkg.get("name")
            if pkg_name:
                version_info = pkg.get("versionInfo")
                deps.append({
                    "package_name": pkg_name,
                    "package_ecosystem": None,
                    "version_constraint": version_info,
                    "is_direct": True,
                })

    return deps


@router.post("/admin/backfill-dependencies", response_model=dict)
@_limiter.limit("5/minute")
async def backfill_dependencies(
    request: Request,
    db: AsyncSession = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
    _admin_key: None = Depends(require_admin_key),
):
    """Fetch SBOM dependency data from GitHub for repos with no dependencies stored.

    Uses the free GitHub SBOM API (dependency-graph/sbom) to extract packages
    from SPDX format. Parses purl references to extract ecosystem and version.
    """
    github_token = os.environ.get("GITHUB_TOKEN", "")
    if not github_token:
        raise HTTPException(status_code=500, detail="GITHUB_TOKEN env var not set")

    # Find repos with no entries in repo_dependencies
    result = await db.execute(text("""
        SELECT r.id, r.owner, r.name
        FROM repos r
        LEFT JOIN repo_dependencies d ON d.repo_id = r.id
        WHERE d.id IS NULL
        ORDER BY r.updated_at DESC
    """))
    rows = result.fetchall()

    if not rows:
        return {
            "total_repos": 0,
            "repos_with_deps": 0,
            "dependencies_inserted": 0,
            "failed": 0,
            "skipped": 0,
        }

    total_repos = len(rows)
    repos_with_deps = 0
    dependencies_inserted = 0
    failed = 0
    skipped = 0

    sem = asyncio.Semaphore(10)
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    async def fetch_sbom(client: httpx.AsyncClient, owner: str, name: str) -> dict | None:
        async with sem:
            url = f"https://api.github.com/repos/{owner}/{name}/dependency-graph/sbom"
            try:
                resp = await client.get(url, headers=headers, timeout=30.0)
                if resp.status_code == 404:
                    return None
                if resp.status_code == 403:
                    logger.warning("GitHub SBOM 403 for %s/%s -- rate limited or no access", owner, name)
                    return None
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as exc:
                logger.warning("SBOM fetch error for %s/%s: %s", owner, name, exc)
                return None
            except Exception as exc:
                logger.warning("SBOM fetch exception for %s/%s: %s", owner, name, exc)
                return None

    async with httpx.AsyncClient() as client:
        for row in rows:
            sbom = await fetch_sbom(client, row.owner, row.name)
            if sbom is None:
                skipped += 1
                continue

            deps = _extract_deps_from_sbom(sbom)
            if not deps:
                skipped += 1
                continue

            repo_inserted = 0
            for dep in deps:
                try:
                    await db.execute(text("""
                        INSERT INTO repo_dependencies (id, repo_id, package_name, package_ecosystem, version_constraint, is_direct)
                        VALUES (gen_random_uuid(), :repo_id, :package_name, :package_ecosystem, :version_constraint, :is_direct)
                        ON CONFLICT ON CONSTRAINT uq_repo_dep_repo_pkg_eco DO NOTHING
                    """), {
                        "repo_id": str(row.id),
                        "package_name": dep["package_name"],
                        "package_ecosystem": dep["package_ecosystem"],
                        "version_constraint": dep["version_constraint"],
                        "is_direct": dep["is_direct"],
                    })
                    repo_inserted += 1
                except Exception as exc:
                    logger.warning("Dep insert failed for %s/%s pkg=%s: %s", row.owner, row.name, dep["package_name"], exc)
                    failed += 1

            if repo_inserted > 0:
                repos_with_deps += 1
                dependencies_inserted += repo_inserted

            # Commit per repo to avoid holding large transactions
            try:
                await db.commit()
            except Exception as exc:
                logger.warning("Commit failed after %s/%s: %s", row.owner, row.name, exc)
                failed += 1

    return {
        "total_repos": total_repos,
        "repos_with_deps": repos_with_deps,
        "dependencies_inserted": dependencies_inserted,
        "failed": failed,
        "skipped": skipped,
    }
