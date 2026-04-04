"""Backfill repo_categories from repo_tags.

Migration 018 restored repo_tags from repo_taxonomy.  This migration
derives repo_categories from those tags using the same 21-category
keyword-matching rules as the ingestion pipeline
(ingestion/enrichment/taxonomy.py).

Idempotent: INSERT … ON CONFLICT DO UPDATE so safe to re-run.

Revision ID: 019
Revises: 018
"""

from alembic import op
from sqlalchemy import text

revision = "019"
down_revision = "018"
branch_labels = None
depends_on = None

# ── 21 categories (mirrors ingestion/enrichment/taxonomy.py) ─────────────────
# Each entry: (category_id, category_name, [keywords]).
# A repo wins a category when ANY keyword is a case-insensitive substring of
# ANY of its tags.  The category with the most keyword hits becomes primary.
_CATEGORIES = [
    ("foundation-models", "Foundation Models", [
        "large language model", "transformer", "openai", "anthropic", "claude",
        "google ai", "huggingface", "long context", "multimodal", "quantization",
        "llama", "gguf", "gpt", "llm", "foundational model",
    ]),
    ("ai-agents", "AI Agents", [
        "ai agent", "multi-agent", "autonomous", "agent memory", "planning",
        "chain-of-thought", "tool use", "langchain", "langgraph", "crewai",
        "autogen", "mcp", "prompt engineering", "context engineering",
        "structured output", "function calling", "agentic",
    ]),
    ("rag-retrieval", "RAG & Retrieval", [
        "rag", "vector database", "embedding", "knowledge graph",
        "semantic search", "hybrid search", "reranking", "llamaindex",
        "document processing", "chunking", "retrieval",
    ]),
    ("model-training", "Model Training", [
        "fine-tuning", "reinforcement learning", "lora", "peft", "rlhf",
        "synthetic data", "dataset", "training", "unsloth", "axolotl",
        "trl", "deepspeed", "fsdp", "pytorch", "tensorflow", "keras", "jax",
    ]),
    ("evals-benchmarking", "Evals & Benchmarking", [
        "eval", "benchmark", "model evaluation", "llm testing", "red teaming",
        "safety evaluation", "mmlu", "humaneval", "code evaluation", "alignment",
    ]),
    ("observability", "Observability & Monitoring", [
        "observability", "tracing", "monitoring", "llm monitoring", "logging",
        "debugging", "langsmith", "phoenix", "mlflow", "weights & biases",
        "experiment tracking",
    ]),
    ("inference-serving", "Inference & Serving", [
        "inference", "llm serving", "model optimization", "vllm", "tensorrt",
        "triton", "ollama", "tgi", "batching", "gpu", "cuda",
        "real-time", "streaming", "deployment",
    ]),
    ("generative-media", "Generative Media", [
        "image generation", "video generation", "text to speech", "speech to text",
        "music generation", "audio ai", "comfyui", "diffusion", "controlnet",
        "stable diffusion", "generative",
    ]),
    ("computer-vision", "Computer Vision", [
        "computer vision", "point cloud", "3d vision", "object detection",
        "segmentation", "depth estimation", "slam", "optical flow",
        "3d reconstruction", "pose estimation", "vision",
    ]),
    ("robotics", "Robotics", [
        "robotics", "robot", "humanoid", "simulation", "ros", "motion planning",
        "grasping", "manipulation", "navigation", "control systems",
    ]),
    ("nlp-text", "NLP & Text", [
        "nlp", "natural language", "text classification", "named entity",
        "sentiment", "summarization", "translation", "question answering",
        "information extraction", "parsing", "tokenization",
    ]),
    ("mlops-infrastructure", "MLOps & Infrastructure", [
        "mlops", "docker", "kubernetes", "ci/cd", "pipeline",
        "feature store", "model registry", "data versioning",
        "dvc", "zenml", "prefect", "airflow", "ray",
        "distributed computing", "devops",
    ]),
    ("dev-tools", "Dev Tools & Automation", [
        "cli tool", "automation", "sdk", "developer tools",
        "code generation", "coding assistant", "systems", "security",
        "database", "backend", "frontend", "full stack",
    ]),
    ("cloud-platforms", "Cloud & Platforms", [
        "google cloud", "aws", "azure", "vertex ai", "sagemaker", "bedrock",
    ]),
    ("learning-resources", "Learning Resources", [
        "tutorial", "course", "roadmap", "cheat sheet", "curated list",
        "interview prep", "research", "open source", "workshop",
    ]),
    ("industry-healthcare", "Industry: Healthcare", [
        "healthcare ai", "medical imaging", "drug discovery",
        "clinical nlp", "bioinformatics", "genomics",
    ]),
    ("industry-fintech", "Industry: FinTech", [
        "fintech", "trading ai", "risk modeling", "fraud detection",
        "financial nlp",
    ]),
    ("spatial-xr", "Spatial & XR", [
        "xr", "virtual reality", "augmented reality", "spatial ai",
        "arkit", "arcore", "meta quest", "apple vision",
    ]),
    ("data-science", "Data Science & Analytics", [
        "data science", "analytics", "visualization", "pandas", "numpy",
        "scikit-learn", "sklearn", "statistical", "jupyter",
    ]),
    ("safety-alignment", "Safety & Alignment", [
        "safety", "alignment", "fairness", "bias", "interpretability",
        "explainability", "robustness", "adversarial", "guardrail",
    ]),
    ("other", "Other AI / ML", [
        "machine learning", "deep learning", "neural network", "artificial intelligence",
    ]),
]


def _assign(tags_lower: list[str]) -> list[tuple[str, str, bool]]:
    """Return (cat_id, cat_name, is_primary) tuples for matching categories."""
    scores: dict[str, int] = {}
    for cat_id, _cat_name, keywords in _CATEGORIES:
        for kw in keywords:
            kw_l = kw.lower()
            if any(kw_l in tl for tl in tags_lower):
                scores[cat_id] = scores.get(cat_id, 0) + 1

    if not scores:
        return []

    max_score = max(scores.values())
    primary_assigned = False
    result = []
    for cat_id, cat_name, _ in _CATEGORIES:
        if cat_id not in scores:
            continue
        is_primary = scores[cat_id] == max_score and not primary_assigned
        if is_primary:
            primary_assigned = True
        result.append((cat_id, cat_name, is_primary))
    return result


def upgrade() -> None:
    conn = op.get_bind()

    # Fetch all repo_id → [tag, ...] mappings in one query
    rows = conn.execute(text(
        "SELECT repo_id::text, tag FROM repo_tags ORDER BY repo_id"
    )).fetchall()

    tags_by_repo: dict[str, list[str]] = {}
    for repo_id, tag in rows:
        tags_by_repo.setdefault(repo_id, []).append(tag)

    # Assign categories and bulk-insert
    batch: list[dict] = []
    for repo_id, tags in tags_by_repo.items():
        tags_lower = [t.lower() for t in tags]
        for cat_id, cat_name, is_primary in _assign(tags_lower):
            batch.append({
                "repo_id": repo_id,
                "cat_id": cat_id,
                "cat_name": cat_name,
                "is_primary": is_primary,
            })

    if not batch:
        return

    # Insert in chunks of 500 to stay within parameter limits
    CHUNK = 500
    for i in range(0, len(batch), CHUNK):
        chunk = batch[i: i + CHUNK]
        conn.execute(text("""
            INSERT INTO repo_categories (repo_id, category_id, category_name, is_primary)
            VALUES (:repo_id, :cat_id, :cat_name, :is_primary)
            ON CONFLICT (repo_id, category_id) DO UPDATE
                SET category_name = EXCLUDED.category_name,
                    is_primary     = EXCLUDED.is_primary
        """), chunk)


def downgrade() -> None:
    # Cannot distinguish migrated rows from pre-existing ones — no-op.
    pass
