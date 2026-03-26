"""
GET /library/full — Returns the complete dataset in the exact shape
the reporium.com frontend expects (LibraryData TypeScript interface).

All fields camelCase, nested objects, all repos in one response.
Cached for 5 minutes to avoid repeated expensive queries.
"""

import asyncio
import logging
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Query, Response
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.cache_redis import redis_cache
from app.database import get_db

logger = logging.getLogger(__name__)

# Map ingestion category names → frontend canonical category names
# Frontend CATEGORIES constant uses these exact names with colors/icons
CATEGORY_MAP = {
    "Agents": "AI Agents",
    "Tooling": "Dev Tools & Automation",
    "Security": "Security & Safety",
    "Observability": "Observability & Monitoring",
    "Research": "Learning Resources",
    "Ocr": "Computer Vision",
    "Vision": "Computer Vision",
    "Llm Serving": "Inference & Serving",
    "Orchestration": "AI Agents",
    "Rag": "RAG & Retrieval",
    "Vector Databases": "MLOps & Infrastructure",
    "Other": "Dev Tools & Automation",
    "Data Processing": "MLOps & Infrastructure",
    "Embeddings": "RAG & Retrieval",
    "Audio": "Industry: Audio & Music",
    "Fine Tuning": "Model Training",
    "Deployment": "MLOps & Infrastructure",
    "Evaluation": "Evals & Benchmarking",
    "Datasets": "Datasets",
}


def _normalize_category(name: str) -> str:
    """Map raw DB category name to the frontend's canonical name."""
    return CATEGORY_MAP.get(name, name)


# System tags — represent repo metadata, not content. Filtered from tag cloud.
SYSTEM_TAGS = {'Active', 'Forked', 'Built by Me', 'Inactive', 'Archived', 'Popular'}

# Known org overrides — (category, displayName) matching frontend buildTaxonomy.ts KNOWN_ORGS.
# Takes priority over the org_category value stored in the DB.
KNOWN_ORG_CATEGORIES: dict = {
    'google':          ('big-tech',  'Google'),
    'google-deepmind': ('ai-lab',    'Google DeepMind'),
    'google-gemini':   ('big-tech',  'Google Gemini'),
    'googlecloudplatform': ('big-tech', 'Google Cloud'),
    'googleapis':      ('big-tech',  'Google APIs'),
    'microsoft':       ('big-tech',  'Microsoft'),
    'meta-llama':      ('big-tech',  'Meta'),
    'facebookresearch':('ai-lab',    'Meta Research'),
    'openai':          ('ai-lab',    'OpenAI'),
    'anthropics':      ('ai-lab',    'Anthropic'),
    'huggingface':     ('ai-lab',    'HuggingFace'),
    'mistralai':       ('ai-lab',    'Mistral AI'),
    'deepseek-ai':     ('ai-lab',    'DeepSeek'),
    'qwenlm':          ('ai-lab',    'Qwen / Alibaba'),
    'nvidia':          ('big-tech',  'NVIDIA'),
    'aws':             ('big-tech',  'Amazon AWS'),
    'apple':           ('big-tech',  'Apple'),
    'langchain-ai':    ('startup',   'LangChain'),
    'vllm-project':    ('startup',   'vLLM'),
    'unslothai':       ('startup',   'Unsloth'),
    'langfuse':        ('startup',   'Langfuse'),
    'chroma-core':     ('startup',   'Chroma'),
    'qdrant':          ('startup',   'Qdrant'),
    'weaviate':        ('startup',   'Weaviate'),
    'infiniflow':      ('startup',   'Infiniflow'),
    'arize-ai':        ('startup',   'Arize AI'),
    'confident-ai':    ('startup',   'Confident AI'),
    'run-llama':       ('startup',   'LlamaIndex'),
    'letta-ai':        ('startup',   'Letta'),
    'mem0ai':          ('startup',   'Mem0'),
    'crewaiinc':       ('startup',   'CrewAI'),
    'agno-agi':        ('startup',   'Agno'),
    'all-hands-ai':    ('startup',   'All Hands AI'),
    'cline':           ('startup',   'Cline'),
    'continuedev':     ('startup',   'Continue'),
    'browser-use':     ('startup',   'Browser Use'),
    'eleutherai':      ('ai-lab',    'EleutherAI'),
    'allenai':         ('ai-lab',    'Allen AI'),
    'stanford-crfm':   ('research',  'Stanford'),
    'mit-han-lab':     ('research',  'MIT Han Lab'),
    'deepset-ai':      ('startup',   'deepset'),
}

# AI Dev Skill taxonomy — 28 skill areas across 6 lifecycle groups.
# Each skill area is a direct match (stored verbatim in repo_ai_dev_skills).

# Ordered list of the 28 skill areas in taxonomy order.
_AI_DEV_SKILLS_ORDERED: list = [
    # Foundation & Training
    "Foundation Model Architecture",
    "Fine-tuning & Alignment",
    "Data Engineering",
    "Synthetic Data",
    # Inference & Deployment
    "Inference & Serving",
    "Model Compression",
    "Edge AI",
    # LLM Application Layer
    "Agents & Orchestration",
    "RAG & Retrieval",
    "Context Engineering",
    "Tool Use",
    "Structured Output",
    "Prompt Engineering",
    "Knowledge Graphs",
    # Eval/Safety/Ops
    "Evaluation",
    "Security & Guardrails",
    "Observability",
    "MLOps",
    "AI Governance",
    # Modality-Specific
    "Computer Vision",
    "Speech & Audio",
    "Generative Media",
    "NLP",
    "Multimodal",
    # Applied AI
    "Coding Assistants",
    "Robotics",
    "AI for Science",
    "Recommendation Systems",
]

# Lifecycle group lookup is now DB-driven (skill_areas table).
# _get_lifecycle_groups(db) queries the DB and caches for 5 minutes.
# This dict is a compile-time fallback used only when the DB is unavailable.
_LIFECYCLE_GROUPS_FALLBACK: dict = {
    "Foundation Model Architecture": "Foundation & Training",
    "Fine-tuning & Alignment": "Foundation & Training",
    "Data Engineering": "Foundation & Training",
    "Synthetic Data": "Foundation & Training",
    "Inference & Serving": "Inference & Deployment",
    "Model Compression": "Inference & Deployment",
    "Edge AI": "Inference & Deployment",
    "Agents & Orchestration": "LLM Application Layer",
    "RAG & Retrieval": "LLM Application Layer",
    "Context Engineering": "LLM Application Layer",
    "Tool Use": "LLM Application Layer",
    "Structured Output": "LLM Application Layer",
    "Prompt Engineering": "LLM Application Layer",
    "Knowledge Graphs": "LLM Application Layer",
    "Evaluation": "Eval / Safety / Ops",
    "Security & Guardrails": "Eval / Safety / Ops",
    "Observability": "Eval / Safety / Ops",
    "MLOps": "Eval / Safety / Ops",
    "AI Governance": "Eval / Safety / Ops",
    "Computer Vision": "Modality-Specific",
    "Speech & Audio": "Modality-Specific",
    "Generative Media": "Modality-Specific",
    "NLP": "Modality-Specific",
    "Multimodal": "Modality-Specific",
    "Coding Assistants": "Applied AI",
    "Robotics": "Applied AI",
    "AI for Science": "Applied AI",
    "Recommendation Systems": "Applied AI",
}

# Backwards-compatible export retained for tests and callers that still import
# the old constant name directly. DB-backed lookup remains the live path.
LIFECYCLE_GROUPS = _LIFECYCLE_GROUPS_FALLBACK

_lifecycle_groups_cache: dict = {}
_LIFECYCLE_GROUPS_TTL = 300  # 5 minutes


async def _get_lifecycle_groups(db: AsyncSession) -> dict:
    """Return {skill_area_name: lifecycle_group}.

    taxonomy_values does not carry a lifecycle_group column, so this function
    returns the compile-time fallback dict which encodes the 28-skill taxonomy.
    The async signature is kept so call sites do not need to change.
    """
    return _LIFECYCLE_GROUPS_FALLBACK

# Keep a set for O(1) membership checks in _build_ai_dev_skill_stats
_AI_DEV_SKILL_SET: set = set(_AI_DEV_SKILLS_ORDERED)

# Legacy reverse-lookup retained for tag-based matching (enrichedTags / topics).
# Maps individual tool/framework tags → the nearest new skill area.
_SKILL_TAG_TO_GROUP: dict = {
    # Observability tools → Observability
    'langfuse': 'Observability', 'phoenix': 'Observability', 'openlit': 'Observability',
    'openllmetry': 'Observability', 'helicone': 'Observability', 'traceloop': 'Observability',
    'weights & biases': 'Observability', 'mlflow': 'Observability',
    'opentelemetry': 'Observability', 'monitoring': 'Observability',
    'tracing': 'Observability', 'llm monitoring': 'Observability',
    # Evaluation tools → Evaluation
    'deepeval': 'Evaluation', 'ragas': 'Evaluation', 'promptfoo': 'Evaluation',
    'lm eval harness': 'Evaluation', 'evals': 'Evaluation', 'benchmarking': 'Evaluation',
    'red teaming': 'Evaluation', 'garak': 'Evaluation', 'pyrit': 'Evaluation',
    'mmlu': 'Evaluation', 'humaneval': 'Evaluation',
    # Inference tools → Inference & Serving
    'vllm': 'Inference & Serving', 'sglang': 'Inference & Serving', 'tgi': 'Inference & Serving',
    'triton': 'Inference & Serving', 'tensorrt': 'Inference & Serving', 'onnx': 'Inference & Serving',
    'llama.cpp': 'Inference & Serving', 'llamafile': 'Inference & Serving',
    'llm serving': 'Inference & Serving', 'quantization': 'Model Compression',
    'speculative decoding': 'Inference & Serving', 'kv cache': 'Inference & Serving',
    'gpu / cuda': 'Inference & Serving', 'inference': 'Inference & Serving',
    # Training tools → Fine-tuning & Alignment
    'unsloth': 'Fine-tuning & Alignment', 'axolotl': 'Fine-tuning & Alignment',
    'trl': 'Fine-tuning & Alignment', 'torchtune': 'Fine-tuning & Alignment',
    'lora / peft': 'Fine-tuning & Alignment', 'rlhf': 'Fine-tuning & Alignment',
    'dpo': 'Fine-tuning & Alignment', 'grpo': 'Fine-tuning & Alignment',
    'deepspeed': 'Fine-tuning & Alignment', 'fsdp': 'Fine-tuning & Alignment',
    'synthetic data': 'Synthetic Data', 'distillation': 'Fine-tuning & Alignment',
    'fine-tuning': 'Fine-tuning & Alignment', 'mergekit': 'Fine-tuning & Alignment',
    # Structured output → Structured Output
    'instructor': 'Structured Output', 'outlines': 'Structured Output',
    'guidance': 'Structured Output', 'guardrails': 'Security & Guardrails',
    'nemo guardrails': 'Security & Guardrails', 'structured output': 'Structured Output',
    'tool use': 'Tool Use', 'pydantic': 'Structured Output',
    # Agents → Agents & Orchestration
    'ai agents': 'Agents & Orchestration', 'langchain': 'Agents & Orchestration',
    'langgraph': 'Agents & Orchestration', 'dspy': 'Agents & Orchestration',
    'semantic kernel': 'Agents & Orchestration', 'haystack': 'Agents & Orchestration',
    'agno': 'Agents & Orchestration', 'crewai': 'Agents & Orchestration',
    'autogen': 'Agents & Orchestration', 'swarm': 'Agents & Orchestration',
    'openai agents sdk': 'Agents & Orchestration', 'multi-agent': 'Agents & Orchestration',
    'mcp': 'Tool Use', 'autonomous systems': 'Agents & Orchestration',
    # RAG → RAG & Retrieval
    'rag': 'RAG & Retrieval', 'vector database': 'RAG & Retrieval',
    'embeddings': 'RAG & Retrieval', 'knowledge graph': 'Knowledge Graphs',
    'chroma': 'RAG & Retrieval', 'qdrant': 'RAG & Retrieval', 'milvus': 'RAG & Retrieval',
    'weaviate': 'RAG & Retrieval', 'pinecone': 'RAG & Retrieval', 'pgvector': 'RAG & Retrieval',
    'reranking': 'RAG & Retrieval', 'hybrid search': 'RAG & Retrieval',
    'graphrag': 'Knowledge Graphs', 'document processing': 'RAG & Retrieval',
    'llamaindex': 'RAG & Retrieval', 'lightrag': 'RAG & Retrieval',
    # Context → Context Engineering
    'context engineering': 'Context Engineering', 'agent memory': 'Context Engineering',
    'letta / memgpt': 'Context Engineering', 'mem0': 'Context Engineering',
    'long context': 'Context Engineering', 'planning / cot': 'Context Engineering',
    'prompt engineering': 'Prompt Engineering',
    # Security → Security & Guardrails
    'ai safety': 'Security & Guardrails', 'prompt injection': 'Security & Guardrails',
    'watermarking': 'Security & Guardrails', 'privacy-preserving ai': 'Security & Guardrails',
    'alignment': 'Fine-tuning & Alignment',
    # Coding assistants → Coding Assistants
    'openhands': 'Coding Assistants', 'cline': 'Coding Assistants',
    'continue.dev': 'Coding Assistants', 'aider': 'Coding Assistants',
    'swe-agent': 'Coding Assistants', 'claude code': 'Coding Assistants',
    'gemini cli': 'Coding Assistants', 'kilocode': 'Coding Assistants',
    'cli tool': 'Coding Assistants', 'automation': 'Coding Assistants',
    # MLOps → MLOps
    'mlops': 'MLOps', 'dvc': 'MLOps', 'zenml': 'MLOps', 'prefect': 'MLOps',
    'airflow': 'MLOps', 'ray': 'MLOps', 'kubeflow': 'MLOps',
    'feature store': 'MLOps', 'docker': 'MLOps', 'kubernetes': 'MLOps',
    'ci/cd': 'MLOps', 'model registry': 'MLOps',
    # Multimodal / Vision → Modality-Specific skill areas
    'computer vision': 'Computer Vision', 'image generation': 'Generative Media',
    'video generation': 'Generative Media', 'multimodal ai': 'Multimodal',
    'point cloud / 3d vision': 'Computer Vision', 'object detection': 'Computer Vision',
    'segmentation': 'Computer Vision', 'depth estimation': 'Computer Vision',
    '3d reconstruction': 'Computer Vision', 'text to speech': 'Speech & Audio',
    'speech to text': 'Speech & Audio', 'music / audio ai': 'Speech & Audio',
}


# Maps taxonomy raw_values (as produced by the AI enricher) → canonical 28 skill names.
# Used in _build_ai_dev_skill_stats when the raw_value doesn't exactly match a canonical name.
# Keys are lowercase for case-insensitive lookup.
_TAXONOMY_RAW_TO_CANONICAL: dict[str, str] = {
    # Foundation Model Architecture
    "transformer architecture": "Foundation Model Architecture",
    "large language models": "Foundation Model Architecture",
    "large language model training": "Foundation Model Architecture",
    "large language model integration": "Foundation Model Architecture",
    "neural network architecture design": "Foundation Model Architecture",
    "attention mechanisms": "Foundation Model Architecture",
    "convolutional neural networks": "Foundation Model Architecture",
    "deep learning": "Foundation Model Architecture",
    "machine learning fundamentals": "Foundation Model Architecture",
    "recurrent neural networks": "Foundation Model Architecture",
    "distributed training": "Foundation Model Architecture",
    "pre-training": "Foundation Model Architecture",
    "language model pretraining": "Foundation Model Architecture",
    "gpt architecture": "Foundation Model Architecture",
    "bert": "Foundation Model Architecture",
    "llm architecture": "Foundation Model Architecture",
    "model architecture": "Foundation Model Architecture",
    "neural architecture search": "Foundation Model Architecture",
    # Fine-tuning & Alignment
    "model fine-tuning": "Fine-tuning & Alignment",
    "transfer learning": "Fine-tuning & Alignment",
    "reinforcement learning": "Fine-tuning & Alignment",
    "policy gradient methods": "Fine-tuning & Alignment",
    "deep learning model training": "Fine-tuning & Alignment",
    "reinforcement learning from human feedback": "Fine-tuning & Alignment",
    "rlhf": "Fine-tuning & Alignment",
    "dpo": "Fine-tuning & Alignment",
    "peft": "Fine-tuning & Alignment",
    "lora": "Fine-tuning & Alignment",
    "alignment": "Fine-tuning & Alignment",
    "instruction tuning": "Fine-tuning & Alignment",
    "supervised fine-tuning": "Fine-tuning & Alignment",
    "knowledge distillation": "Fine-tuning & Alignment",
    "model distillation": "Fine-tuning & Alignment",
    # Data Engineering
    "feature engineering": "Data Engineering",
    "data pipeline engineering": "Data Engineering",
    "data preprocessing": "Data Engineering",
    "data pipeline": "Data Engineering",
    "dataset curation": "Data Engineering",
    "data collection": "Data Engineering",
    "data annotation": "Data Engineering",
    "etl pipeline": "Data Engineering",
    "data labeling": "Data Engineering",
    "web scraping": "Data Engineering",
    # Synthetic Data
    "synthetic data generation": "Synthetic Data",
    "data synthesis": "Synthetic Data",
    "data augmentation": "Synthetic Data",
    "generative data": "Synthetic Data",
    # Inference & Serving
    "model deployment": "Inference & Serving",
    "large language model deployment": "Inference & Serving",
    "gpu computing": "Inference & Serving",
    "cuda programming": "Inference & Serving",
    "llm serving": "Inference & Serving",
    "model serving": "Inference & Serving",
    "api deployment": "Inference & Serving",
    "serverless deployment": "Inference & Serving",
    "distributed inference": "Inference & Serving",
    "batch inference": "Inference & Serving",
    # Model Compression
    "model quantization": "Model Compression",
    "model optimization": "Model Compression",
    "neural network pruning": "Model Compression",
    "model pruning": "Model Compression",
    "weight compression": "Model Compression",
    "int8 quantization": "Model Compression",
    # Edge AI
    "edge computing": "Edge AI",
    "on-device ai": "Edge AI",
    "embedded ai": "Edge AI",
    "iot ai": "Edge AI",
    "mobile ai": "Edge AI",
    "tinyml": "Edge AI",
    # Agents & Orchestration
    "agent orchestration": "Agents & Orchestration",
    "multi-agent systems": "Agents & Orchestration",
    "ai agent development": "Agents & Orchestration",
    "agentic ai systems": "Agents & Orchestration",
    "agentic ai development": "Agents & Orchestration",
    "ai agent architecture": "Agents & Orchestration",
    "ai agent orchestration": "Agents & Orchestration",
    "agent communication protocols": "Agents & Orchestration",
    "workflow orchestration": "Agents & Orchestration",
    "conversational ai": "Agents & Orchestration",
    "task planning": "Agents & Orchestration",
    "autonomous agents": "Agents & Orchestration",
    "multi-agent coordination": "Agents & Orchestration",
    "agent framework": "Agents & Orchestration",
    "llm agents": "Agents & Orchestration",
    "ai pipeline": "Agents & Orchestration",
    "chatbot development": "Agents & Orchestration",
    # RAG & Retrieval
    "retrieval-augmented generation": "RAG & Retrieval",
    "semantic search": "RAG & Retrieval",
    "information retrieval": "RAG & Retrieval",
    "vector database management": "RAG & Retrieval",
    "document processing": "RAG & Retrieval",
    "vector search": "RAG & Retrieval",
    "embedding search": "RAG & Retrieval",
    "hybrid search": "RAG & Retrieval",
    "reranking": "RAG & Retrieval",
    "dense retrieval": "RAG & Retrieval",
    "chunking strategies": "RAG & Retrieval",
    "document indexing": "RAG & Retrieval",
    # Context Engineering
    "memory management": "Context Engineering",
    "long context processing": "Context Engineering",
    "context window management": "Context Engineering",
    "agent memory": "Context Engineering",
    "episodic memory": "Context Engineering",
    "working memory": "Context Engineering",
    # Tool Use
    "function calling": "Tool Use",
    "tool integration": "Tool Use",
    "external tool use": "Tool Use",
    "api tool use": "Tool Use",
    "mcp (model context protocol)": "Tool Use",
    "model context protocol": "Tool Use",
    # Structured Output
    "json schema generation": "Structured Output",
    "schema-guided generation": "Structured Output",
    "output parsing": "Structured Output",
    "structured generation": "Structured Output",
    # Knowledge Graphs
    "knowledge graph": "Knowledge Graphs",
    "knowledge graph construction": "Knowledge Graphs",
    "graph databases": "Knowledge Graphs",
    "ontology engineering": "Knowledge Graphs",
    "ontology design": "Knowledge Graphs",
    "semantic web": "Knowledge Graphs",
    "graph rag": "Knowledge Graphs",
    "graphrag": "Knowledge Graphs",
    # Evaluation
    "model evaluation": "Evaluation",
    "ai benchmarking": "Evaluation",
    "benchmarking": "Evaluation",
    "llm evaluation": "Evaluation",
    "performance evaluation": "Evaluation",
    "evals": "Evaluation",
    "red teaming": "Evaluation",
    "adversarial testing": "Evaluation",
    "human evaluation": "Evaluation",
    # Security & Guardrails
    "ai safety": "Security & Guardrails",
    "prompt injection": "Security & Guardrails",
    "adversarial robustness": "Security & Guardrails",
    "ai red teaming": "Security & Guardrails",
    "content filtering": "Security & Guardrails",
    "pii detection": "Security & Guardrails",
    "bias detection": "Security & Guardrails",
    "watermarking": "Security & Guardrails",
    "privacy-preserving ai": "Security & Guardrails",
    # Observability
    "ai monitoring": "Observability",
    "model monitoring": "Observability",
    "llm observability": "Observability",
    "logging": "Observability",
    "tracing": "Observability",
    "cost tracking": "Observability",
    "latency monitoring": "Observability",
    # MLOps
    "hyperparameter optimization": "MLOps",
    "machine learning pipeline": "MLOps",
    "data version control": "MLOps",
    "experiment tracking": "MLOps",
    "model registry": "MLOps",
    "ci/cd for ml": "MLOps",
    "model versioning": "MLOps",
    "feature store": "MLOps",
    "workflow management": "MLOps",
    # AI Governance
    "ai regulation": "AI Governance",
    "responsible ai": "AI Governance",
    "ai ethics": "AI Governance",
    "model transparency": "AI Governance",
    "ai compliance": "AI Governance",
    "explainability": "AI Governance",
    "fairness": "AI Governance",
    # Computer Vision
    "object detection": "Computer Vision",
    "image processing": "Computer Vision",
    "sensor fusion": "Computer Vision",
    "optical character recognition": "Computer Vision",
    "optical character recognition (ocr)": "Computer Vision",
    "video processing": "Computer Vision",
    "slam (simultaneous localization and mapping)": "Computer Vision",
    "slam": "Computer Vision",
    "image segmentation": "Computer Vision",
    "image classification": "Computer Vision",
    "pose estimation": "Computer Vision",
    "3d reconstruction": "Computer Vision",
    "depth estimation": "Computer Vision",
    "face recognition": "Computer Vision",
    "visual question answering": "Computer Vision",
    # Speech & Audio
    "audio signal processing": "Speech & Audio",
    "text-to-speech synthesis": "Speech & Audio",
    "speech recognition": "Speech & Audio",
    "speech processing": "Speech & Audio",
    "speech to text": "Speech & Audio",
    "automatic speech recognition": "Speech & Audio",
    "voice synthesis": "Speech & Audio",
    "audio generation": "Speech & Audio",
    "music generation": "Speech & Audio",
    # Generative Media
    "diffusion models": "Generative Media",
    "generative ai": "Generative Media",
    "image generation": "Generative Media",
    "video generation": "Generative Media",
    "text-to-image generation": "Generative Media",
    "3d generation": "Generative Media",
    "creative ai": "Generative Media",
    "content generation": "Generative Media",
    # NLP
    "natural language processing": "NLP",
    "text classification": "NLP",
    "named entity recognition": "NLP",
    "information extraction": "NLP",
    "machine translation": "NLP",
    "text summarization": "NLP",
    "sentiment analysis": "NLP",
    "question answering": "NLP",
    "relation extraction": "NLP",
    "text mining": "NLP",
    # Multimodal
    "multimodal ai": "Multimodal",
    "multimodal learning": "Multimodal",
    "vision-language models": "Multimodal",
    "visual language model": "Multimodal",
    "audio-visual learning": "Multimodal",
    # Coding Assistants
    "code generation": "Coding Assistants",
    "code intelligence": "Coding Assistants",
    "software development ai": "Coding Assistants",
    "ai-assisted coding": "Coding Assistants",
    "automated code review": "Coding Assistants",
    "code completion": "Coding Assistants",
    "ai code generation": "Coding Assistants",
    "developer tools": "Coding Assistants",
    # Robotics
    "slam (simultaneous localization and mapping)": "Robotics",
    "robot learning": "Robotics",
    "control systems": "Robotics",
    "robot perception": "Robotics",
    "autonomous systems": "Robotics",
    "motion planning": "Robotics",
    # AI for Science
    "time series analysis": "AI for Science",
    "time series forecasting": "AI for Science",
    "graph neural networks": "AI for Science",
    "bioinformatics": "AI for Science",
    "drug discovery": "AI for Science",
    "climate ai": "AI for Science",
    "materials science ai": "AI for Science",
    "computational biology": "AI for Science",
    "scientific computing": "AI for Science",
    # Recommendation Systems
    "collaborative filtering": "Recommendation Systems",
    "matrix factorization": "Recommendation Systems",
    "content-based filtering": "Recommendation Systems",
    "personalization": "Recommendation Systems",
}

router = APIRouter(tags=["Library"])

# In-memory cache: two tiers
#   _cache["page_{page}_{page_size}"] → per-page enriched repos (5 min TTL)
#   _cache["aggregates"]              → stats/categories/tagMetrics across all repos (5 min TTL)
_cache: dict = {}
CACHE_TTL = 300  # 5 minutes


def invalidate_library_cache() -> None:
    """Bust the in-memory /library/full cache. Called by ingest router after writes."""
    _cache.clear()
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(redis_cache.clear_prefix("library:"))
        else:
            loop.run_until_complete(redis_cache.clear_prefix("library:"))
    except Exception:
        logger.warning("invalidate_library_cache: could not clear Redis prefix", exc_info=True)


def sanitize_repo(repo: dict) -> dict:
    """
    Enforce CONTRACT.md — every field gets a valid value, never null.
    Logs a warning for each fallback applied so enrichment gaps are visible.
    """
    name = repo.get("name", "unknown")

    # Required fields — apply fallbacks
    if not repo.get("description"):
        summary = repo.get("readmeSummary") or ""
        repo["description"] = summary[:150] if summary else name
        logger.warning("Contract fallback: %s missing description", name)

    if not repo.get("readmeSummary"):
        repo["readmeSummary"] = repo["description"]
        logger.warning("Contract fallback: %s missing readmeSummary", name)

    if repo.get("openIssuesCount") is None:
        repo["openIssuesCount"] = 0

    if not repo.get("primaryCategory") or repo["primaryCategory"] == "Other":
        repo["primaryCategory"] = "Uncategorized"

    if not repo.get("allCategories"):
        repo["allCategories"] = [repo["primaryCategory"]]
        logger.warning("Contract fallback: %s missing categories", name)

    if not repo.get("enrichedTags"):
        repo["enrichedTags"] = []

    if not repo.get("builders"):
        # For forks, use upstream owner as builder; for owned repos, use the repo owner
        forked_from = repo.get("forkedFrom") or ""
        if forked_from and "/" in forked_from:
            owner = forked_from.split("/")[0]
        else:
            owner = repo.get("fullName", "").split("/")[0] if repo.get("fullName") else "perditioinc"
        repo["builders"] = [{"login": owner, "name": None, "type": "user",
                             "avatarUrl": f"https://avatars.githubusercontent.com/{owner}",
                             "isKnownOrg": False, "orgCategory": "individual"}]

    if not repo.get("pmSkills"):
        repo["pmSkills"] = []
    if not repo.get("industries"):
        repo["industries"] = []
    if not repo.get("aiDevSkills"):
        repo["aiDevSkills"] = []
    if not repo.get("programmingLanguages"):
        repo["programmingLanguages"] = []
    if not repo.get("topics"):
        repo["topics"] = []

    # Date fields — conservative fallbacks only for fields that have safe proxies.
    # Never substitute ingested_at for upstream_created_at — that shows the wrong date.
    last_updated = repo.get("lastUpdated") or ""
    if last_updated:
        ps = repo.get("parentStats")
        if ps and not ps.get("lastCommitDate"):
            ps["lastCommitDate"] = last_updated
        if not repo.get("upstreamLastPushAt"):
            repo["upstreamLastPushAt"] = last_updated if repo.get("isFork") else ""
        # Do NOT fall back upstreamCreatedAt to createdAt/ingested_at — that produces a
        # misleading "Project created: Mar 2026" for repos that were created years ago.
        # Leave it empty until a proper GitHub API backfill populates upstream_created_at.

    # Commit stats — never null
    if not repo.get("commitStats"):
        repo["commitStats"] = {"today": 0, "last7Days": 0, "last30Days": 0,
                               "last90Days": 0, "recentCommits": []}

    # Arrays that must never be null
    for arr_field in ("recentCommits", "commitsLast7Days", "commitsLast30Days", "commitsLast90Days"):
        if not repo.get(arr_field):
            repo[arr_field] = []

    # Objects that must never be null
    if not repo.get("languageBreakdown"):
        repo["languageBreakdown"] = {}
    if not repo.get("languagePercentages"):
        repo["languagePercentages"] = {}

    # Scalars with safe defaults
    if repo.get("stars") is None:
        repo["stars"] = 0
    if repo.get("forks") is None:
        repo["forks"] = 0
    if repo.get("weeklyCommitCount") is None:
        repo["weeklyCommitCount"] = 0
    if repo.get("totalCommitsFetched") is None:
        repo["totalCommitsFetched"] = 0

    return repo


def _iso(val) -> str:
    """Convert a datetime or string to ISO format string."""
    if val is None:
        return ""
    if isinstance(val, datetime):
        return val.isoformat()
    return str(val)


def _build_enriched_repo(repo: dict, languages: list, categories: list,
                         ai_skills: list, tags: list, pm_skills: list,
                         builders: list = None, industries: list = None,
                         lifecycle_groups: dict = None,
                         taxonomy: list = None,
                         commits: list = None) -> dict:
    """Transform a DB repo row + junction data into the frontend EnrichedRepo shape."""
    forked_from = repo.get("forked_from")
    owner = repo.get("owner", "perditioinc")
    name = repo.get("name", "")

    # Build language breakdown
    lang_breakdown = {}
    lang_percentages = {}
    for lang in languages:
        lang_breakdown[lang["language"]] = lang.get("bytes", 0)
        lang_percentages[lang["language"]] = lang.get("percentage", 0)

    # Build parent stats if forked
    parent_stats = None
    if forked_from:
        parts = forked_from.split("/", 1)
        parent_owner = parts[0] if len(parts) == 2 else ""
        parent_repo = parts[1] if len(parts) == 2 else forked_from
        parent_stats = {
            "owner": parent_owner,
            "repo": parent_repo,
            "stars": repo.get("parent_stars") or 0,
            "forks": repo.get("parent_forks") or 0,
            "openIssues": 0,
            "lastCommitDate": _iso(repo.get("upstream_last_push_at")),
            "isArchived": repo.get("parent_is_archived") or False,
            "description": repo.get("description"),
            "url": f"https://github.com/{forked_from}",
        }

    # Fork sync status — behind_by/ahead_by are often stale (0) in the DB,
    # so cross-check with dates: if upstream pushed after our last sync,
    # the fork is behind regardless of what the commit counts say.
    fork_sync = None
    if repo.get("is_fork"):
        behind = repo.get("behind_by") or 0
        ahead = repo.get("ahead_by") or 0

        # Date-based override: compare your_last_push_at vs upstream_last_push_at
        your_push = repo.get("your_last_push_at")
        upstream_push = repo.get("upstream_last_push_at")
        date_says_behind = False
        if your_push and upstream_push:
            from datetime import datetime, timezone
            def _parse_dt(v):
                if isinstance(v, datetime):
                    return v
                if isinstance(v, str):
                    try:
                        return datetime.fromisoformat(v.replace("Z", "+00:00"))
                    except Exception:
                        return None
                return None
            yp = _parse_dt(your_push)
            up = _parse_dt(upstream_push)
            if yp and up and up > yp:
                date_says_behind = True

        if behind == 0 and ahead == 0 and not date_says_behind:
            state = "up-to-date"
        elif date_says_behind or behind > 0:
            if behind > 0 and ahead > 0:
                state = "diverged"
            else:
                state = "behind"
                # If commit count is 0 but dates say behind, estimate ~1
                if behind == 0:
                    behind = 1
        elif ahead > 0:
            state = "ahead"
        else:
            state = "unknown"
        fork_sync = {
            "state": state,
            "behindBy": behind,
            "aheadBy": ahead,
            "upstreamBranch": "main",
        }

    c7 = repo.get("commits_last_7_days") or 0
    c30 = repo.get("commits_last_30_days") or 0
    c90 = repo.get("commits_last_90_days") or 0

    # Bin commit history from repo_commits table into time buckets
    all_commit_data = commits or []
    now = datetime.now(tz=None)  # naive UTC-ish for comparison
    commits_7d = []
    commits_30d = []
    commits_90d = []
    for cmt in all_commit_data:
        date_str = cmt.get("date", "")
        if not date_str:
            continue
        try:
            cdate = datetime.fromisoformat(date_str.replace("Z", "+00:00")).replace(tzinfo=None)
        except (ValueError, TypeError):
            continue
        days_ago = (now - cdate).days
        if days_ago <= 7:
            commits_7d.append(cmt)
        if days_ago <= 30:
            commits_30d.append(cmt)
        if days_ago <= 90:
            commits_90d.append(cmt)

    # Use actual commit counts when DB scalars are 0 but we have commit rows
    effective_c7 = max(c7, len(commits_7d))
    effective_c30 = max(c30, len(commits_30d))
    effective_c90 = max(c90, len(commits_90d))

    all_cats = list(dict.fromkeys(_normalize_category(c["category_name"]) for c in categories))
    primary_cat = all_cats[0] if all_cats else "Dev Tools & Automation"

    # Use the DB-computed full_name (owner/name) as canonical identity.
    # Falls back to constructing it if the column is somehow NULL (shouldn't happen post-006).
    full_name = repo.get("full_name") or f"{owner}/{name}"

    return {
        "id": str(repo.get("id")),  # Stable DB UUID — never use hash() which changes per restart
        "name": name,
        "fullName": full_name,
        "description": repo.get("description"),
        "isFork": repo.get("is_fork", False),
        "forkedFrom": forked_from,
        "language": repo.get("primary_language"),
        "topics": [t["tag"] for t in tags],
        "enrichedTags": list(dict.fromkeys([s["skill"] for s in ai_skills] + [t["tag"] for t in tags])),
        "stars": repo.get("parent_stars") if repo.get("is_fork") else (repo.get("stargazers_count") or 0),
        "forks": repo.get("parent_forks") if repo.get("is_fork") else (repo.get("fork_count") or 0),
        "openIssuesCount": repo.get("open_issues_count") or 0,
        "lastUpdated": _iso(repo.get("updated_at") or repo.get("github_updated_at")),
        "url": repo.get("github_url") or f"https://github.com/{owner}/{name}",
        "isArchived": repo.get("parent_is_archived") or False,
        "readmeSummary": repo.get("readme_summary"),
        "parentStats": parent_stats,
        "recentCommits": all_commit_data[:10],
        "createdAt": _iso(
            repo.get("upstream_created_at")
            if repo.get("forked_from")
            else (repo.get("ingested_at") or repo.get("github_updated_at"))
        ),
        "forkedAt": _iso(repo.get("forked_at")),
        "yourLastPushAt": _iso(repo.get("your_last_push_at")),
        "upstreamLastPushAt": _iso(repo.get("upstream_last_push_at")),
        "upstreamCreatedAt": _iso(repo.get("upstream_created_at")),
        "forkSync": fork_sync,
        "weeklyCommitCount": effective_c7,
        "languageBreakdown": lang_breakdown,
        "languagePercentages": lang_percentages,
        "commitsLast7Days": commits_7d,
        "commitsLast30Days": commits_30d,
        "commitsLast90Days": commits_90d,
        "totalCommitsFetched": len(all_commit_data),
        "primaryCategory": primary_cat,
        "allCategories": all_cats,
        "commitStats": {
            "today": len([c for c in commits_7d if c.get("date") and (now - datetime.fromisoformat(c["date"].replace("Z", "+00:00")).replace(tzinfo=None)).days == 0]),
            "last7Days": effective_c7,
            "last30Days": effective_c30,
            "last90Days": effective_c90,
            "recentCommits": all_commit_data[:5],
        },
        "latestRelease": None,
        "aiDevSkills": [
            {"skill": s["skill"], "lifecycleGroup": (lifecycle_groups or _LIFECYCLE_GROUPS_FALLBACK).get(s["skill"], "")}
            for s in ai_skills
        ],
        "pmSkills": [s["skill"] for s in pm_skills],
        "industries": [ind["industry"] for ind in (industries or [])],
        "programmingLanguages": list(lang_breakdown.keys()),
        "taxonomy": [
            {
                "dimension": t["dimension"],
                "value": t["raw_value"],
                "similarityScore": t["similarity_score"],
                "assignedBy": t["assigned_by"],
            }
            for t in (taxonomy or [])
        ],
        "problemSolved": repo.get("problem_solved"),
        "licenseSpdx": repo.get("license_spdx"),
        "qualitySignals": repo.get("quality_signals"),
        "securitySignals": repo.get("security_signals"),
        "builders": [
            {
                "login": b["login"],
                "name": b.get("display_name") or b["login"],
                "type": "organization" if b.get("is_known_org") else "user",
                "avatarUrl": f"https://avatars.githubusercontent.com/{b['login']}",
                "isKnownOrg": b.get("is_known_org", False),
                "orgCategory": b.get("org_category"),
            }
            for b in (builders or [])
        ],
    }


def _build_stats(repos: list) -> dict:
    """Build LibraryStats from enriched repos."""
    languages = set()
    tag_counter = Counter()
    built = 0
    forked = 0

    for r in repos:
        if r["isFork"]:
            forked += 1
        else:
            built += 1
        if r["language"]:
            languages.add(r["language"])
        for t in r["enrichedTags"]:
            tag_counter[t] += 1

    return {
        "total": len(repos),
        "built": built,
        "forked": forked,
        "languages": sorted(languages),
        "topTags": [t for t, _ in tag_counter.most_common(20)],
    }


def _build_tag_metrics(repos: list) -> list:
    """Build TagMetrics[] from enriched repos. System tags are excluded."""
    tag_repos = defaultdict(list)
    for r in repos:
        for t in r["enrichedTags"]:
            if t not in SYSTEM_TAGS:
                tag_repos[t].append(r)

    metrics = []
    total = len(repos) if repos else 1
    for tag, tag_repo_list in sorted(tag_repos.items()):
        lang_counter = Counter()
        for r in tag_repo_list:
            if r["language"]:
                lang_counter[r["language"]] += 1

        metrics.append({
            "tag": tag,
            "repoCount": len(tag_repo_list),
            "percentage": round(len(tag_repo_list) / total * 100, 1),
            "topLanguage": lang_counter.most_common(1)[0][0] if lang_counter else None,
            "languageBreakdown": dict(lang_counter),
            "updatedLast30Days": 0,
            "updatedLast90Days": 0,
            "olderThan90Days": 0,
            "activityScore": 0,
            "relatedTags": [],
            "mostRecentRepo": tag_repo_list[0]["name"] if tag_repo_list else "",
            "mostRecentDate": tag_repo_list[0]["lastUpdated"] if tag_repo_list else "",
            "repos": [r["name"] for r in tag_repo_list[:20]],
            "avgUpstreamAge": 0,
            "avgTimeSinceForked": 0,
            "mostOutdatedRepo": "",
            "avgBehindBy": 0,
        })

    return metrics


def _build_categories(repos: list) -> list:
    """Build Category[] from enriched repos."""
    cat_repos = defaultdict(list)
    for r in repos:
        for c in r["allCategories"]:
            cat_repos[c].append(r)

    COLORS = {
        "Foundation Models": "#6d28d9",
        "AI Agents": "#7c3aed",
        "RAG & Retrieval": "#2563eb",
        "Model Training": "#0891b2",
        "Evals & Benchmarking": "#6366f1",
        "Observability & Monitoring": "#14b8a6",
        "Inference & Serving": "#8b5cf6",
        "Generative Media": "#ec4899",
        "Computer Vision": "#f97316",
        "Robotics": "#84cc16",
        "Spatial & XR": "#06b6d4",
        "MLOps & Infrastructure": "#f59e0b",
        "Dev Tools & Automation": "#10b981",
        "Cloud & Platforms": "#3b82f6",
        "Learning Resources": "#06b6d4",
        "Industry: Healthcare": "#ef4444",
        "Industry: FinTech": "#10b981",
        "Industry: Audio & Music": "#ec4899",
        "Industry: Gaming": "#8b5cf6",
        "Security & Safety": "#64748b",
        "Data Science & Analytics": "#a855f7",
    }
    ICONS = {
        "Foundation Models": "🧠",
        "AI Agents": "🤖",
        "RAG & Retrieval": "🔍",
        "Model Training": "🔧",
        "Evals & Benchmarking": "📏",
        "Observability & Monitoring": "👁",
        "Inference & Serving": "⚡",
        "Generative Media": "🎨",
        "Computer Vision": "📷",
        "Robotics": "🦾",
        "Spatial & XR": "🥽",
        "MLOps & Infrastructure": "🚀",
        "Dev Tools & Automation": "🛠",
        "Cloud & Platforms": "☁️",
        "Learning Resources": "📚",
        "Industry: Healthcare": "🏥",
        "Industry: FinTech": "💰",
        "Industry: Audio & Music": "🎵",
        "Industry: Gaming": "🎮",
        "Security & Safety": "🔒",
        "Data Science & Analytics": "📊",
    }

    categories = []
    for cat, cat_repo_list in sorted(cat_repos.items()):
        # Collect tags in this category
        cat_tags = set()
        for r in cat_repo_list:
            cat_tags.update(r["enrichedTags"])

        categories.append({
            "id": cat.lower().replace(" ", "-"),
            "name": cat,
            "description": f"Repos related to {cat.lower()}",
            "tags": sorted(cat_tags),
            "repoCount": len(cat_repo_list),
            "color": COLORS.get(cat, "#94a3b8"),
            "icon": ICONS.get(cat, "📦"),
        })

    return categories


def _build_skill_stats(repos: list, skill_field: str) -> list:
    """Build SkillStats[] from enriched repos."""
    skill_repos = defaultdict(list)
    for r in repos:
        for s in r.get(skill_field, []):
            skill_repos[s].append(r)

    total = len(repos) if repos else 1
    stats = []
    for skill, skill_repo_list in sorted(skill_repos.items()):
        count = len(skill_repo_list)
        pct = count / total
        if pct >= 0.1:
            coverage = "strong"
        elif pct >= 0.05:
            coverage = "moderate"
        elif pct >= 0.01:
            coverage = "weak"
        else:
            coverage = "none"

        stats.append({
            "skill": skill,
            "repoCount": count,
            "coverage": coverage,
            "topRepos": [r["name"] for r in sorted(
                skill_repo_list, key=lambda x: x["stars"], reverse=True
            )[:5]],
        })

    return stats


def _build_ai_dev_skill_stats(repos: list, lifecycle_groups: dict = None) -> list:
    """Build AI Dev Skill stats for the 28-skill taxonomy.

    First checks aiDevSkills for direct matches against the canonical 28 skill names.
    Falls back to mapping enrichedTags through _SKILL_TAG_TO_GROUP for legacy data.
    Returns one entry per skill area in taxonomy order.
    """
    skill_repo_names: dict = defaultdict(set)
    skill_top_repos: dict = defaultdict(list)

    for r in repos:
        matched: set = set()

        # Primary: direct match against canonical 28 skill names.
        # Also normalises taxonomy raw_values via _TAXONOMY_RAW_TO_CANONICAL so that
        # values like "Retrieval-Augmented Generation" map to "RAG & Retrieval".
        # aiDevSkills entries are dicts {"skill": ..., "lifecycleGroup": ...}
        for entry in r.get("aiDevSkills", []):
            raw = entry["skill"] if isinstance(entry, dict) else entry
            # Exact canonical match first, then normalised lookup
            skill = raw if raw in _AI_DEV_SKILL_SET else _TAXONOMY_RAW_TO_CANONICAL.get(raw.lower())
            if skill and skill in _AI_DEV_SKILL_SET and skill not in matched:
                matched.add(skill)
                skill_repo_names[skill].add(r["name"])
                skill_top_repos[skill].append((r.get("stars", 0), r["name"]))

        # Fallback: map enrichedTags through legacy tag→skill lookup
        for tag in r.get("enrichedTags", []):
            skill = _SKILL_TAG_TO_GROUP.get(tag.lower())
            if skill and skill in _AI_DEV_SKILL_SET and skill not in matched:
                matched.add(skill)
                skill_repo_names[skill].add(r["name"])
                skill_top_repos[skill].append((r.get("stars", 0), r["name"]))

    total = len(repos) if repos else 1
    stats = []
    for skill in _AI_DEV_SKILLS_ORDERED:
        names = skill_repo_names.get(skill, set())
        count = len(names)
        pct = count / total
        if pct >= 0.1:
            coverage = "strong"
        elif pct >= 0.05:
            coverage = "moderate"
        elif pct >= 0.01:
            coverage = "weak"
        else:
            coverage = "none"
        top = sorted(skill_top_repos.get(skill, []), reverse=True)[:5]
        stats.append({
            "skill": skill,
            "lifecycleGroup": (lifecycle_groups or _LIFECYCLE_GROUPS_FALLBACK).get(skill, ""),
            "repoCount": count,
            "coverage": coverage,
            "topRepos": [name for _, name in top],
        })
    return stats


def _build_builder_stats(repos: list) -> list:
    """Build BuilderStats from enriched repos, sorted by repoCount descending.

    KNOWN_ORG_CATEGORIES overrides the DB org_category so that orgs like
    anthropics / huggingface / facebookresearch are not classified as 'individual'
    and are visible in the frontend's Builders section.
    """
    builder_data: dict = defaultdict(lambda: {
        "repoCount": 0, "totalParentStars": 0, "topRepos": [],
        "category": "individual", "displayName": "", "avatarUrl": "",
    })
    for r in repos:
        for b in r.get("builders", []):
            login = b["login"]
            login_lower = login.lower()
            bd = builder_data[login]
            bd["repoCount"] += 1
            bd["totalParentStars"] += r.get("stars", 0)
            bd["topRepos"].append(r["name"])
            bd["avatarUrl"] = b.get("avatarUrl", "")
            if login_lower in KNOWN_ORG_CATEGORIES:
                cat, display = KNOWN_ORG_CATEGORIES[login_lower]
                bd["category"] = cat
                bd["displayName"] = display
            else:
                bd["category"] = b.get("orgCategory") or "individual"
                if not bd["displayName"]:
                    bd["displayName"] = login

    stats = []
    for login, bd in sorted(builder_data.items(), key=lambda x: x[1]["repoCount"], reverse=True):
        stats.append({
            "login": login,
            "displayName": bd["displayName"] or login,
            "category": bd["category"],
            "repoCount": bd["repoCount"],
            "totalParentStars": bd["totalParentStars"],
            "topRepos": bd["topRepos"][:5],
            "avatarUrl": bd["avatarUrl"],
        })
    return stats[:200]  # Top 200 builders by repo count


async def _fetch_page_repos(
    db: AsyncSession, page: int, page_size: int
) -> tuple[list[dict], int]:
    """
    Fetch one page of enriched repos. Junction data is fetched only for the
    current page's IDs — never the full table — so memory is O(page_size), not O(N).
    Returns (enriched_repos, total_count).
    """
    offset = (page - 1) * page_size

    # Main repos query — paginated
    result = await db.execute(text("""
        SELECT id, name, owner, (owner || '/' || name) AS full_name, description, is_fork, forked_from, primary_language,
               github_url, fork_sync_state, behind_by, ahead_by,
               github_created_at, upstream_created_at, forked_at, your_last_push_at, upstream_last_push_at,
               parent_stars, parent_forks, parent_is_archived, stargazers_count, open_issues_count,
               commits_last_7_days, commits_last_30_days, commits_last_90_days,
               readme_summary, activity_score, ingested_at, updated_at, github_updated_at,
               problem_solved, license_spdx, quality_signals, has_tests, has_ci, security_signals
        FROM repos
        WHERE is_private = false
        ORDER BY COALESCE(parent_stars, stargazers_count, 0) DESC
        LIMIT :lim OFFSET :off
    """), {"lim": page_size, "off": offset})
    rows = result.fetchall()
    columns = list(result.keys())

    count_result = await db.execute(text(
        "SELECT COUNT(*) FROM repos WHERE is_private = false"
    ))
    total = count_result.scalar() or 0

    if not rows:
        return [], total

    # Extract just this page's IDs for targeted junction fetches
    repo_dicts = [dict(zip(columns, row)) for row in rows]
    page_ids = [str(r["id"]) for r in repo_dicts]

    # Fetch junction data only for this page
    def _junction(query: str) -> dict:
        return {}  # placeholder — populated by async calls below

    async def _fetch_junction(q: str) -> list:
        r = await db.execute(text(q), {"ids": page_ids})
        return r.fetchall()

    lang_rows, cat_rows, skill_rows, tag_rows, pm_rows, builder_rows, taxonomy_rows, commit_rows = (
        await asyncio.gather(
            _fetch_junction("SELECT repo_id, language, bytes, percentage FROM repo_languages WHERE repo_id::text = ANY(:ids)"),
            _fetch_junction("SELECT repo_id, category_name, is_primary FROM repo_categories WHERE repo_id::text = ANY(:ids)"),
            _fetch_junction("SELECT repo_id, raw_value AS skill FROM repo_taxonomy WHERE dimension = 'skill_area' AND repo_id::text = ANY(:ids)"),
            _fetch_junction("SELECT repo_id, tag FROM repo_tags WHERE repo_id::text = ANY(:ids)"),
            _fetch_junction("SELECT repo_id, skill FROM repo_pm_skills WHERE repo_id::text = ANY(:ids)"),
            _fetch_junction("SELECT repo_id, login, display_name, org_category, is_known_org FROM repo_builders WHERE repo_id::text = ANY(:ids)"),
            _fetch_junction("SELECT repo_id, dimension, raw_value, similarity_score, assigned_by FROM repo_taxonomy WHERE repo_id::text = ANY(:ids)"),
            _fetch_junction(
                "SELECT repo_id, sha, message, author, committed_at, url FROM repo_commits "
                "WHERE repo_id::text = ANY(:ids) ORDER BY committed_at DESC"
            ),
        )
    )

    all_languages: dict = defaultdict(list)
    for r in lang_rows:
        all_languages[str(r.repo_id)].append({"language": r.language, "bytes": r.bytes, "percentage": r.percentage})

    all_categories: dict = defaultdict(list)
    for r in cat_rows:
        all_categories[str(r.repo_id)].append({"category_name": r.category_name, "is_primary": r.is_primary})

    all_ai_skills: dict = defaultdict(list)
    for r in skill_rows:
        all_ai_skills[str(r.repo_id)].append({"skill": r.skill})

    all_tags: dict = defaultdict(list)
    for r in tag_rows:
        all_tags[str(r.repo_id)].append({"tag": r.tag})

    all_pm_skills: dict = defaultdict(list)
    for r in pm_rows:
        all_pm_skills[str(r.repo_id)].append({"skill": r.skill})

    all_builders: dict = defaultdict(list)
    for r in builder_rows:
        all_builders[str(r.repo_id)].append({
            "login": r.login, "display_name": r.display_name,
            "org_category": r.org_category, "is_known_org": r.is_known_org,
        })

    all_taxonomy: dict = defaultdict(list)
    for r in taxonomy_rows:
        all_taxonomy[str(r.repo_id)].append({
            "dimension": r.dimension,
            "raw_value": r.raw_value,
            "similarity_score": r.similarity_score,
            "assigned_by": r.assigned_by,
        })

    all_commits: dict = defaultdict(list)
    for r in commit_rows:
        all_commits[str(r.repo_id)].append({
            "sha": r.sha,
            "message": r.message,
            "author": r.author,
            "date": r.committed_at.isoformat() if r.committed_at else "",
            "url": r.url or "",
        })

    lifecycle_groups = await _get_lifecycle_groups(db)

    enriched = []
    for repo in repo_dicts:
        rid = str(repo["id"])
        enriched.append(sanitize_repo(_build_enriched_repo(
            repo,
            languages=all_languages.get(rid, []),
            categories=all_categories.get(rid, []),
            ai_skills=all_ai_skills.get(rid, []),
            tags=all_tags.get(rid, []),
            pm_skills=all_pm_skills.get(rid, []),
            builders=all_builders.get(rid, []),
            industries=[],
            lifecycle_groups=lifecycle_groups,
            taxonomy=all_taxonomy.get(rid, []),
            commits=all_commits.get(rid, []),
        )))

    return enriched, total


async def _fetch_aggregates(db: AsyncSession) -> dict:
    """
    Compute library-wide aggregates (stats, categories, tagMetrics, etc.) by loading
    all repos in pages to avoid a single OOM-inducing fetch.

    This runs at most once per CACHE_TTL window — cached under _cache['aggregates'].
    """
    now = time.time()
    cached = _cache.get("aggregates")
    if cached and cached.get("expires_at", 0) > now:
        return cached["data"]

    t0 = time.monotonic()
    all_repos: list[dict] = []
    page = 1
    while True:
        page_repos, total = await _fetch_page_repos(db, page=page, page_size=500)
        all_repos.extend(page_repos)
        if len(all_repos) >= total or not page_repos:
            break
        page += 1

    aggregates = {
        "stats": _build_stats(all_repos),
        "tagMetrics": _build_tag_metrics(all_repos),
        "categories": _build_categories(all_repos),
        "builderStats": _build_builder_stats(all_repos),
        "aiDevSkillStats": _build_ai_dev_skill_stats(all_repos, lifecycle_groups=await _get_lifecycle_groups(db)),
        "pmSkillStats": _build_skill_stats(all_repos, "pmSkills"),
    }
    logger.info(f"Aggregates built in {time.monotonic() - t0:.1f}s across {len(all_repos)} repos")

    _cache["aggregates"] = {"data": aggregates, "expires_at": now + CACHE_TTL}
    return aggregates


@router.get("/library/full", response_model=dict)
async def library_full(
    response: Response,
    db: AsyncSession = Depends(get_db),
    page: int = Query(default=1, ge=1, description="1-based page number"),
    page_size: int = Query(default=200, ge=1, le=500, description="Repos per page (max 500)"),
):
    """
    Returns a paginated page of LibraryData. Aggregates (stats, categories, tagMetrics)
    are included on every page from a separate cache — they reflect the full corpus.

    ?page=1&page_size=200  → first 200 repos
    ?page=2&page_size=200  → next 200 repos

    Junction data (tags, categories, languages, etc.) is fetched only for the current
    page — memory is O(page_size), not O(total). Safe at 10K+ repos.
    """
    cache_key = f"page_{page}_{page_size}"
    redis_key = f"library:page:{page}:size:{page_size}"
    now = time.time()

    response.headers["Cache-Control"] = "public, max-age=300, stale-while-revalidate=3600"

    # 1. Check Redis cache first (shared, survives restarts)
    redis_hit = await redis_cache.get(redis_key)
    if redis_hit is not None:
        logger.info(f"Redis hit /library/full page={page} page_size={page_size}")
        # Warm in-memory cache too so subsequent requests on this instance are instant
        _cache[cache_key] = {"data": redis_hit, "expires_at": now + CACHE_TTL}
        return redis_hit

    # 2. Fall back to in-memory cache (per-instance, zero latency)
    mem_cached = _cache.get(cache_key)
    if mem_cached and mem_cached.get("expires_at", 0) > now:
        logger.info(f"Memory hit /library/full page={page} page_size={page_size}")
        return mem_cached["data"]

    t0 = time.monotonic()
    logger.info(f"Building /library/full page={page} page_size={page_size}...")

    # SECURITY: Only return public repos — is_private=false enforced inside _fetch_page_repos
    enriched_repos, total = await _fetch_page_repos(db, page=page, page_size=page_size)
    aggregates = await _fetch_aggregates(db)

    response = {
        "username": "perditioinc",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "page": page,
        "pageSize": page_size,
        "totalRepos": total,
        "totalPages": (total + page_size - 1) // page_size,
        "repos": enriched_repos,
        "gapAnalysis": {"generatedAt": datetime.now(timezone.utc).isoformat(), "gaps": []},
        **aggregates,
    }

    elapsed = time.monotonic() - t0
    logger.info(f"/library/full page={page} built in {elapsed:.1f}s — {len(enriched_repos)}/{total} repos")

    # Store in both caches
    _cache[cache_key] = {"data": response, "expires_at": now + CACHE_TTL}
    await redis_cache.set(redis_key, response, ttl=CACHE_TTL)
    return response


@router.get("/forks", response_model=dict)
async def list_forks(
    db: AsyncSession = Depends(get_db),
    limit: int = 100,
    offset: int = 0,
):
    """Returns fork repos for internal/intelligence use. Not displayed on reporium.com."""
    result = await db.execute(text("""
        SELECT id, name, owner, forked_from, primary_language, parent_stars, parent_forks,
               readme_summary, problem_solved, behind_by, ahead_by
        FROM repos
        WHERE is_fork = true
        ORDER BY parent_stars DESC NULLS LAST
        LIMIT :limit OFFSET :offset;
    """), {"limit": limit, "offset": offset})
    rows = result.fetchall()
    columns = result.keys()

    count_result = await db.execute(text("SELECT COUNT(*) FROM repos WHERE is_fork = true;"))
    total = count_result.scalar()

    return {
        "forks": [dict(zip(columns, row)) for row in rows],
        "total": total,
        "limit": limit,
        "offset": offset,
    }
