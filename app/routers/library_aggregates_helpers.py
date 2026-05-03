"""
Shared aggregate builders for /library/full and /library/aggregates (KAN-188).

Per KAN-188 (4h perf audit P2), `/library/full` ships ~1.46 MB warm because the
aggregate fields (stats, gapAnalysis, tagMetrics, categories, builderStats,
aiDevSkillStats, pmSkillStats) are bundled with the per-repo array. This module
extracts the aggregate-building helpers so a lean `/library/aggregates` endpoint
can call them WITHOUT paying the per-repo array cost.

Both endpoints feed `enriched_repos` (output of `_fetch_page_repos` from
library_full) into these helpers. Behaviour is byte-identical to the prior
inline implementations — this is a pure code-organisation refactor.

The constants (`SYSTEM_TAGS`, `KNOWN_ORG_CATEGORIES`, `_AI_DEV_SKILLS_ORDERED`,
`_LIFECYCLE_GROUPS_FALLBACK`, `_SKILL_TAG_TO_GROUP`, `_TAXONOMY_RAW_TO_CANONICAL`)
also live here so the helpers are self-contained. library_full re-exports them
for back-compat with tests/external code that imports the old names.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Constants — moved from library_full.py
# ---------------------------------------------------------------------------

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

_AI_DEV_SKILL_SET: set = set(_AI_DEV_SKILLS_ORDERED)

# Legacy reverse-lookup retained for tag-based matching (enrichedTags / topics).
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


# Backwards-compatible export retained for tests and callers that still import
# the old constant name directly. DB-backed lookup remains the live path.
LIFECYCLE_GROUPS = _LIFECYCLE_GROUPS_FALLBACK


# ---------------------------------------------------------------------------
# Aggregate builders — consume a list of `enriched_repos` (dicts) and return
# the aggregate slice that /library/full embeds and /library/aggregates returns.
# ---------------------------------------------------------------------------


def build_stats(repos: list) -> dict:
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


def build_tag_metrics(repos: list) -> list:
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
            # KAN-193: per-tag `repos: [name1, name2, ...]` array dropped
            # (~70% payload reduction on /library/aggregates: 3.8 MB → ~1 MB).
            # Consumer audit (perditioinc): no production reader of
            # tagMetric.repos in reporium frontend, reporium-mcp, reporium-evals,
            # or reporium-audit. Callers needing tag→repos mapping should
            # derive from per-repo `enrichedTags` in /library/preview or
            # /library/full.
            "avgUpstreamAge": 0,
            "avgTimeSinceForked": 0,
            "mostOutdatedRepo": "",
            "avgBehindBy": 0,
        })

    return metrics


def build_categories(repos: list) -> list:
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
        "Foundation Models": "\U0001F9E0",
        "AI Agents": "\U0001F916",
        "RAG & Retrieval": "\U0001F50D",
        "Model Training": "\U0001F527",
        "Evals & Benchmarking": "\U0001F4CF",
        "Observability & Monitoring": "\U0001F441",
        "Inference & Serving": "⚡",
        "Generative Media": "\U0001F3A8",
        "Computer Vision": "\U0001F4F7",
        "Robotics": "\U0001F9BE",
        "Spatial & XR": "\U0001F97D",
        "MLOps & Infrastructure": "\U0001F680",
        "Dev Tools & Automation": "\U0001F6E0",
        "Cloud & Platforms": "☁️",
        "Learning Resources": "\U0001F4DA",
        "Industry: Healthcare": "\U0001F3E5",
        "Industry: FinTech": "\U0001F4B0",
        "Industry: Audio & Music": "\U0001F3B5",
        "Industry: Gaming": "\U0001F3AE",
        "Security & Safety": "\U0001F512",
        "Data Science & Analytics": "\U0001F4CA",
    }

    categories = []
    for cat, cat_repo_list in sorted(cat_repos.items()):
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
            "icon": ICONS.get(cat, "\U0001F4E6"),
        })

    return categories


def build_skill_stats(repos: list, skill_field: str) -> list:
    """Build SkillStats[] from enriched repos (used for pmSkills)."""
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


def build_ai_dev_skill_stats(repos: list, lifecycle_groups: dict = None) -> list:
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
        for entry in r.get("aiDevSkills", []):
            raw = entry["skill"] if isinstance(entry, dict) else entry
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


def build_builder_stats(repos: list) -> list:
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
    return stats[:50]  # Top 50 builders by repo count


def build_gap_analysis(repos: list) -> dict:
    """Build the gapAnalysis stub.

    Currently returns an empty `gaps` array — actual gap analysis is a future
    feature. The stub matches the shape /library/full has been emitting since
    KAN-151, so consumers (Workato + MCP + eval runner) keep working.
    """
    return {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "gaps": [],
    }
