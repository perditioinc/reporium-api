"""Shared utility functions across routers."""
from __future__ import annotations

import difflib
import logging
import os
from typing import TYPE_CHECKING

from app.config import settings

if TYPE_CHECKING:
    import anthropic

logger = logging.getLogger(__name__)
_logger = logging.getLogger(__name__)


def get_anthropic_key() -> str:
    """Get Anthropic API key from env or config."""
    key = os.getenv("ANTHROPIC_API_KEY") or getattr(settings, "anthropic_api_key", None)
    if not key:
        raise ValueError("ANTHROPIC_API_KEY not configured")
    return key


# KAN-197 / Issue #215: Lazy singleton Anthropic client shared across routers.
# Avoids creating a new client per request and keeps a single source of truth.
_anthropic_client: "anthropic.Anthropic | None" = None


def get_anthropic_client() -> "anthropic.Anthropic":
    """Return the process-wide lazy singleton Anthropic client."""
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        _anthropic_client = anthropic.Anthropic(api_key=get_anthropic_key())
    return _anthropic_client


def log_nonfatal(
    operation: str,
    *,
    session_id: str | None = None,
    extra_context: str | None = None,
) -> None:
    """Log a non-fatal exception from a fire-and-forget handler.

    Must be called from within an ``except`` block — uses ``exc_info=True``
    so the active exception traceback is included at WARNING level.
    """
    msg = f"{operation} failed (non-fatal)"
    if session_id:
        msg += f" for session {session_id}"
    if extra_context:
        msg += f" ({extra_context})"
    _logger.warning(msg, exc_info=True)


def vec_to_pg(vec) -> str:
    """Convert a float sequence (list or numpy array) to a pgvector-compatible string."""
    items = vec.tolist() if hasattr(vec, "tolist") else vec
    return "[" + ",".join(f"{x:.8f}" for x in items) + "]"


# ---------------------------------------------------------------------------
# Tag canonicalization — kept in sync with ingestion/enrichment/canonicalize.py
# ---------------------------------------------------------------------------

_CANONICAL_TAGS: frozenset[str] = frozenset({
    "Large Language Models", "OpenAI", "Anthropic / Claude", "Google AI",
    "DeepSeek", "Qwen", "Llama", "Mistral", "Phi", "Gemma", "Claude", "GPT",
    "Reasoning Models",
    "RAG", "Vector Database", "Embeddings", "Semantic Search", "Hybrid Search",
    "Reranking", "Document Processing", "Chunking", "GraphRAG",
    "Chroma", "Qdrant", "Milvus", "Weaviate", "Pinecone", "pgvector",
    "AI Agents", "Multi-Agent", "Agent Memory", "Planning / CoT",
    "Tool Use", "Structured Output", "Context Engineering", "MCP",
    "LangChain", "LangGraph", "LlamaIndex", "CrewAI", "AutoGen",
    "DSPy", "Semantic Kernel", "Haystack", "LiteLLM", "Agno",
    "Letta / MemGPT", "Mem0", "Swarm", "OpenAI Agents SDK",
    "Fine-Tuning", "LoRA / PEFT", "RLHF", "DPO", "GRPO",
    "Synthetic Data", "Distillation", "DeepSpeed", "FSDP", "TRL",
    "Unsloth", "Axolotl", "TorchTune", "MergeKit",
    "LLM Serving", "vLLM", "TGI", "Triton", "TensorRT", "llama.cpp",
    "ExLlama", "GPT4All", "PrivateGPT", "Llamafile", "SGLang",
    "Quantization", "Speculative Decoding", "KV Cache", "Batching",
    "Model Optimization", "Inference",
    "MLOps", "DVC", "ZenML", "Prefect", "Airflow", "Ray", "Kubeflow",
    "Feature Store", "Model Registry", "SageMaker", "Vertex AI",
    "Azure AI", "AWS Bedrock", "AWS", "Google Cloud",
    "Evals", "Benchmarking", "MMLU", "HumanEval", "LM Eval Harness",
    "DeepEval", "RAGAS", "PromptFoo", "Red Teaming", "Garak", "PyRIT",
    "LangSmith", "Phoenix", "MLflow", "Weights & Biases", "Tracing",
    "Monitoring", "Langfuse", "OpenLLMetry", "OpenLIT", "Helicone",
    "Traceloop", "OpenTelemetry",
    "Computer Vision", "Object Detection", "Segmentation", "Depth Estimation",
    "Pose Estimation", "3D Reconstruction", "Point Cloud / 3D Vision",
    "Robotics", "ROS", "ROS 2", "Motion Planning", "Grasping",
    "Humanoid Robotics", "Robot Arms", "Robot Learning", "Sim-to-Real", "SLAM",
    "Autonomous Systems",
    "Image Generation", "Video Generation", "Text to Speech", "Speech to Text",
    "Music / Audio AI", "Music Generation", "Voice Cloning",
    "Stable Diffusion", "ControlNet", "ComfyUI", "SD WebUI", "Whisper",
    "Multimodal AI", "XR / Spatial Computing", "Virtual Reality",
    "Augmented Reality", "Mixed Reality", "Immersive Media",
    "WebXR", "ARKit", "ARCore", "Meta Quest", "Apple Vision", "Apple Vision Pro",
    "Python", "TypeScript", "JavaScript", "Rust", "Go", "Java", "C++",
    "Backend", "Frontend", "Full Stack", "Systems",
    "React / Next.js", "Python Web Framework", "Node.js",
    "Docker", "Kubernetes", "DevOps", "API", "GraphQL",
    "Database", "Caching",
    "Pydantic", "Instructor", "Outlines", "Guidance", "Guardrails",
    "NeMo Guardrails", "Prompt Engineering",
    "Machine Learning", "Deep Learning", "Transformers", "PyTorch",
    "TensorFlow", "Keras", "JAX", "GPU / CUDA",
    "Reinforcement Learning", "Long Context",
    "Data Science", "Pandas", "Jupyter", "Data Visualization", "NumPy",
    "Scikit-learn", "Spark", "Data Engineering", "Statistics", "Visualization",
    "Continue.dev", "Aider", "SWE-Agent", "OpenDevin", "OpenHands",
    "Cline", "Claude Code", "Gemini CLI", "Kilocode",
    "Langflow", "Flowise", "n8n", "No-Code Automation", "Automation",
    "Tutorial", "Course", "Roadmap", "Cheat Sheet", "Curated List",
    "Interview Prep", "Research / Papers", "Open Source",
    "FinTech", "Healthcare AI", "Music Tech", "Game Dev",
    "Security", "Web3", "Mobile", "Knowledge Graph", "Real-Time / Streaming",
    "AI Safety", "Adversarial", "Watermarking", "Privacy",
    "Privacy-Preserving AI", "Prompt Injection",
    "CLI Tool", "Simulation", "HuggingFace", "Ollama",
    "ONNX", "Popular", "Active", "Inactive", "Forked", "Built by Me",
    "Archived", "NLP", "Frontend Framework", "Testing",
})

_lower_to_canonical: dict[str, str] = {t.lower(): t for t in _CANONICAL_TAGS}
_vocab_list: list[str] = sorted(_CANONICAL_TAGS)
_vocab_lower: list[str] = [t.lower() for t in _vocab_list]


def canonicalize_tag(raw: str) -> str | None:
    """Return canonical form of raw tag, or None if no match at threshold 0.82."""
    if not raw or not isinstance(raw, str):
        return None
    clean = raw.strip()
    if not clean:
        return None
    lower = clean.lower()
    if lower in _lower_to_canonical:
        return _lower_to_canonical[lower]
    matches = difflib.get_close_matches(lower, _vocab_lower, n=1, cutoff=0.82)
    if matches:
        return _vocab_list[_vocab_lower.index(matches[0])]
    return None


def canonicalize_tags(tags: list[str]) -> list[str]:
    """Canonicalize a list of raw tags. Returns unique canonical forms; drops unmatched."""
    seen: set[str] = set()
    result: list[str] = []
    for raw in tags:
        canonical = canonicalize_tag(raw)
        if canonical and canonical not in seen:
            seen.add(canonical)
            result.append(canonical)
    return result
