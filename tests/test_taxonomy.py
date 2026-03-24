"""
Taxonomy validation unit tests for the 28-skill taxonomy structure.

Validates:
- 28 skill areas with no duplicates
- 6 lifecycle groups
- Every skill area maps to exactly one lifecycle group
- 58 categories defined
- No single skill area exceeds 15% of repos (requires DB — skipped)
- LIFECYCLE_GROUPS constant in library_full.py (if present)
"""

import importlib

import pytest

# ---------------------------------------------------------------------------
# Inline taxonomy definitions
# ---------------------------------------------------------------------------

SKILL_AREAS = [
    "Foundation Model Architecture",
    "Fine-tuning & Alignment",
    "Data Engineering",
    "Synthetic Data",
    "Inference & Serving",
    "Model Compression",
    "Edge AI",
    "Agents & Orchestration",
    "RAG & Retrieval",
    "Context Engineering",
    "Tool Use",
    "Structured Output",
    "Prompt Engineering",
    "Knowledge Graphs",
    "Evaluation",
    "Security & Guardrails",
    "Observability",
    "MLOps",
    "AI Governance",
    "Computer Vision",
    "Speech & Audio",
    "Generative Media",
    "NLP",
    "Multimodal",
    "Coding Assistants",
    "Robotics",
    "AI for Science",
    "Recommendation Systems",
]

LIFECYCLE_GROUPS = [
    "Foundation & Training",
    "Inference & Deployment",
    "LLM Application Layer",
    "Eval / Safety / Ops",
    "Modality-Specific",
    "Applied AI",
]

# Mapping: skill area -> lifecycle group
SKILL_AREA_TO_GROUP = {
    "Foundation Model Architecture": "Foundation & Training",
    "Fine-tuning & Alignment":       "Foundation & Training",
    "Data Engineering":              "Foundation & Training",
    "Synthetic Data":                "Foundation & Training",
    "Inference & Serving":           "Inference & Deployment",
    "Model Compression":             "Inference & Deployment",
    "Edge AI":                       "Inference & Deployment",
    "Agents & Orchestration":        "LLM Application Layer",
    "RAG & Retrieval":               "LLM Application Layer",
    "Context Engineering":           "LLM Application Layer",
    "Tool Use":                      "LLM Application Layer",
    "Structured Output":             "LLM Application Layer",
    "Prompt Engineering":            "LLM Application Layer",
    "Knowledge Graphs":              "LLM Application Layer",
    "Evaluation":                    "Eval / Safety / Ops",
    "Security & Guardrails":         "Eval / Safety / Ops",
    "Observability":                 "Eval / Safety / Ops",
    "MLOps":                         "Eval / Safety / Ops",
    "AI Governance":                 "Eval / Safety / Ops",
    "Computer Vision":               "Modality-Specific",
    "Speech & Audio":                "Modality-Specific",
    "Generative Media":              "Modality-Specific",
    "NLP":                           "Modality-Specific",
    "Multimodal":                    "Modality-Specific",
    "Coding Assistants":             "Applied AI",
    "Robotics":                      "Applied AI",
    "AI for Science":                "Applied AI",
    "Recommendation Systems":        "Applied AI",
}

CATEGORIES = [
    # Foundation Model Architecture
    "Transformer Architecture",
    "Attention Mechanisms",
    "Pre-training & Scaling",
    # Fine-tuning & Alignment
    "Fine-tuning Methods",
    "RLHF & Alignment",
    "DPO & Preference Learning",
    # Data Engineering
    "Dataset Curation",
    "Data Pipelines",
    # Synthetic Data
    "Synthetic Dataset Generation",
    "Data Augmentation",
    # Inference & Serving
    "Inference Engines",
    "Serving Infrastructure",
    "KV Cache Optimization",
    # Model Compression
    "Quantization",
    "Model Pruning",
    "Knowledge Distillation",
    # Edge AI
    "On-Device Inference",
    "Mobile & Edge ML",
    "WebGPU / WASM Inference",
    # Agents & Orchestration
    "Agent Frameworks",
    "Multi-Agent Systems",
    # RAG & Retrieval
    "RAG Pipelines",
    "Vector Databases",
    "Chunking & Embedding",
    # Context Engineering
    "Memory Systems",
    # Tool Use
    "Function Calling",
    "MCP Servers & Clients",
    "Browser Automation",
    # Structured Output
    "JSON / Structured Extraction",
    # Prompt Engineering
    "Prompt Optimization",
    "Chain-of-Thought",
    # Knowledge Graphs
    "Graph Databases",
    "Entity Extraction",
    # Evaluation
    "Eval Frameworks",
    "Benchmarking",
    "LLM-as-Judge",
    # Security & Guardrails
    "Content Moderation",
    "Red Teaming",
    "Prompt Injection Defense",
    # Observability
    "LLM Tracing & Logging",
    "Cost & Latency Monitoring",
    # MLOps
    "Experiment Tracking",
    "Model Registry",
    "ML CI/CD",
    # AI Governance
    "Bias & Fairness",
    "Model Cards & Compliance",
    # Computer Vision
    "Image Classification",
    "Object Detection",
    "Semantic Segmentation",
    "Vision Transformers",
    # Speech & Audio
    "Speech Recognition (ASR)",
    "Text-to-Speech (TTS)",
    "Audio Generation",
    # Generative Media
    "Diffusion Models",
    "Image Generation",
    "Video Generation",
    # NLP
    "Text Classification",
    "Named Entity Recognition",
]

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_all_28_skill_areas_defined():
    """Assert the skill area list contains exactly 28 items with no duplicates."""
    assert len(SKILL_AREAS) == 28, (
        f"Expected 28 skill areas, got {len(SKILL_AREAS)}"
    )
    assert len(SKILL_AREAS) == len(set(SKILL_AREAS)), (
        "Duplicate skill area names detected: "
        + str([s for s in SKILL_AREAS if SKILL_AREAS.count(s) > 1])
    )


def test_all_6_lifecycle_groups_present():
    """Assert all 6 lifecycle group names are present."""
    assert len(LIFECYCLE_GROUPS) == 6, (
        f"Expected 6 lifecycle groups, got {len(LIFECYCLE_GROUPS)}"
    )
    assert len(LIFECYCLE_GROUPS) == len(set(LIFECYCLE_GROUPS)), (
        "Duplicate lifecycle group names detected"
    )

    expected = {
        "Foundation & Training",
        "Inference & Deployment",
        "LLM Application Layer",
        "Eval / Safety / Ops",
        "Modality-Specific",
        "Applied AI",
    }
    assert set(LIFECYCLE_GROUPS) == expected, (
        f"Lifecycle groups mismatch.\nExpected: {expected}\nGot: {set(LIFECYCLE_GROUPS)}"
    )


def test_skill_areas_map_to_groups():
    """Assert every skill area maps to exactly one lifecycle group."""
    # Every skill area must appear in the mapping
    for skill in SKILL_AREAS:
        assert skill in SKILL_AREA_TO_GROUP, (
            f"Skill area '{skill}' has no lifecycle group mapping"
        )

    # The mapping must not contain skill areas not in SKILL_AREAS
    for skill in SKILL_AREA_TO_GROUP:
        assert skill in SKILL_AREAS, (
            f"Mapping contains unknown skill area '{skill}'"
        )

    # Every mapped group must be a valid lifecycle group
    for skill, group in SKILL_AREA_TO_GROUP.items():
        assert group in LIFECYCLE_GROUPS, (
            f"Skill area '{skill}' maps to unknown group '{group}'"
        )

    # All 6 lifecycle groups must be used
    used_groups = set(SKILL_AREA_TO_GROUP.values())
    assert used_groups == set(LIFECYCLE_GROUPS), (
        f"Not all lifecycle groups are used. Missing: {set(LIFECYCLE_GROUPS) - used_groups}"
    )


def test_58_categories_defined():
    """Assert the category list has exactly 58 items with no duplicates."""
    assert len(CATEGORIES) == 58, (
        f"Expected 58 categories, got {len(CATEGORIES)}"
    )
    assert len(CATEGORIES) == len(set(CATEGORIES)), (
        "Duplicate category names detected: "
        + str([c for c in CATEGORIES if CATEGORIES.count(c) > 1])
    )


def test_no_skill_area_exceeds_15_percent():
    """
    Assert no single skill area accounts for more than 15% of all repos.

    NOTE: This test requires real database data and cannot be run in a unit
    test context. It is skipped here as a placeholder — implement as an
    integration test once a test DB fixture with representative data is
    available.
    """
    pytest.skip(
        "Requires real DB data: connect to a populated database and query "
        "repo counts per skill area to assert each is <= 15% of total."
    )


def test_lifecycle_groups_constant():
    """
    If LIFECYCLE_GROUPS dict exists in app/routers/library_full.py,
    import it and assert it has exactly 28 keys (one per skill area).
    """
    try:
        module = importlib.import_module("app.routers.library_full")
    except ImportError as exc:
        pytest.skip(f"Could not import app.routers.library_full: {exc}")

    if not hasattr(module, "LIFECYCLE_GROUPS"):
        pytest.skip(
            "LIFECYCLE_GROUPS constant not defined in app/routers/library_full.py — "
            "add it to enable this test."
        )

    groups_const = module.LIFECYCLE_GROUPS
    assert isinstance(groups_const, dict), (
        f"Expected LIFECYCLE_GROUPS to be a dict, got {type(groups_const)}"
    )
    assert len(groups_const) == 28, (
        f"Expected LIFECYCLE_GROUPS to have 28 keys (one per skill area), "
        f"got {len(groups_const)}"
    )
