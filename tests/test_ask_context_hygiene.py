"""
KAN-ask-cache: Tests for context hygiene in _build_sources_block.

The sources block sent to Claude must contain ONLY these per-repo fields:
  - name, owner, primary_category, stars, description (truncated to 240 chars)

Excluded for cost/noise reasons:
  - readme_summary, problem_solved
  - language, license_spdx, activity_score
  - has_tests, has_ci, relevance_score
  - forked_from, secondary_category lists
  - embedding arrays
  - created_at / updated_at / last_push_at timestamps
"""
from app.routers.intelligence import _build_sources_block, _SOURCES_DESCRIPTION_MAX


def _mock_repo():
    return {
        "name": "langchain",
        "owner": "langchain-ai",
        "primary_category": "LLM Framework",
        "stars": 95000,
        "description": "Build LLM applications with composable primitives.",
        # Fields that MUST be filtered out:
        "readme_summary": "LangChain is a framework for developing applications powered by language models. It enables applications that are context-aware and reason.",
        "problem_solved": "Orchestrating LLM pipelines and connecting them to external tools.",
        "language": "Python",
        "license_spdx": "MIT",
        "activity_score": 98,
        "has_tests": True,
        "has_ci": True,
        "similarity": 0.9234,
        "forked_from": None,
        "secondary_categories": ["Tooling", "Agents"],
        "embedding_vec": [0.1] * 384,
        "created_at": "2022-10-01T00:00:00Z",
        "updated_at": "2026-03-20T00:00:00Z",
        "integration_tags": ["openai", "anthropic"],
    }


def test_sources_block_includes_required_fields():
    block = _build_sources_block([_mock_repo()])
    assert "langchain" in block
    assert "langchain-ai" in block
    assert "LLM Framework" in block
    assert "95000" in block
    assert "Build LLM applications" in block


def test_sources_block_excludes_secondary_fields():
    block = _build_sources_block([_mock_repo()])
    # Heavy fields that used to live in the prompt
    assert "readme_summary" not in block
    assert "LangChain is a framework for developing" not in block  # full README summary text
    assert "problem_solved" not in block
    assert "Orchestrating LLM pipelines" not in block
    assert "language:" not in block
    assert "Python" not in block
    assert "license:" not in block
    assert "MIT" not in block
    assert "activity_score" not in block
    assert "has_tests" not in block
    assert "has_ci" not in block
    assert "relevance_score" not in block
    assert "0.9234" not in block
    assert "secondary_categories" not in block
    assert "Tooling" not in block
    assert "embedding" not in block
    assert "created_at" not in block
    assert "updated_at" not in block
    assert "2026-03-20" not in block
    assert "integration_tags" not in block


def test_description_truncated_to_240_chars():
    repo = _mock_repo()
    repo["description"] = "x" * 500
    block = _build_sources_block([repo])
    # The truncation is 240 chars on the description field specifically
    assert "x" * _SOURCES_DESCRIPTION_MAX in block
    assert "x" * (_SOURCES_DESCRIPTION_MAX + 1) not in block


def test_empty_repo_list_produces_empty_block():
    assert _build_sources_block([]) == ""


def test_multiple_repos_numbered():
    repos = [_mock_repo(), {**_mock_repo(), "name": "llamaindex", "owner": "run-llama"}]
    block = _build_sources_block(repos)
    assert 'index="1"' in block
    assert 'index="2"' in block
    assert "langchain" in block
    assert "llamaindex" in block


def test_missing_optional_fields_are_omitted():
    # If primary_category or description are None/missing, they just aren't
    # emitted — no "category: None" or "description: None" noise.
    repo = {
        "name": "tiny",
        "owner": "me",
        "stars": 0,
        "primary_category": None,
        "description": None,
    }
    block = _build_sources_block([repo])
    assert "tiny" in block
    assert "None" not in block
    assert "category:" not in block
    assert "description:" not in block
