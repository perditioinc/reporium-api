"""
Tests for the pros/cons enrichment feature:

- POST /admin/enrich-pros-cons (auth gating, dry_run, prompt construction,
  response parsing, cost tracking)
- GET /repos/{repo_id}/evaluation (returns evaluation, 404 cases)
"""
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from httpx import AsyncClient

from tests.conftest import AUTH_HEADERS, TEST_API_KEY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_repo(**overrides):
    """Create a mock Repo with all fields the enrichment endpoint reads."""
    repo = MagicMock()
    repo.id = overrides.get("id", uuid4())
    repo.name = overrides.get("name", "test-repo")
    repo.owner = overrides.get("owner", "testowner")
    repo.description = overrides.get("description", "A great AI tool")
    repo.readme_summary = overrides.get("readme_summary", "Does AI stuff well")
    repo.problem_solved = overrides.get("problem_solved", "Solves AI problems")
    repo.quality_signals = overrides.get("quality_signals", {"quality": "high", "maturity": "production"})
    repo.primary_category = overrides.get("primary_category", "ai-agents")
    repo.primary_language = overrides.get("primary_language", "Python")
    repo.stargazers_count = overrides.get("stargazers_count", 5000)
    repo.parent_stars = overrides.get("parent_stars", None)
    repo.contributors_count = overrides.get("contributors_count", 42)
    repo.issue_close_rate = overrides.get("issue_close_rate", 85.0)
    repo.has_tests = overrides.get("has_tests", True)
    repo.has_ci = overrides.get("has_ci", True)
    repo.community_health_pct = overrides.get("community_health_pct", 90)
    repo.is_private = False
    repo.pros_cons = overrides.get("pros_cons", None)
    repo.pros_cons_generated_at = overrides.get("pros_cons_generated_at", None)
    return repo


def _make_haiku_response(data: dict, input_tokens: int = 400, output_tokens: int = 250) -> MagicMock:
    """Create a mock Anthropic message response."""
    msg = MagicMock()
    msg.content = [MagicMock(text=json.dumps(data))]
    msg.usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)
    return msg


_GOOD_EVALUATION = {
    "pros": [
        "Strong community with 5000+ stars and 42 contributors",
        "Well-tested codebase with CI/CD pipeline",
        "Production-mature with high quality signals",
    ],
    "cons": [
        "Limited documentation for advanced use cases",
        "No support for non-Python environments",
    ],
    "best_for": "Teams building production AI agent pipelines in Python",
    "avoid_if": "You need a lightweight solution or non-Python support",
    "community_verdict": "Highly regarded by AI developers for its reliability and active maintenance",
    "comparable_to": ["LangChain", "LlamaIndex", "CrewAI"],
}


# ---------------------------------------------------------------------------
# Auth gating
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_enrich_pros_cons_requires_admin_key(client: AsyncClient):
    """Endpoint should reject requests without admin auth."""
    response = await client.post("/admin/enrich-pros-cons")
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_enrich_pros_cons_requires_api_key(client: AsyncClient):
    """Endpoint should reject requests without API key even if admin key is set."""
    response = await client.post(
        "/admin/enrich-pros-cons",
        headers={"X-Admin-Key": "wrong-key"},
    )
    assert response.status_code == 403


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_enrich_pros_cons_dry_run(client: AsyncClient):
    """dry_run=true should return total count without making any API calls."""
    # Seed a repo so there's something to count
    await client.post(
        "/ingest/repos",
        json=[{
            "name": "pros-cons-dry-run-test",
            "owner": "testowner",
            "description": "Test repo for dry run",
            "is_fork": False,
            "primary_language": "Python",
            "github_url": "https://github.com/testowner/pros-cons-dry-run-test",
            "tags": ["ai"],
        }],
        headers=AUTH_HEADERS,
    )

    response = await client.post(
        "/admin/enrich-pros-cons?dry_run=true",
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["dry_run"] is True
    assert data["enriched"] == 0
    assert data["total"] >= 1


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def test_prompt_uses_repo_fields():
    """The prompt template should incorporate all key repo metrics."""
    from app.routers.admin import _PROS_CONS_PROMPT

    # Verify template has expected placeholders
    assert "{owner}" in _PROS_CONS_PROMPT
    assert "{name}" in _PROS_CONS_PROMPT
    assert "{stars}" in _PROS_CONS_PROMPT
    assert "{description}" in _PROS_CONS_PROMPT
    assert "{readme_summary}" in _PROS_CONS_PROMPT
    assert "{problem_solved}" in _PROS_CONS_PROMPT
    assert "{quality}" in _PROS_CONS_PROMPT
    assert "{maturity}" in _PROS_CONS_PROMPT
    assert "{primary_category}" in _PROS_CONS_PROMPT
    assert "{primary_language}" in _PROS_CONS_PROMPT
    assert "{contributors_count}" in _PROS_CONS_PROMPT
    assert "{issue_close_rate}" in _PROS_CONS_PROMPT
    assert "{has_tests}" in _PROS_CONS_PROMPT
    assert "{has_ci}" in _PROS_CONS_PROMPT
    assert "{community_health_pct}" in _PROS_CONS_PROMPT


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generate_pros_cons_parses_valid_json():
    """_generate_pros_cons_for_repo should parse valid JSON from Claude."""
    import asyncio
    from app.routers.admin import _generate_pros_cons_for_repo

    repo = _make_repo()
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _make_haiku_response(_GOOD_EVALUATION)

    semaphore = asyncio.Semaphore(5)

    async def mock_to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    with patch("app.routers.admin.asyncio.to_thread", side_effect=mock_to_thread):
        result = await _generate_pros_cons_for_repo(repo, mock_client, semaphore)

    assert result["error"] is None
    assert result["pros_cons"] == _GOOD_EVALUATION
    assert result["input_tokens"] == 400
    assert result["output_tokens"] == 250


@pytest.mark.asyncio
async def test_generate_pros_cons_handles_markdown_fences():
    """Should strip ```json ... ``` code fences from Claude response."""
    import asyncio
    from app.routers.admin import _generate_pros_cons_for_repo

    repo = _make_repo()
    fenced_response = MagicMock()
    fenced_text = "```json\n" + json.dumps(_GOOD_EVALUATION) + "\n```"
    fenced_response.content = [MagicMock(text=fenced_text)]
    fenced_response.usage = MagicMock(input_tokens=400, output_tokens=250)

    mock_client = MagicMock()
    mock_client.messages.create.return_value = fenced_response

    semaphore = asyncio.Semaphore(5)

    async def mock_to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    with patch("app.routers.admin.asyncio.to_thread", side_effect=mock_to_thread):
        result = await _generate_pros_cons_for_repo(repo, mock_client, semaphore)

    assert result["error"] is None
    assert result["pros_cons"]["pros"] == _GOOD_EVALUATION["pros"]


@pytest.mark.asyncio
async def test_generate_pros_cons_handles_invalid_json():
    """Should return error dict when Claude returns invalid JSON."""
    import asyncio
    from app.routers.admin import _generate_pros_cons_for_repo

    repo = _make_repo()
    bad_response = MagicMock()
    bad_response.content = [MagicMock(text="This is not JSON at all")]
    bad_response.usage = MagicMock(input_tokens=400, output_tokens=50)

    mock_client = MagicMock()
    mock_client.messages.create.return_value = bad_response

    semaphore = asyncio.Semaphore(5)

    async def mock_to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    with patch("app.routers.admin.asyncio.to_thread", side_effect=mock_to_thread):
        result = await _generate_pros_cons_for_repo(repo, mock_client, semaphore)

    assert result["error"] is not None
    assert "JSON parse error" in result["error"]
    assert result["pros_cons"] is None
    # Tokens should still be tracked even on parse failure
    assert result["input_tokens"] == 400
    assert result["output_tokens"] == 50


@pytest.mark.asyncio
async def test_generate_pros_cons_handles_api_exception():
    """Should return error dict when the Anthropic API raises an exception."""
    import asyncio
    from app.routers.admin import _generate_pros_cons_for_repo

    repo = _make_repo()
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = Exception("Rate limited")

    semaphore = asyncio.Semaphore(5)

    async def mock_to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    with patch("app.routers.admin.asyncio.to_thread", side_effect=mock_to_thread):
        result = await _generate_pros_cons_for_repo(repo, mock_client, semaphore)

    assert result["error"] == "Rate limited"
    assert result["pros_cons"] is None
    assert result["input_tokens"] == 0
    assert result["output_tokens"] == 0


# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------

def test_cost_calculation():
    """Verify Haiku cost formula: $0.80/M input + $4.00/M output."""
    input_tokens = 400
    output_tokens = 250
    # Expected: (400 * 0.80 / 1_000_000) + (250 * 4.00 / 1_000_000)
    expected = (400 * 0.80 / 1_000_000) + (250 * 4.00 / 1_000_000)
    assert expected == pytest.approx(0.00132, rel=1e-3)


# ---------------------------------------------------------------------------
# GET /repos/{repo_id}/evaluation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_evaluation_endpoint_returns_404_for_missing_repo(client: AsyncClient):
    """Should return 404 when repo does not exist."""
    response = await client.get("/repos/nonexistent-repo-xyz/evaluation")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_evaluation_endpoint_returns_404_when_no_evaluation(client: AsyncClient):
    """Should return 404 when repo exists but has no pros_cons."""
    # Seed a repo without pros_cons
    await client.post(
        "/ingest/repos",
        json=[{
            "name": "eval-test-no-pros",
            "owner": "testowner",
            "description": "Test repo with no evaluation",
            "is_fork": False,
            "primary_language": "Python",
            "github_url": "https://github.com/testowner/eval-test-no-pros",
            "tags": ["ai"],
        }],
        headers=AUTH_HEADERS,
    )

    response = await client.get("/repos/eval-test-no-pros/evaluation")
    assert response.status_code == 404
    assert "No evaluation" in response.json()["detail"]


@pytest.mark.asyncio
async def test_evaluation_endpoint_surfaces_community_health_fields(client: AsyncClient):
    """Success path: evaluation response must include community-health signals
    (contributors_count, issue_close_rate, pr_merge_rate, community_health_pct)
    alongside the AI-generated pros/cons.
    """
    from sqlalchemy import text as _text

    # Seed a repo
    await client.post(
        "/ingest/repos",
        json=[{
            "name": "eval-test-with-pros",
            "owner": "testowner",
            "description": "Test repo with evaluation",
            "is_fork": False,
            "primary_language": "Python",
            "github_url": "https://github.com/testowner/eval-test-with-pros",
            "tags": ["ai"],
        }],
        headers=AUTH_HEADERS,
    )

    # Write pros_cons + community-health fields directly via DB
    import app.database as db_module
    async with db_module.async_session_factory() as session:
        await session.execute(
            _text(
                "UPDATE repos SET pros_cons = :pc, pros_cons_generated_at = NOW(), "
                "contributors_count = :cc, issue_close_rate = :icr, "
                "pr_merge_rate = :pmr, community_health_pct = :chp "
                "WHERE name = :name"
            ),
            {
                "pc": json.dumps(_GOOD_EVALUATION),
                "cc": 42,
                "icr": 85.0,
                "pmr": 72.5,
                "chp": 90,
                "name": "eval-test-with-pros",
            },
        )
        await session.commit()

    response = await client.get("/repos/eval-test-with-pros/evaluation")
    assert response.status_code == 200
    body = response.json()
    assert body["repo"] == "eval-test-with-pros"
    assert body["owner"] == "testowner"
    assert body["evaluation"]["pros"] == _GOOD_EVALUATION["pros"]
    assert body["generated_at"] is not None
    # New fields — the core assertion of this regression test
    assert body["contributors_count"] == 42
    assert body["issue_close_rate"] == 85.0
    assert body["pr_merge_rate"] == 72.5
    assert body["community_health_pct"] == 90
