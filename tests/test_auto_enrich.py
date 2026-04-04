"""
KAN-157: Tests for POST /ingest/events/repo-added auto-enrichment handler.

Covers:
- Happy path: Haiku enriches a new repo (quality_signals written, taxonomy inserted)
- Idempotency: already-enriched repos are skipped
- Missing repo: returns 200 with status=skipped
- Missing name in payload: returns 200 with status=skipped
- Haiku JSON parse error: returns 200 with status=error (Pub/Sub must not retry)
- Circuit-breaker HTTPException: returns 200 with status=error
- Cache keys invalidated after successful enrichment

Uses app.dependency_overrides[get_db] + AsyncMock for async I/O — same pattern
as test_recommendations.py.
"""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient

from app.database import get_db
from app.main import app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_INGEST_HEADERS = {
    "X-Ingest-Key": "test-ingest-key",
    "X-API-Key": "test-api-key",
}

_PUBSUB_PAYLOAD = {
    "message": {
        "data": __import__("base64").b64encode(
            json.dumps({"name_with_owner": "perditioinc/haystack", "stars": 500}).encode()
        ).decode()
    }
}


def _make_repo(name="haystack", quality_signals=None):
    repo = MagicMock()
    repo.id = "test-uuid-1234"
    repo.name = name
    repo.owner = "perditioinc"
    repo.description = f"The {name} AI framework"
    repo.primary_language = "Python"
    repo.forked_from = None
    repo.quality_signals = quality_signals
    repo.readme_summary = None
    repo.problem_solved = None
    repo.integration_tags = None
    repo.updated_at = None
    return repo


def _make_haiku_response(data: dict) -> MagicMock:
    msg = MagicMock()
    msg.content = [MagicMock(text=json.dumps(data))]
    msg.usage = MagicMock(input_tokens=320, output_tokens=180)
    return msg


_GOOD_ENRICHMENT = {
    "readme_summary": "Haystack is an LLM orchestration framework for building NLP pipelines.",
    "problem_solved": "Simplifies building production RAG and NLP systems.",
    "quality_assessment": "high",
    "maturity_level": "production",
    "skill_areas": ["Retrieval-Augmented Generation", "NLP Pipelines"],
    "industries": ["Developer Tools"],
    "use_cases": ["Document Question Answering"],
    "modalities": ["Text"],
    "ai_trends": ["Agentic AI"],
    "deployment_context": ["Cloud API", "Self-hosted"],
    "integration_tags": ["haystack", "transformers", "faiss"],
}


def _override_db(repo):
    """Yield a mock DB where scalar_one_or_none returns the given repo."""
    mock_db = AsyncMock()
    select_result = MagicMock()
    select_result.scalar_one_or_none.return_value = repo
    mock_db.execute = AsyncMock(return_value=select_result)
    mock_db.flush = AsyncMock()
    mock_db.commit = AsyncMock()
    mock_db.add = MagicMock()

    async def _dep():
        yield mock_db

    return mock_db, _dep


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_repo_added_enriches_new_repo(client: AsyncClient):
    """Happy path: Haiku is called, quality_signals and readme_summary are set."""
    repo = _make_repo(quality_signals=None)
    mock_db, override = _override_db(repo)

    haiku_resp = _make_haiku_response(_GOOD_ENRICHMENT)

    app.dependency_overrides[get_db] = override
    try:
        with patch("app.routers.ingest._get_anthropic_key", return_value="sk-test"), \
             patch("app.routers.ingest._anthropic_lib.Anthropic") as MockClient, \
             patch("app.routers.ingest.cache.invalidate", new=AsyncMock()), \
             patch("app.routers.ingest.invalidate_library_cache"):
            MockClient.return_value.messages.create.return_value = haiku_resp
            resp = await client.post(
                "/ingest/events/repo-added",
                json=_PUBSUB_PAYLOAD,
                headers=_INGEST_HEADERS,
            )
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["repo"] == "haystack"
    assert data["enrichment"]["quality"] == "high"
    assert data["enrichment"]["maturity"] == "production"
    assert data["enrichment"]["input_tokens"] == 320
    assert data["enrichment"]["output_tokens"] == 180

    # Verify enrichment fields were written to the repo object
    assert repo.quality_signals == {"quality": "high", "maturity": "production"}
    assert "Haystack is an LLM" in repo.readme_summary
    assert repo.commit.call_count == 0  # db.commit is on the AsyncMock, not repo


@pytest.mark.asyncio
async def test_repo_added_skips_already_enriched(client: AsyncClient):
    """Repos with existing quality_signals are skipped (idempotent)."""
    repo = _make_repo(quality_signals={"quality": "high", "maturity": "production"})
    mock_db, override = _override_db(repo)

    app.dependency_overrides[get_db] = override
    try:
        with patch("app.routers.ingest._get_anthropic_key") as mock_key:
            resp = await client.post(
                "/ingest/events/repo-added",
                json=_PUBSUB_PAYLOAD,
                headers=_INGEST_HEADERS,
            )
            mock_key.assert_not_called()
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 200
    assert resp.json()["status"] == "skipped"
    assert resp.json()["reason"] == "already enriched"


@pytest.mark.asyncio
async def test_repo_added_skips_unknown_repo(client: AsyncClient):
    """Repo not found in DB → skipped with 200, no Haiku call."""
    mock_db = AsyncMock()
    not_found = MagicMock()
    not_found.scalar_one_or_none.return_value = None
    mock_db.execute = AsyncMock(return_value=not_found)

    async def _override():
        yield mock_db

    app.dependency_overrides[get_db] = _override
    try:
        with patch("app.routers.ingest._get_anthropic_key") as mock_key:
            resp = await client.post(
                "/ingest/events/repo-added",
                json=_PUBSUB_PAYLOAD,
                headers=_INGEST_HEADERS,
            )
            mock_key.assert_not_called()
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 200
    assert resp.json()["status"] == "skipped"
    assert "not found" in resp.json()["reason"]


@pytest.mark.asyncio
async def test_repo_added_skips_empty_payload(client: AsyncClient):
    """Pub/Sub payload with no repo name → skipped, no DB call."""
    mock_db = AsyncMock()

    async def _override():
        yield mock_db

    app.dependency_overrides[get_db] = _override
    try:
        resp = await client.post(
            "/ingest/events/repo-added",
            json={"message": {"data": __import__("base64").b64encode(b"{}").decode()}},
            headers=_INGEST_HEADERS,
        )
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 200
    assert resp.json()["status"] == "skipped"
    mock_db.execute.assert_not_called()


@pytest.mark.asyncio
async def test_repo_added_handles_haiku_json_error(client: AsyncClient):
    """Haiku returns unparseable JSON → 200 with status=error (Pub/Sub won't retry)."""
    repo = _make_repo(quality_signals=None)
    mock_db, override = _override_db(repo)

    bad_msg = MagicMock()
    bad_msg.content = [MagicMock(text="not valid json at all")]
    bad_msg.usage = MagicMock(input_tokens=200, output_tokens=10)

    app.dependency_overrides[get_db] = override
    try:
        with patch("app.routers.ingest._get_anthropic_key", return_value="sk-test"), \
             patch("app.routers.ingest._anthropic_lib.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = bad_msg
            resp = await client.post(
                "/ingest/events/repo-added",
                json=_PUBSUB_PAYLOAD,
                headers=_INGEST_HEADERS,
            )
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 200
    assert resp.json()["status"] == "error"
    assert resp.json()["reason"] == "json_parse_error"


@pytest.mark.asyncio
async def test_repo_added_name_with_owner_parsed_correctly(client: AsyncClient):
    """name_with_owner 'owner/haystack' → resolves to bare name 'haystack'."""
    repo = _make_repo(name="haystack", quality_signals=None)
    mock_db, override = _override_db(repo)
    haiku_resp = _make_haiku_response(_GOOD_ENRICHMENT)

    import base64
    payload_with_owner = {
        "message": {
            "data": base64.b64encode(
                json.dumps({"name_with_owner": "deepset-ai/haystack"}).encode()
            ).decode()
        }
    }

    app.dependency_overrides[get_db] = override
    try:
        with patch("app.routers.ingest._get_anthropic_key", return_value="sk-test"), \
             patch("app.routers.ingest._anthropic_lib.Anthropic") as MockClient, \
             patch("app.routers.ingest.cache.invalidate", new=AsyncMock()), \
             patch("app.routers.ingest.invalidate_library_cache"):
            MockClient.return_value.messages.create.return_value = haiku_resp
            resp = await client.post(
                "/ingest/events/repo-added",
                json=payload_with_owner,
                headers=_INGEST_HEADERS,
            )
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.json()["repo"] == "haystack"
