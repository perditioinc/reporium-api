"""
Golden-set answer quality tests for POST /intelligence/ask.

These tests mock the DB and Anthropic client to verify:
- Response structure is correct (has answer, sources, model, tokens_used)
- Sources are ordered by relevance_score descending
- Answer is non-empty
- Edge case: empty question → 422
- Edge case: no matching repos → answer still returned

A standalone ``client_no_db`` fixture is used so that no real PostgreSQL
connection is required — the FastAPI get_db dependency is overridden inline
before each request and cleared afterward.  The fixture is session-scoped to
avoid re-running the app lifespan (DB connection probe) for every test.
"""
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient


# ---------------------------------------------------------------------------
# Standalone test client fixture (no real DB required)
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture(scope="session")
async def client_no_db():
    """
    Session-scoped AsyncClient that starts the app once without any DB setup.
    The get_db dependency is overridden per-test via dependency_overrides.

    check_db_connection is patched out so that no real PostgreSQL is needed
    and the lifespan finishes immediately.
    """
    from app.main import app
    from app.database import check_db_connection

    with patch("app.main.check_db_connection", new_callable=AsyncMock):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            yield ac


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_db_row(
    repo_id: str,
    name: str,
    owner: str,
    description: str,
    problem_solved: str,
    similarity: float,
    stars: int = 100,
    integration_tags=None,
    dependencies=None,
    forked_from: str | None = None,
    readme_summary: str | None = None,
):
    """Return a MagicMock that mimics a SQLAlchemy row from the repos/embeddings join."""
    row = MagicMock()
    row.id = repo_id
    row.name = name
    row.owner = owner
    row.forked_from = forked_from
    row.description = description
    row.parent_stars = stars
    row.readme_summary = readme_summary or f"Summary for {name}"
    row.problem_solved = problem_solved
    row.integration_tags = integration_tags or []
    row.dependencies = dependencies or []
    row.similarity = similarity
    return row


# Golden-set fixture repos — three repos with known similarity scores (high → low)
GOLDEN_ROWS = [
    _make_db_row(
        repo_id=str(uuid.uuid4()),
        name="langchain",
        owner="langchain-ai",
        forked_from="langchain-ai/langchain",
        description="Build context-aware reasoning applications",
        problem_solved="Orchestrating LLMs with tools and memory",
        similarity=0.9321,
        stars=85000,
        integration_tags=["llm", "rag", "agents"],
    ),
    _make_db_row(
        repo_id=str(uuid.uuid4()),
        name="llama_index",
        owner="run-llama",
        forked_from="run-llama/llama_index",
        description="LlamaIndex (GPT Index) is a data framework for LLM applications",
        problem_solved="Connecting LLMs to external data sources",
        similarity=0.8754,
        stars=32000,
        integration_tags=["rag", "llm", "data"],
    ),
    _make_db_row(
        repo_id=str(uuid.uuid4()),
        name="haystack",
        owner="deepset-ai",
        forked_from="deepset-ai/haystack",
        description="Open-source LLM framework to build production-ready NLP applications",
        problem_solved="End-to-end NLP pipelines with RAG support",
        similarity=0.7102,
        stars=15000,
        integration_tags=["rag", "nlp", "search"],
    ),
]


def _make_anthropic_message(
    answer_text: str, input_tokens: int = 1800, output_tokens: int = 220
):
    """Return a MagicMock mimicking an anthropic.types.Message."""
    content_block = MagicMock()
    content_block.text = answer_text

    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens

    message = MagicMock()
    message.content = [content_block]
    message.usage = usage
    return message


def _make_mock_db(rows):
    """
    Return an AsyncMock db session whose execute() returns the given rows for
    the embedding query and an empty result for the other queries.

    The endpoint makes three DB calls in order:
      1. _find_semantic_cache_hit  → result.first() should be None (cache miss)
      2. pgvector similarity query → result.fetchall() returns the repo rows
      3. knowledge-graph edge query → result.fetchall() returns []
    """
    mock_cache_result = MagicMock()
    mock_cache_result.first.return_value = None  # no semantic cache hit

    mock_result = MagicMock()
    mock_result.fetchall.return_value = rows

    mock_edge_result = MagicMock()
    mock_edge_result.fetchall.return_value = []

    mock_db = AsyncMock()
    # 1st → semantic cache check; 2nd → vector similarity search; 3rd → knowledge-graph edges
    mock_db.execute = AsyncMock(side_effect=[mock_cache_result, mock_result, mock_edge_result])
    return mock_db


CONTROLLED_ANSWER = (
    "Based on the repository data, the best RAG frameworks are LangChain "
    "(langchain-ai/langchain, 85k stars) and LlamaIndex (run-llama/llama_index, 32k stars). "
    "Both provide robust tooling for connecting LLMs to external data sources."
)


# ---------------------------------------------------------------------------
# Shared patch helpers
# ---------------------------------------------------------------------------

def _patch_embedding_model():
    """Patch get_embedding_model() to return a dummy model that encodes to a zero vector."""
    import numpy as np
    mock_model = MagicMock()
    mock_model.encode.return_value = np.zeros(384)
    return patch("app.routers.intelligence.get_embedding_model", return_value=mock_model)


def _patch_anthropic(answer_text: str = CONTROLLED_ANSWER):
    """Patch the Anthropic client constructor to return a controlled answer."""
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _make_anthropic_message(answer_text)
    return patch("anthropic.Anthropic", return_value=mock_client)


def _patch_anthropic_key():
    return patch(
        "app.routers.intelligence.get_anthropic_key",
        return_value="sk-ant-test-key",
    )


def _patch_log_query():
    """Suppress fire-and-forget DB writes inside _log_query."""
    return patch("app.routers.intelligence._log_query", new_callable=AsyncMock)


def _patch_create_task():
    """
    Patch asyncio.create_task inside the intelligence module to a no-op.

    create_task is synchronous and accepts a coroutine.  We replace it with a
    function that closes the coroutine immediately (preventing 'never awaited'
    warnings) and returns a dummy value.
    """
    def _noop_create_task(coro, *args, **kwargs):
        coro.close()  # suppress RuntimeWarning: coroutine was never awaited
        return MagicMock()

    return patch("app.routers.intelligence.asyncio.create_task", side_effect=_noop_create_task)


def _override_db(rows):
    """
    Return (mock_db, override_fn) — override_fn is an async generator suitable
    for FastAPI dependency_overrides[get_db].
    """
    mock_db = _make_mock_db(rows)

    async def _override():
        yield mock_db

    return mock_db, _override


# ---------------------------------------------------------------------------
# Tests — response structure and content
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ask_response_structure(client_no_db: AsyncClient):
    """Response must contain answer, sources, model, tokens_used, question, answered_at."""
    from app.main import app
    from app.database import get_db

    _, override = _override_db(GOLDEN_ROWS)
    app.dependency_overrides[get_db] = override

    with (
        _patch_embedding_model(),
        _patch_anthropic_key(),
        _patch_anthropic(),
        _patch_log_query(),
        _patch_create_task(),
    ):
        try:
            response = await client_no_db.post(
                "/intelligence/ask",
                json={"question": "What are the best RAG frameworks?"},
            )
        finally:
            app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 200, (
        f"Unexpected status: {response.status_code} — {response.text}"
    )
    data = response.json()

    # Required top-level keys
    for key in ("answer", "sources", "model", "tokens_used", "question", "answered_at"):
        assert key in data, f"Response missing required key: '{key}'"

    # tokens_used sub-structure
    tokens = data["tokens_used"]
    assert "input" in tokens
    assert "output" in tokens
    assert "total" in tokens
    assert tokens["total"] == tokens["input"] + tokens["output"]


@pytest.mark.asyncio
async def test_ask_answer_is_non_empty(client_no_db: AsyncClient):
    """The answer field must be a non-empty string."""
    from app.main import app
    from app.database import get_db

    _, override = _override_db(GOLDEN_ROWS)
    app.dependency_overrides[get_db] = override

    with (
        _patch_embedding_model(),
        _patch_anthropic_key(),
        _patch_anthropic(CONTROLLED_ANSWER),
        _patch_log_query(),
        _patch_create_task(),
    ):
        try:
            response = await client_no_db.post(
                "/intelligence/ask",
                json={"question": "What are the best RAG frameworks?"},
            )
        finally:
            app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 200
    answer = response.json()["answer"]
    assert isinstance(answer, str)
    assert len(answer.strip()) > 0, "Answer must not be empty"


@pytest.mark.asyncio
async def test_ask_returns_controlled_answer_text(client_no_db: AsyncClient):
    """The answer text must match exactly what the mocked Anthropic client returns."""
    from app.main import app
    from app.database import get_db

    _, override = _override_db(GOLDEN_ROWS)
    app.dependency_overrides[get_db] = override

    with (
        _patch_embedding_model(),
        _patch_anthropic_key(),
        _patch_anthropic(CONTROLLED_ANSWER),
        _patch_log_query(),
        _patch_create_task(),
    ):
        try:
            response = await client_no_db.post(
                "/intelligence/ask",
                json={"question": "What are the best RAG frameworks?"},
            )
        finally:
            app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 200
    assert response.json()["answer"] == CONTROLLED_ANSWER


# ---------------------------------------------------------------------------
# Tests — source ordering
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ask_sources_ordered_by_relevance_descending(client_no_db: AsyncClient):
    """Sources list must be ordered by relevance_score descending."""
    # Supply rows already in ascending-similarity order so we detect any sorting.
    rows_asc = sorted(GOLDEN_ROWS, key=lambda r: r.similarity)
    _, override = _override_db(rows_asc)

    from app.main import app
    from app.database import get_db

    app.dependency_overrides[get_db] = override

    with (
        _patch_embedding_model(),
        _patch_anthropic_key(),
        _patch_anthropic(),
        _patch_log_query(),
        _patch_create_task(),
    ):
        try:
            response = await client_no_db.post(
                "/intelligence/ask",
                json={"question": "What are the best RAG frameworks?"},
            )
        finally:
            app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 200
    sources = response.json()["sources"]
    assert len(sources) >= 2, "Expected at least 2 sources in response"
    scores = [s["relevance_score"] for s in sources]
    assert scores == sorted(scores, reverse=True), (
        f"Sources must be ordered by relevance_score descending, got: {scores}"
    )


# ---------------------------------------------------------------------------
# Tests — source schema
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ask_source_schema(client_no_db: AsyncClient):
    """Each source must contain the required SourceRepo fields."""
    _, override = _override_db(GOLDEN_ROWS[:1])

    from app.main import app
    from app.database import get_db

    app.dependency_overrides[get_db] = override

    with (
        _patch_embedding_model(),
        _patch_anthropic_key(),
        _patch_anthropic(),
        _patch_log_query(),
        _patch_create_task(),
    ):
        try:
            response = await client_no_db.post(
                "/intelligence/ask",
                json={"question": "What are the best RAG frameworks?"},
            )
        finally:
            app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 200
    source = response.json()["sources"][0]
    for field in ("name", "owner", "relevance_score", "integration_tags"):
        assert field in source, f"Source is missing required field: '{field}'"
    assert isinstance(source["relevance_score"], float)
    assert isinstance(source["integration_tags"], list)


# ---------------------------------------------------------------------------
# Tests — model field
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ask_model_field_references_claude(client_no_db: AsyncClient):
    """The model field must identify the Claude model used."""
    _, override = _override_db(GOLDEN_ROWS[:1])

    from app.main import app
    from app.database import get_db

    app.dependency_overrides[get_db] = override

    with (
        _patch_embedding_model(),
        _patch_anthropic_key(),
        _patch_anthropic(),
        _patch_log_query(),
        _patch_create_task(),
    ):
        try:
            response = await client_no_db.post(
                "/intelligence/ask",
                json={"question": "What are the best RAG frameworks?"},
            )
        finally:
            app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 200
    model_name = response.json()["model"]
    assert "claude" in model_name.lower(), (
        f"model field should reference claude, got: {model_name!r}"
    )


# ---------------------------------------------------------------------------
# Edge case: empty question → 422
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ask_empty_question_returns_422(client_no_db: AsyncClient):
    """POST /intelligence/ask with an empty string question must return 422."""
    response = await client_no_db.post(
        "/intelligence/ask",
        json={"question": ""},
    )
    assert response.status_code == 422, (
        f"Expected 422 for empty question, got {response.status_code}"
    )


# ---------------------------------------------------------------------------
# Edge case: no matching repos → answer still returned
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ask_no_matching_repos_answer_still_returned(client_no_db: AsyncClient):
    """When the DB returns zero repo rows the endpoint must still return 200 with an answer."""
    # Both the embedding query and knowledge-graph query return empty results.
    # first() must return None so _find_semantic_cache_hit sees a cache miss.
    empty_result = MagicMock()
    empty_result.fetchall.return_value = []
    empty_result.first.return_value = None
    mock_db = AsyncMock()
    mock_db.execute = AsyncMock(return_value=empty_result)

    async def _override():
        yield mock_db

    no_data_answer = (
        "The context doesn't contain enough information to answer your question. "
        "No matching repositories were found in the knowledge base."
    )

    from app.main import app
    from app.database import get_db

    app.dependency_overrides[get_db] = _override

    with (
        _patch_embedding_model(),
        _patch_anthropic_key(),
        _patch_anthropic(no_data_answer),
        _patch_log_query(),
        _patch_create_task(),
    ):
        try:
            response = await client_no_db.post(
                "/intelligence/ask",
                json={"question": "What are the best RAG frameworks?"},
            )
        finally:
            app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 200, (
        f"Expected 200 even with no matching repos, got {response.status_code} — {response.text}"
    )
    data = response.json()
    assert data["sources"] == [], "Expected empty sources list when no repos match"
    assert len(data["answer"].strip()) > 0, "Answer must still be present even with no matching repos"
