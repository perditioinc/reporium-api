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
    row.primary_category = None
    row.language = None
    row.license_spdx = None
    row.activity_score = None
    row.has_tests = None
    row.has_ci = None
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
    Return an AsyncMock db session whose execute() routes by SQL shape:
      - semantic-cache lookup (``query_log`` + ``question_embedding_vec``)
        → result.first() returns None (cache miss)
      - pgvector similarity (``repo_embeddings`` JOIN ``repos``)
        → result.fetchall() returns the repo rows
      - knowledge-graph edges (``repo_edges``) → result.fetchall() returns []

    KAN-182: previously this used an ordered ``side_effect=[cache, rows, edges]``
    list. KAN-169 (PR #467) made the semantic-cache lookup conditional — for
    negation queries (``alternatives to <token>``) ``_prepare_query`` skips
    ``_find_semantic_cache_hit`` entirely, so the call count drops from 3 to 2
    and the ordered list returns ``cache`` (a None-row mock) when the code
    actually wanted ``rows`` — surfacing as an empty source list and silent
    test breakage. Routing by SQL shape is robust to call-count changes.
    """
    mock_cache_result = MagicMock()
    mock_cache_result.first.return_value = None  # no semantic cache hit

    mock_result = MagicMock()
    mock_result.fetchall.return_value = rows

    mock_edge_result = MagicMock()
    mock_edge_result.fetchall.return_value = []

    # Default for any unrecognized query — empty result, fail-soft.
    mock_default = MagicMock()
    mock_default.first.return_value = None
    mock_default.fetchall.return_value = []

    async def _route(stmt, *args, **kwargs):
        # ``stmt`` is a SQLAlchemy ``TextClause``; str() yields the raw SQL.
        sql = str(stmt).lower()
        if "query_log" in sql and "question_embedding_vec" in sql:
            return mock_cache_result
        if "repo_embeddings" in sql and "repos" in sql:
            return mock_result
        if "repo_edges" in sql:
            return mock_edge_result
        return mock_default

    mock_db = AsyncMock()
    mock_db.execute = AsyncMock(side_effect=_route)
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
    """Patch _get_client() to return a mock client with a controlled answer."""
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _make_anthropic_message(answer_text)
    return patch("app.routers.intelligence._get_client", return_value=mock_client)


def _patch_anthropic_key():
    """No-op kept for backward compat — _get_client bypasses get_anthropic_key."""
    return patch(
        "app.routers.intelligence.get_anthropic_key",
        return_value="sk-ant-test-key",
        create=True,
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


# ---------------------------------------------------------------------------
# KAN-166: /ask negation post-filter — close P1 #365 self-match
#
# When a user asks "alternatives to X", the retrieval layer can return X
# itself as the top match. PR #439 added the `forbidden_repos` golden-set
# primitive; KAN-166 adds the structural enforcement on the handler side.
# ---------------------------------------------------------------------------

# --- Module-level helper unit tests (no DB / no HTTP) ----------------------

def test_extract_negation_token_alternatives_to():
    from app.routers.intelligence import _extract_negation_token
    assert _extract_negation_token("alternatives to pinecone") == "pinecone"
    assert _extract_negation_token("vector database alternatives to pinecone") == "pinecone"
    assert _extract_negation_token("Alternatives To Pinecone") == "pinecone"


def test_extract_negation_token_similar_to():
    from app.routers.intelligence import _extract_negation_token
    assert _extract_negation_token("repos similar to langchain") == "langchain"
    assert _extract_negation_token("tools comparable to weaviate") == "weaviate"


def test_extract_negation_token_instead_of():
    from app.routers.intelligence import _extract_negation_token
    assert _extract_negation_token("a vector db instead of pinecone") == "pinecone"


def test_extract_negation_token_owner_slash_name_normalized():
    from app.routers.intelligence import _extract_negation_token
    # "alternatives to pinecone-io/pinecone" should normalize to the repo name
    assert _extract_negation_token("alternatives to pinecone-io/pinecone") == "pinecone"


def test_extract_negation_token_short_token_returns_none():
    """Short tokens (< 3 chars) like "ai" or "ml" must NOT trigger the filter."""
    from app.routers.intelligence import _extract_negation_token
    assert _extract_negation_token("alternatives to ai") is None
    assert _extract_negation_token("alternatives to ml") is None


def test_extract_negation_token_no_negation_returns_none():
    from app.routers.intelligence import _extract_negation_token
    assert _extract_negation_token("What is PyTorch used for?") is None
    assert _extract_negation_token("show me python libraries") is None
    assert _extract_negation_token("") is None
    assert _extract_negation_token(None) is None  # type: ignore[arg-type]


def test_source_matches_negated_token_pinecone_variants():
    from app.routers.intelligence import _source_matches_negated_token
    # Direct name match
    assert _source_matches_negated_token(
        {"name": "pinecone", "owner": "pinecone-io", "forked_from": None},
        "pinecone",
    ) is True
    # Vendor-prefixed name (the canonical #365 case)
    assert _source_matches_negated_token(
        {"name": "pinecone-python-client", "owner": "pinecone-io", "forked_from": None},
        "pinecone",
    ) is True


def test_source_matches_negated_token_does_not_overmatch():
    """Word-boundary alignment — token "ai" inside "openai" should NOT match.

    (Belt-and-suspenders: short-token guard already excludes 'ai', but the
    matcher itself should also be conservative so that future relaxations of
    the min-length cap don't accidentally over-filter.)
    """
    from app.routers.intelligence import _source_matches_negated_token
    # "ai" is below MIN_LEN, but even if forced through, "openai" should not
    # match because of word boundaries — assert via a longer non-matching pair.
    assert _source_matches_negated_token(
        {"name": "weaviate", "owner": "weaviate", "forked_from": None},
        "pinecone",
    ) is False
    assert _source_matches_negated_token(
        {"name": "qdrant", "owner": "qdrant", "forked_from": None},
        "pinecone",
    ) is False


def test_apply_negation_filter_no_token_is_noop():
    from app.routers.intelligence import _apply_negation_filter
    sources = [{"name": "x", "owner": "y", "forked_from": None}]
    assert _apply_negation_filter(sources, None) == sources
    assert _apply_negation_filter(sources, "") == sources


def test_apply_negation_filter_drops_self_match_keeps_others():
    from app.routers.intelligence import _apply_negation_filter
    sources = [
        {"name": "pinecone-python-client", "owner": "pinecone-io", "forked_from": None},
        {"name": "weaviate", "owner": "weaviate", "forked_from": None},
        {"name": "qdrant", "owner": "qdrant", "forked_from": None},
    ]
    kept = _apply_negation_filter(sources, "pinecone")
    kept_names = [s["name"] for s in kept]
    assert "pinecone-python-client" not in kept_names
    assert "weaviate" in kept_names
    assert "qdrant" in kept_names


# --- End-to-end /ask handler tests with mocked DB rows ---------------------

# Negation-filter fixture rows: pinecone (the to-be-dropped) + 3 alternatives.
NEGATION_ROWS = [
    _make_db_row(
        repo_id=str(uuid.uuid4()),
        name="pinecone-python-client",
        owner="pinecone-io",
        forked_from="pinecone-io/pinecone-python-client",
        description="Official Python client for Pinecone vector database",
        problem_solved="Vector storage and similarity search via managed service",
        similarity=0.95,
        stars=2500,
        integration_tags=["vector-db", "managed", "saas"],
    ),
    _make_db_row(
        repo_id=str(uuid.uuid4()),
        name="weaviate",
        owner="weaviate",
        forked_from="weaviate/weaviate",
        description="Open-source vector database",
        problem_solved="Self-hosted vector DB with hybrid search",
        similarity=0.88,
        stars=12000,
        integration_tags=["vector-db", "open-source"],
    ),
    _make_db_row(
        repo_id=str(uuid.uuid4()),
        name="qdrant",
        owner="qdrant",
        forked_from="qdrant/qdrant",
        description="High-performance vector similarity search engine",
        problem_solved="Rust-based vector DB with payload filtering",
        similarity=0.86,
        stars=20000,
        integration_tags=["vector-db", "rust"],
    ),
    _make_db_row(
        repo_id=str(uuid.uuid4()),
        name="chroma",
        owner="chroma-core",
        forked_from="chroma-core/chroma",
        description="AI-native embedding database",
        problem_solved="Lightweight local vector store for prototyping RAG apps",
        similarity=0.84,
        stars=17000,
        integration_tags=["vector-db", "embeddings"],
    ),
]


@pytest.mark.asyncio
async def test_ask_negation_drops_self_match(client_no_db: AsyncClient):
    """KAN-166: 'alternatives to pinecone' must NOT return pinecone in sources."""
    from app.main import app
    from app.database import get_db

    _, override = _override_db(NEGATION_ROWS)
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
                json={"question": "vector database alternatives to pinecone"},
            )
        finally:
            app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 200, response.text
    sources = response.json()["sources"]
    # Hard assertion: NO source whose name lowercased contains "pinecone".
    for s in sources:
        full = f"{(s.get('owner') or '').lower()}/{(s.get('name') or '').lower()}"
        assert "pinecone" not in full, (
            f"KAN-166 regression: pinecone leaked into sources as {full!r}"
        )


@pytest.mark.asyncio
async def test_ask_negation_preserves_non_match(client_no_db: AsyncClient):
    """KAN-166: vector-db alternatives that don't match the negated token stay."""
    from app.main import app
    from app.database import get_db

    _, override = _override_db(NEGATION_ROWS)
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
                json={"question": "vector database alternatives to pinecone"},
            )
        finally:
            app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 200, response.text
    source_names = {(s.get("name") or "").lower() for s in response.json()["sources"]}
    # Three alternatives must survive — they're the actual answer.
    assert "weaviate" in source_names
    assert "qdrant" in source_names
    assert "chroma" in source_names


@pytest.mark.asyncio
async def test_ask_negation_short_token_skipped(client_no_db: AsyncClient):
    """KAN-166: 'alternatives to ai' has a short token — filter must NOT fire."""
    from app.main import app
    from app.database import get_db

    # Use a fixture that contains "ai" as a substring of multiple repos so we
    # can prove they all stay (filter is a no-op for sub-3-char tokens).
    ai_rows = [
        _make_db_row(
            repo_id=str(uuid.uuid4()),
            name="openai-cookbook",
            owner="openai",
            forked_from="openai/openai-cookbook",
            description="Examples and guides for using the OpenAI API",
            problem_solved="Reference implementations for LLM workflows",
            similarity=0.91,
            stars=55000,
            integration_tags=["llm", "examples"],
        ),
        _make_db_row(
            repo_id=str(uuid.uuid4()),
            name="langchain",
            owner="langchain-ai",
            forked_from="langchain-ai/langchain",
            description="Build context-aware reasoning applications",
            problem_solved="Orchestrating LLMs with tools and memory",
            similarity=0.88,
            stars=85000,
            integration_tags=["llm", "agents"],
        ),
    ]
    _, override = _override_db(ai_rows)
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
                json={"question": "alternatives to ai"},
            )
        finally:
            app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 200, response.text
    source_names = {(s.get("name") or "").lower() for s in response.json()["sources"]}
    # The short-token guard skipped the filter — both repos remain.
    assert "openai-cookbook" in source_names
    assert "langchain" in source_names


@pytest.mark.asyncio
async def test_ask_no_negation_no_filter(client_no_db: AsyncClient):
    """KAN-166: a non-negation query keeps every retrieved source intact."""
    from app.main import app
    from app.database import get_db

    _, override = _override_db(NEGATION_ROWS)
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
                json={"question": "What are the best vector databases?"},
            )
        finally:
            app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 200, response.text
    source_names = {(s.get("name") or "").lower() for s in response.json()["sources"]}
    # No negation phrase in the question → pinecone must remain in sources.
    assert "pinecone-python-client" in source_names
    # And the others stay too.
    assert "weaviate" in source_names


# ---------------------------------------------------------------------------
# KAN-169: cache-hit gap from KAN-166 — negation queries must bypass cache
#
# KAN-166 added a post-filter that drops "alternatives to X" self-matches at
# retrieval time. But the cache layers (Redis fast-path + pgvector semantic
# cache) sit BEFORE the post-filter, so an entry written before the fix (or
# under a query variant the regex didn't catch) can still surface a stale
# self-match on HIT for up to TTL=1800s.
#
# Approach A (chosen): when _extract_negation_token captures a token, skip
# both cache lookups in _prepare_query AND skip both cache writes in the
# handler. Negation queries are rare; the cache-miss cost is well worth the
# correctness guarantee.
# ---------------------------------------------------------------------------


def _patch_cache(llm_response_returns=None):
    """
    Patch app.routers.intelligence.cache.get to return ``llm_response_returns``
    when called with a key matching the ``llm_response:`` prefix (the Redis
    fast-path used by _prepare_query for the negation-bypass test). All other
    cache namespaces (smart_route, graph_edges, etc.) get None — otherwise the
    same stale payload would be misinterpreted as a smart-route result and
    short-circuit before our negation gate even runs.

    cache.set is replaced with a tracking AsyncMock so the test can assert it
    is NOT called for negation queries (proves write-side bypass).

    Uses ``new=AsyncMock(...)`` rather than ``side_effect`` so the patched
    attribute is awaitable — a plain MagicMock would hang on ``await``.
    """
    from unittest.mock import AsyncMock

    async def _get_side_effect(key):
        if key and isinstance(key, str) and key.startswith("llm_response:"):
            return llm_response_returns
        return None

    get_mock = AsyncMock(side_effect=_get_side_effect)
    set_mock = AsyncMock()

    cache_get_patch = patch(
        "app.routers.intelligence.cache.get",
        new=get_mock,
    )
    cache_set_patch = patch(
        "app.routers.intelligence.cache.set",
        new=set_mock,
    )
    return cache_get_patch, cache_set_patch, set_mock


@pytest.mark.asyncio
async def test_ask_negation_bypasses_cache(client_no_db: AsyncClient):
    """
    KAN-169: a negation query ("alternatives to pinecone") must bypass the
    Redis cache even if a stale entry from before KAN-166 still has pinecone
    in its sources. We pre-populate cache.get with that stale payload and
    verify the live response is filtered AND that cache.set is NOT called
    for this query (so the bypass is symmetric — no future calls inherit
    a fresh self-match either).
    """
    from app.main import app
    from app.database import get_db

    _, override = _override_db(NEGATION_ROWS)
    app.dependency_overrides[get_db] = override

    # Stale cache entry: pre-fix payload that includes pinecone in sources.
    # If the bypass works, this should NEVER appear in the response.
    stale_payload = {
        "answer": "STALE: Pinecone is the best vector database!",
        "sources": [{
            "name": "pinecone-python-client",
            "owner": "pinecone-io",
            "forked_from": "pinecone-io/pinecone-python-client",
            "relevance_score": 0.95,
            "integration_tags": ["vector-db", "managed"],
        }],
        "tokens_used": {"input": 0, "output": 0, "total": 0},
        "model": "stale-cache",
    }

    cache_get_patch, cache_set_patch, set_mock = _patch_cache(llm_response_returns=stale_payload)

    with (
        _patch_embedding_model(),
        _patch_anthropic_key(),
        _patch_anthropic(),
        _patch_log_query(),
        _patch_create_task(),
        cache_get_patch,
        cache_set_patch,
    ):
        try:
            response = await client_no_db.post(
                "/intelligence/ask",
                json={"question": "vector database alternatives to pinecone"},
            )
        finally:
            app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 200, response.text
    data = response.json()
    # Stale cache answer must NOT have leaked through.
    assert "STALE" not in data["answer"], (
        "KAN-169 regression: stale cache entry returned for a negation query"
    )
    # Sources must NOT include pinecone — proves the negation filter ran on
    # the live retrieval path (not the bypassed cache).
    for s in data["sources"]:
        full = f"{(s.get('owner') or '').lower()}/{(s.get('name') or '').lower()}"
        assert "pinecone" not in full, (
            f"KAN-169 regression: pinecone leaked into sources via cache as {full!r}"
        )
    # And no ``llm_response:`` cache write must occur for this negation query
    # — so this turn's response can't pollute future cache hits either.
    #
    # KAN-182: scope the assertion to the ``llm_response:`` namespace
    # (mirroring the ``_get_side_effect`` filter above). The handler still
    # legitimately writes ``llm_cost:<date>`` via ``cost_tracker.record_cost``
    # for daily-budget accounting on EVERY live LLM call (KAN-169's bypass is
    # explicitly scoped to the response cache, not unrelated counters).
    # Asserting ``call_count == 0`` would conflate the two and silently fail
    # whenever cost tracking runs.
    llm_response_writes = [
        c for c in set_mock.call_args_list
        if c.args and isinstance(c.args[0], str) and c.args[0].startswith("llm_response:")
    ]
    assert len(llm_response_writes) == 0, (
        f"KAN-169 regression: cache.set was called for the llm_response: namespace "
        f"on a negation query (expected 0). llm_response: calls: {llm_response_writes}"
    )


@pytest.mark.asyncio
async def test_ask_non_negation_uses_cache(client_no_db: AsyncClient):
    """
    KAN-169: a non-negation query MUST still use the Redis cache fast-path.
    Verifies the bypass is scoped to negation queries only (no regression
    on cache hit rate for the common case).
    """
    from app.main import app
    from app.database import get_db

    _, override = _override_db(GOLDEN_ROWS)
    app.dependency_overrides[get_db] = override

    # Cached entry for a normal RAG-frameworks question — must be returned.
    cached_payload = {
        "answer": "CACHED: LangChain and LlamaIndex are great RAG frameworks.",
        "sources": [{
            "name": "langchain",
            "owner": "langchain-ai",
            "forked_from": "langchain-ai/langchain",
            "relevance_score": 0.93,
            "integration_tags": ["llm", "rag"],
        }],
        "tokens_used": {"input": 0, "output": 0, "total": 0},
        "model": "redis-cache",
    }

    cache_get_patch, cache_set_patch, set_mock = _patch_cache(llm_response_returns=cached_payload)

    with (
        _patch_embedding_model(),
        _patch_anthropic_key(),
        _patch_anthropic(),
        _patch_log_query(),
        _patch_create_task(),
        cache_get_patch,
        cache_set_patch,
    ):
        try:
            response = await client_no_db.post(
                "/intelligence/ask",
                json={"question": "What are the best RAG frameworks?"},
            )
        finally:
            app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 200, response.text
    data = response.json()
    # Cache hit was honored — the cached answer text must come through.
    assert data["answer"].startswith("CACHED:"), (
        "KAN-169 regression: non-negation query did NOT use the Redis cache"
    )
