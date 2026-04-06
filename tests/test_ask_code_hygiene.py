"""Tests for KAN-ask-code-hygiene: PII redaction wiring, dead-code removal,
stale TODO cleanup, and embedding warm-up timing."""
from __future__ import annotations

import ast
import importlib
import textwrap
import time


# ---------------------------------------------------------------------------
# Task 1: PII redaction is wired into _log_query
# ---------------------------------------------------------------------------


def test_log_query_calls_redact_pii():
    """The _log_query function must apply redact_pii to the question before
    inserting into the database."""
    import inspect
    from app.routers.intelligence import _log_query

    source = inspect.getsource(_log_query)
    assert "redact_pii" in source, (
        "_log_query should call redact_pii on the question before INSERT"
    )


def test_redact_pii_imported_in_intelligence():
    """redact_pii must be imported at module level in intelligence.py."""
    from app.routers import intelligence

    assert hasattr(intelligence, "redact_pii"), (
        "intelligence module should import redact_pii from app.privacy"
    )


# ---------------------------------------------------------------------------
# Task 2: Dead code removal
# ---------------------------------------------------------------------------


def test_cosine_similarity_removed():
    """cosine_similarity was dead code and should no longer exist."""
    from app.routers import intelligence

    assert not hasattr(intelligence, "cosine_similarity"), (
        "cosine_similarity should have been removed (dead code)"
    )


def test_get_anthropic_key_not_imported():
    """get_anthropic_key was unused and should no longer be imported."""
    import inspect
    from app.routers import intelligence

    source = inspect.getsource(intelligence)
    # Ensure the symbol isn't imported anywhere in the module
    assert "get_anthropic_key" not in source, (
        "get_anthropic_key import should have been removed (unused)"
    )


# ---------------------------------------------------------------------------
# Task 3: Stale TODO comments removed
# ---------------------------------------------------------------------------


def test_no_stale_session_cleanup_todo():
    """The TODO about session cleanup should be removed — retention is
    implemented via retention_loop."""
    import inspect
    from app.routers import intelligence

    source = inspect.getsource(intelligence)
    assert "TODO" not in source or "periodic cleanup" not in source, (
        "Stale TODO about periodic session cleanup should have been removed"
    )


# ---------------------------------------------------------------------------
# Task 4: Embedding warm-up timing in main.py
# ---------------------------------------------------------------------------


def test_embedding_warmup_has_timing():
    """The lifespan function in main.py should time the embedding warm-up."""
    import inspect
    from app.main import lifespan

    source = inspect.getsource(lifespan)
    assert "perf_counter" in source, (
        "Embedding warm-up should be timed with time.perf_counter"
    )
    # The log message should include the timing value
    assert "ms" in source.lower(), (
        "Embedding warm-up log message should report timing in ms"
    )
