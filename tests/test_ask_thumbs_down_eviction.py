"""KAN-586 / reporium#433: a thumbs-down answer must NOT be re-served from the
semantic cache.

Root cause: ``_find_semantic_cache_hit`` selected the nearest prior answer by
embedding distance with no regard for whether a user (or admin via PATCH
/admin/asks) had marked it sentiment='negative'. So once a wrong/unhelpful
answer landed in query_log, the next near-identical question kept getting that
same bad answer served instantly from cache -- the feedback loop did nothing.

Fix: the cache lookup now excludes rows with sentiment='negative'. These tests
prove (a) the SQL carries the exclusion filter, and (b) end-to-end against a
real Postgres (DB-gated, skipped in CI without a DB) a negative row is skipped
while a sibling positive/neutral row is still served.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
from httpx import AsyncClient
from sqlalchemy import text

from app.database import async_session_factory
from app.routers.intelligence import _find_semantic_cache_hit


def _emb() -> np.ndarray:
    return np.full(384, 0.1, dtype=np.float32)


@pytest.mark.asyncio
async def test_semantic_cache_query_excludes_negative_sentiment():
    """The lookup SQL must filter out sentiment='negative' rows so a
    thumbs-down answer is never a cache candidate."""
    captured: dict = {}

    async def fake_execute(stmt, params=None):
        captured["sql"] = str(stmt)
        captured["params"] = params
        return SimpleNamespace(first=lambda: None)

    db = AsyncMock()
    db.execute = fake_execute

    result = await _find_semantic_cache_hit(db, question_embedding=_emb())

    assert result is None  # no row -> falls through to fresh generation
    sql = captured["sql"].lower()
    # The exclusion filter must be present and reference the sentiment column.
    assert "sentiment" in sql, captured["sql"]
    assert "negative" in sql, captured["sql"]
    # Belt-and-braces: a NULL or non-negative sentiment is still cache-eligible.
    assert "is null" in sql or "<> 'negative'" in sql, captured["sql"]


@pytest.mark.asyncio
async def test_positive_and_neutral_rows_still_served():
    """A row that is NOT thumbs-down (the DB returns it because it passed the
    sentiment filter) is still returned by the cache lookup -- the fix is a
    no-op for the common case and only suppresses negatives."""
    row = SimpleNamespace(
        answer_full="Helpful cached answer",
        sources=[{"owner": "perditioinc", "name": "reporium", "relevance_score": 0.9}],
        model="claude-sonnet-4-20250514",
    )
    db = AsyncMock()
    db.execute = AsyncMock(return_value=SimpleNamespace(first=lambda: row))

    cached = await _find_semantic_cache_hit(db, question_embedding=_emb())

    assert cached is not None
    answer, sources, model = cached
    assert answer == "Helpful cached answer"
    assert sources[0].owner == "perditioinc"


# ---------------------------------------------------------------------------
# DB-gated end-to-end proof: requires a real Postgres with pgvector. Skips
# cleanly in CI without a DB (mirrors conftest's _test_db_available gate).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_thumbs_down_row_evicted_end_to_end(client: AsyncClient):
    """Insert two near-identical cached answers; mark one sentiment='negative'.
    The lookup must return the NON-negative answer, never the thumbs-down one.

    Depends on the ``client`` fixture only to pull in conftest's ``_setup_db``
    (which creates the schema + the question_embedding_vec column). DB-gated:
    skips cleanly in CI without a reachable Postgres.
    """
    base = [0.05] * 384
    near = [0.05] * 384
    near[0] = 0.051  # tiny perturbation -> still within the distance threshold

    def _vec_literal(v):
        return "[" + ",".join(f"{x:.6f}" for x in v) + "]"

    async with async_session_factory() as session:
        # Clean slate for determinism.
        await session.execute(text("DELETE FROM query_log"))
        # Negative (thumbs-down) row -- closest to the query.
        await session.execute(
            text(
                """
                INSERT INTO query_log
                    (question, answer_full, sources, model, sentiment,
                     question_embedding_vec)
                VALUES
                    (:q, :a, CAST(:s AS jsonb), :m, 'negative',
                     CAST(:vec AS vector))
                """
            ),
            {
                "q": "what is vllm",
                "a": "WRONG thumbs-down answer",
                "s": "[]",
                "m": "test-model",
                "vec": _vec_literal(base),
            },
        )
        # Positive row -- slightly further but still within threshold.
        await session.execute(
            text(
                """
                INSERT INTO query_log
                    (question, answer_full, sources, model, sentiment,
                     question_embedding_vec)
                VALUES
                    (:q, :a, CAST(:s AS jsonb), :m, 'positive',
                     CAST(:vec AS vector))
                """
            ),
            {
                "q": "what is vllm engine",
                "a": "GOOD positive answer",
                "s": "[]",
                "m": "test-model",
                "vec": _vec_literal(near),
            },
        )
        await session.commit()

    async with async_session_factory() as session:
        cached = await _find_semantic_cache_hit(
            session, question_embedding=np.array(base, dtype=np.float32)
        )

    # The closest row is the negative one; it MUST be skipped. We get the
    # positive sibling -- never the thumbs-down answer.
    assert cached is not None, "positive sibling should still be cache-eligible"
    answer, _sources, _model = cached
    assert answer == "GOOD positive answer"
    assert answer != "WRONG thumbs-down answer"
