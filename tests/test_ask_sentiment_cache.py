"""
KAN-ask-sentiment-cache: a cached answer that the user rated thumbs-DOWN
(sentiment = 'negative') must NOT be re-served as a semantic cache hit.

The semantic cache lookup (_find_semantic_cache_hit) previously selected the
nearest neighbour from query_log purely on embedding distance, ignoring the
``sentiment`` column. That allowed a negatively-rated answer to be served again
on the next similar question, defeating the thumbs-down feedback loop.

These tests pin the fix at the SQL layer (the query must exclude negative
sentiment) and behaviourally (a negative-only neighbour is treated as a miss).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest

from app.routers.intelligence import _find_semantic_cache_hit


def _captured_sql(db_mock):
    """Return the SQL text string passed to the first db.execute call."""
    call = db_mock.execute.call_args
    sql_arg = call[0][0]
    # SQLAlchemy TextClause stringifies to its SQL body.
    return str(sql_arg).lower()


@pytest.mark.asyncio
async def test_cache_lookup_sql_excludes_negative_sentiment():
    """The SELECT must filter out rows whose sentiment is 'negative'."""
    db = AsyncMock()
    db.execute = AsyncMock(return_value=SimpleNamespace(first=lambda: None))

    await _find_semantic_cache_hit(
        db, question_embedding=np.full(384, 0.1, dtype=np.float32)
    )

    sql = _captured_sql(db)
    assert "sentiment" in sql, "cache query must reference the sentiment column"
    # Must explicitly exclude the negative sentiment.
    assert "negative" in sql


@pytest.mark.asyncio
async def test_negative_rated_answer_not_served():
    """When the DB (correctly filtering) returns no row, result is a miss."""
    db = AsyncMock()
    db.execute = AsyncMock(return_value=SimpleNamespace(first=lambda: None))

    cached = await _find_semantic_cache_hit(
        db, question_embedding=np.full(384, 0.1, dtype=np.float32)
    )
    assert cached is None


@pytest.mark.asyncio
async def test_positive_or_neutral_answer_still_served():
    """A non-negative cached answer is still returned as a hit (no regression)."""
    row = SimpleNamespace(
        answer_full="Good cached answer",
        sources=[{"owner": "perditioinc", "name": "reporium", "relevance_score": 0.9}],
        model="claude-sonnet-4-20250514",
    )
    db = AsyncMock()
    db.execute = AsyncMock(return_value=SimpleNamespace(first=lambda: row))

    cached = await _find_semantic_cache_hit(
        db, question_embedding=np.full(384, 0.1, dtype=np.float32)
    )
    assert cached is not None
    answer, sources, model = cached
    assert answer == "Good cached answer"
    assert sources[0].owner == "perditioinc"
    assert model == "claude-sonnet-4-20250514"
