from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from app.routers.analytics import cross_dimension_analytics


class _FetchAllResult:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


@pytest.mark.asyncio
async def test_cross_dimension_analytics_orders_pairs_from_query():
    db = AsyncMock()
    db.execute = AsyncMock(
        return_value=_FetchAllResult(
            [
                SimpleNamespace(dim1_value="DevTools", dim2_value="Agentic AI", repo_count=31),
                SimpleNamespace(dim1_value="Healthcare", dim2_value="Agentic AI", repo_count=12),
            ]
        )
    )

    response = await cross_dimension_analytics(dim1="industry", dim2="ai_trend", limit=2, db=db)

    assert response.dim1 == "industry"
    assert response.dim2 == "ai_trend"
    assert response.limit == 2
    assert [(pair.dim1_value, pair.dim2_value, pair.repo_count) for pair in response.pairs] == [
        ("DevTools", "Agentic AI", 31),
        ("Healthcare", "Agentic AI", 12),
    ]
    stmt, params = db.execute.await_args.args
    assert "repo_taxonomy" in str(stmt)
    assert params["dim1"] == "industry"
    assert params["dim2"] == "ai_trend"
    assert params["limit"] == 2


@pytest.mark.asyncio
async def test_cross_dimension_analytics_rejects_same_dimension():
    db = AsyncMock()

    with pytest.raises(HTTPException) as exc:
        await cross_dimension_analytics(dim1="industry", dim2="industry", limit=10, db=db)

    assert exc.value.status_code == 400
    assert "must be different" in exc.value.detail
