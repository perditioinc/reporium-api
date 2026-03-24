from datetime import datetime, timedelta, timezone

import pytest
from httpx import ASGITransport, AsyncClient

from app.database import get_db
from app.main import app
from app.routers import trends as trends_router


class _FakeScalarResult:
    def __init__(self, value):
        self._value = value

    def scalar_one(self):
        return self._value


class _FakeMappingsResult:
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return self

    def all(self):
        return self._rows


class _FakeTupleResult:
    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)


class FakeTrendSession:
    def __init__(self):
        self.latest = datetime(2026, 3, 24, 12, 0, tzinfo=timezone.utc)
        self.earliest = self.latest - timedelta(days=14)
        self.previous = self.latest - timedelta(days=7)

    async def execute(self, stmt):
        sql = str(stmt)
        if "max(trend_snapshots.snapshotted_at)" in sql and "WHERE trend_snapshots.snapshotted_at <" not in sql:
            return _FakeScalarResult(self.latest)
        if "min(trend_snapshots.snapshotted_at)" in sql and "GROUP BY" not in sql:
            return _FakeScalarResult(self.earliest)
        if "count(distinct(date(trend_snapshots.snapshotted_at)))" in sql:
            return _FakeScalarResult(3)
        if "WHERE trend_snapshots.snapshotted_at <" in sql:
            return _FakeScalarResult(self.previous)
        if "ORDER BY trend_snapshots.commit_count_7d DESC" in sql:
            return _FakeMappingsResult([
                {"tag": "Agents", "repo_count": 12, "commit_count_7d": 30},
                {"tag": "RAG", "repo_count": 9, "commit_count_7d": 12},
                {"tag": "Vision", "repo_count": 4, "commit_count_7d": 6},
            ])
        if "SELECT trend_snapshots.tag, trend_snapshots.commit_count_7d" in sql:
            return _FakeTupleResult([
                type("Row", (), {"tag": "Agents", "commit_count_7d": 10})(),
                type("Row", (), {"tag": "RAG", "commit_count_7d": 6})(),
            ])
        if "min(trend_snapshots.snapshotted_at) AS first_seen_at" in sql:
            return _FakeMappingsResult([
                {"tag": "Vision", "first_seen_at": self.latest - timedelta(days=3)},
                {"tag": "Agents", "first_seen_at": self.latest - timedelta(days=2)},
            ])
        raise AssertionError(f"Unexpected SQL: {sql}")


@pytest.mark.asyncio
async def test_get_trends_report_returns_sorted_report(monkeypatch):
    async def fake_get_db():
        yield FakeTrendSession()

    async def fake_cache_get(_key):
        return None

    async def fake_cache_set(_key, _value, ttl=None):
        return None

    app.dependency_overrides[get_db] = fake_get_db
    monkeypatch.setattr(trends_router.cache, "get", fake_cache_get)
    monkeypatch.setattr(trends_router.cache, "set", fake_cache_set)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/trends/report")

    app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["period"]["snapshots"] == 3
    assert payload["trending"][0]["name"] == "Agents"
    assert payload["trending"][0]["repoCount"] == 12
    assert payload["trending"][0]["changePercent"] == 200.0
    assert payload["emerging"][0]["name"] == "Agents"
