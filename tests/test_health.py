import pytest
from httpx import ASGITransport, AsyncClient

from app.main import _pool_stats, app
import app.main as main_module


class _SuccessfulSession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def execute(self, _stmt):
        return None


class _FailingSession:
    async def __aenter__(self):
        raise RuntimeError("database offline")

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _Factory:
    def __init__(self, session):
        self._session = session

    def __call__(self):
        return self._session


_POOL_KEYS = {"size", "checked_out", "overflow"}


@pytest.mark.asyncio
async def test_health_returns_ok_when_database_query_succeeds(monkeypatch):
    monkeypatch.setattr(main_module, "async_session_factory", _Factory(_SuccessfulSession()))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["db"] == "ok"
    assert set(body["pool"].keys()) == _POOL_KEYS


@pytest.mark.asyncio
async def test_health_returns_503_when_database_query_fails(monkeypatch):
    monkeypatch.setattr(main_module, "async_session_factory", _Factory(_FailingSession()))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["db"] == "error"
    assert set(body["pool"].keys()) == _POOL_KEYS


def test_pool_stats_returns_none_for_nullpool_without_counter_attrs():
    # NullPool stand-in: no size/checkedout/overflow attrs.
    class _NullPoolLike:
        pass

    stats = _pool_stats(_NullPoolLike())

    assert stats == {"size": None, "checked_out": None, "overflow": None}


def test_pool_stats_returns_none_when_counter_methods_raise():
    class _BrokenPool:
        def size(self):
            raise RuntimeError("pool not initialized")

        def checkedout(self):
            raise RuntimeError("pool not initialized")

        def overflow(self):
            raise RuntimeError("pool not initialized")

    stats = _pool_stats(_BrokenPool())

    assert stats == {"size": None, "checked_out": None, "overflow": None}


def test_pool_stats_reports_counters_from_queue_pool_like():
    class _QueuePoolLike:
        def size(self):
            return 5

        def checkedout(self):
            return 2

        def overflow(self):
            return 1

    stats = _pool_stats(_QueuePoolLike())

    assert stats == {"size": 5, "checked_out": 2, "overflow": 1}
