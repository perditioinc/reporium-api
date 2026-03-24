import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
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


@pytest.mark.asyncio
async def test_health_returns_ok_when_database_query_succeeds(monkeypatch):
    monkeypatch.setattr(main_module, "async_session_factory", _Factory(_SuccessfulSession()))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "db": "ok"}


@pytest.mark.asyncio
async def test_health_returns_503_when_database_query_fails(monkeypatch):
    monkeypatch.setattr(main_module, "async_session_factory", _Factory(_FailingSession()))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 503
    assert response.json()["status"] == "degraded"
    assert response.json()["db"] == "error"
