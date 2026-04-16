import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

import app.auth as auth


@pytest.fixture(autouse=True)
def clean_ingest_env(monkeypatch):
    """Ensure no env var leaks between ingest-auth tests."""
    monkeypatch.delenv("INGEST_API_KEY", raising=False)
    monkeypatch.delenv("INGESTION_API_KEY", raising=False)
    monkeypatch.setattr(auth.settings, "ingestion_api_key", None)
    yield


@pytest.mark.asyncio
async def test_verify_api_key_accepts_ingest_api_key_alias(monkeypatch):
    monkeypatch.setenv("INGEST_API_KEY", "secret-ingest")

    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer",
        credentials="secret-ingest",
    )

    assert await auth.verify_api_key(credentials) == "secret-ingest"


@pytest.mark.asyncio
async def test_require_ingest_key_accepts_ingestion_api_key_alias(monkeypatch):
    monkeypatch.setattr(auth, "_IS_PRODUCTION", True)
    monkeypatch.setenv("INGESTION_API_KEY", "secret-ingest")

    assert await auth.require_ingest_key(
        x_ingest_key="secret-ingest",
        x_admin_key=None,
    ) is None


@pytest.mark.asyncio
async def test_verify_api_key_returns_500_when_ingest_key_missing():
    # clean_ingest_env (autouse) already clears INGEST_API_KEY, INGESTION_API_KEY, settings
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer",
        credentials="anything",
    )

    with pytest.raises(HTTPException) as exc_info:
        await auth.verify_api_key(credentials)

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Server misconfiguration"
