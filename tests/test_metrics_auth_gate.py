"""
Tests for the optional admin-key gate on observability endpoints (#236).

The gate is controlled by the ``METRICS_REQUIRE_AUTH`` env var:
  - unset / empty  -> endpoints stay open (pre-#236 behavior)
  - "1"            -> X-Admin-Key header required

Covers ``app.auth.require_metrics_access`` directly and the /metrics/slo
endpoint end-to-end via the TestClient fixture in conftest.
"""
import pytest
from fastapi import HTTPException

from app.auth import require_metrics_access


# ---------------------------------------------------------------------------
# Unit tests — require_metrics_access dependency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_require_metrics_access_noop_when_flag_unset(monkeypatch):
    monkeypatch.delenv("METRICS_REQUIRE_AUTH", raising=False)
    monkeypatch.setenv("ADMIN_API_KEY", "secret-key")
    # No header, no flag -> must be a no-op (no exception).
    assert await require_metrics_access(x_admin_key=None) is None


@pytest.mark.asyncio
async def test_require_metrics_access_noop_when_flag_empty(monkeypatch):
    monkeypatch.setenv("METRICS_REQUIRE_AUTH", "")
    monkeypatch.setenv("ADMIN_API_KEY", "secret-key")
    assert await require_metrics_access(x_admin_key=None) is None


@pytest.mark.asyncio
async def test_require_metrics_access_403_when_flag_set_no_header(monkeypatch):
    monkeypatch.setenv("METRICS_REQUIRE_AUTH", "1")
    monkeypatch.setenv("ADMIN_API_KEY", "secret-key")
    with pytest.raises(HTTPException) as exc_info:
        await require_metrics_access(x_admin_key=None)
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_require_metrics_access_403_when_flag_set_wrong_key(monkeypatch):
    monkeypatch.setenv("METRICS_REQUIRE_AUTH", "1")
    monkeypatch.setenv("ADMIN_API_KEY", "secret-key")
    with pytest.raises(HTTPException) as exc_info:
        await require_metrics_access(x_admin_key="nope")
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_require_metrics_access_passes_with_correct_key(monkeypatch):
    monkeypatch.setenv("METRICS_REQUIRE_AUTH", "1")
    monkeypatch.setenv("ADMIN_API_KEY", "secret-key")
    # Must not raise and must return None.
    assert await require_metrics_access(x_admin_key="secret-key") is None


# ---------------------------------------------------------------------------
# Integration tests — /metrics/slo end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_metrics_slo_open_when_flag_unset(client, monkeypatch):
    """Backward-compat: gate off by default, endpoint stays open."""
    monkeypatch.delenv("METRICS_REQUIRE_AUTH", raising=False)
    resp = await client.get("/metrics/slo")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_metrics_slo_403_when_flag_set_no_header(client, monkeypatch):
    monkeypatch.setenv("METRICS_REQUIRE_AUTH", "1")
    monkeypatch.setenv("ADMIN_API_KEY", "secret-key")
    resp = await client.get("/metrics/slo")
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_metrics_slo_200_when_flag_set_correct_key(client, monkeypatch):
    monkeypatch.setenv("METRICS_REQUIRE_AUTH", "1")
    monkeypatch.setenv("ADMIN_API_KEY", "secret-key")
    resp = await client.get("/metrics/slo", headers={"X-Admin-Key": "secret-key"})
    assert resp.status_code == 200
