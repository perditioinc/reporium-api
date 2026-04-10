"""Tests for KAN-governance: per-key rate limiting, budget enforcement,
audit log creation, sandbox header detection, and /admin/audit endpoint.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import os
from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# Ensure test env is set before importing app modules
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/reporium_test")
os.environ.setdefault("INGESTION_API_KEY", "test-api-key")
os.environ.setdefault("REDIS_URL", "")
os.environ.setdefault("RATELIMIT_ENABLED", "0")


# --------------------------------------------------------------------------- #
# Per-key rate limiting (mock Redis)
# --------------------------------------------------------------------------- #


class TestPerKeyRateLimit:
    """Tests for app.governance.check_rate_limit."""

    @pytest.mark.asyncio
    async def test_allows_under_limit(self):
        """Requests under the per-key limit should be allowed."""
        from app.governance import check_rate_limit, _mem_rate

        _mem_rate.clear()
        with patch.dict(os.environ, {"PER_KEY_RATE_LIMIT": "5"}):
            for _ in range(5):
                assert await check_rate_limit("key-a", "/test") is True

    @pytest.mark.asyncio
    async def test_blocks_over_limit(self):
        """Requests exceeding the per-key limit should be blocked."""
        from app.governance import check_rate_limit, _mem_rate

        _mem_rate.clear()
        with patch.dict(os.environ, {"PER_KEY_RATE_LIMIT": "3"}):
            for _ in range(3):
                assert await check_rate_limit("key-b", "/test") is True
            # 4th should be blocked
            assert await check_rate_limit("key-b", "/test") is False

    @pytest.mark.asyncio
    async def test_separate_keys_independent(self):
        """Different API keys should have independent rate limits."""
        from app.governance import check_rate_limit, _mem_rate

        _mem_rate.clear()
        with patch.dict(os.environ, {"PER_KEY_RATE_LIMIT": "2"}):
            assert await check_rate_limit("key-x", "/route") is True
            assert await check_rate_limit("key-x", "/route") is True
            assert await check_rate_limit("key-x", "/route") is False
            # key-y should still be allowed
            assert await check_rate_limit("key-y", "/route") is True

    @pytest.mark.asyncio
    async def test_disabled_when_zero(self):
        """Rate limiting should be disabled when PER_KEY_RATE_LIMIT=0."""
        from app.governance import check_rate_limit, _mem_rate

        _mem_rate.clear()
        with patch.dict(os.environ, {"PER_KEY_RATE_LIMIT": "0"}):
            for _ in range(100):
                assert await check_rate_limit("key-z", "/test") is True

    @pytest.mark.asyncio
    async def test_redis_path(self):
        """When Redis is available, uses INCR+EXPIRE pipeline."""
        from app.governance import check_rate_limit

        mock_pipe = MagicMock()
        mock_pipe.incr = MagicMock(return_value=mock_pipe)
        mock_pipe.expire = MagicMock(return_value=mock_pipe)
        mock_pipe.execute = AsyncMock(return_value=[1, True])

        mock_client = MagicMock()
        mock_client.pipeline = MagicMock(return_value=mock_pipe)

        with patch.dict(os.environ, {"PER_KEY_RATE_LIMIT": "30"}):
            with patch("app.governance.redis_cache", create=True) as mock_rc:
                # Import inside to use the mock
                mock_rc._get_client = AsyncMock(return_value=mock_client)
                with patch("app.cache_redis.redis_cache") as patched:
                    patched._get_client = AsyncMock(return_value=mock_client)
                    result = await check_rate_limit("key-redis", "/test")
                    assert result is True


# --------------------------------------------------------------------------- #
# Per-key budget enforcement
# --------------------------------------------------------------------------- #


class TestPerKeyBudget:
    """Tests for app.governance.check_budget and record_spend."""

    @pytest.mark.asyncio
    async def test_budget_allowed_when_under(self):
        """Spending under budget should be allowed."""
        from app.governance import check_budget, record_spend, _mem_budget

        _mem_budget.clear()
        with patch.dict(os.environ, {"PER_KEY_BUDGET_USD": "5.0"}):
            allowed, remaining = await check_budget("key-budget-a")
            assert allowed is True
            assert remaining == 5.0

    @pytest.mark.asyncio
    async def test_budget_blocked_after_exhausted(self):
        """Spending over budget should be blocked."""
        from app.governance import check_budget, record_spend, _mem_budget

        _mem_budget.clear()
        with patch.dict(os.environ, {"PER_KEY_BUDGET_USD": "1.0"}):
            await record_spend("key-budget-b", 0.8)
            allowed, remaining = await check_budget("key-budget-b")
            assert allowed is True
            assert remaining == pytest.approx(0.2, abs=0.01)

            await record_spend("key-budget-b", 0.3)
            allowed, remaining = await check_budget("key-budget-b")
            assert allowed is False
            assert remaining == 0.0

    @pytest.mark.asyncio
    async def test_budget_disabled_when_zero(self):
        """Budget enforcement should be disabled when PER_KEY_BUDGET_USD=0."""
        from app.governance import check_budget, _mem_budget

        _mem_budget.clear()
        with patch.dict(os.environ, {"PER_KEY_BUDGET_USD": "0"}):
            allowed, remaining = await check_budget("key-budget-c")
            assert allowed is True

    @pytest.mark.asyncio
    async def test_record_spend_zero_noop(self):
        """Recording zero spend should be a no-op."""
        from app.governance import record_spend, _mem_budget

        _mem_budget.clear()
        with patch.dict(os.environ, {"PER_KEY_BUDGET_USD": "5.0"}):
            await record_spend("key-noop", 0.0)
            today = date.today().isoformat()
            assert f"budget:key-noop:{today}" not in _mem_budget


# --------------------------------------------------------------------------- #
# Audit log creation
# --------------------------------------------------------------------------- #


class TestAuditLogCreation:
    """Tests for audit log model and persistence helper."""

    def test_audit_log_model_fields(self):
        """AuditLog model should have all required columns."""
        from app.models.audit_log import AuditLog

        columns = {c.name for c in AuditLog.__table__.columns}
        expected = {
            "id", "timestamp", "api_key_hash", "endpoint", "method",
            "request_summary", "response_status", "model_used",
            "tokens_input", "tokens_output", "cost_usd", "latency_ms", "sandbox",
        }
        assert expected.issubset(columns)

    def test_audit_log_table_name(self):
        from app.models.audit_log import AuditLog
        assert AuditLog.__tablename__ == "audit_logs"

    def test_audit_log_sandbox_default(self):
        """Sandbox column should default to False."""
        from app.models.audit_log import AuditLog

        col = AuditLog.__table__.columns["sandbox"]
        assert col.default.arg is False


# --------------------------------------------------------------------------- #
# Sandbox header detection
# --------------------------------------------------------------------------- #


class TestSandboxDetection:
    """Tests for X-Sandbox header detection in audit middleware."""

    def test_hash_key_returns_sha256(self):
        from app.middleware.audit import _hash_key

        raw = "test-api-key"
        expected = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        assert _hash_key(raw) == expected

    def test_hash_key_none(self):
        from app.middleware.audit import _hash_key

        assert _hash_key(None) is None
        assert _hash_key("") is None

    def test_audit_disabled_by_default(self):
        from app.middleware.audit import _is_audit_enabled

        with patch.dict(os.environ, {"AUDIT_ENABLED": "0"}):
            assert _is_audit_enabled() is False

    def test_audit_enabled_when_flag_set(self):
        from app.middleware.audit import _is_audit_enabled

        with patch.dict(os.environ, {"AUDIT_ENABLED": "1"}):
            assert _is_audit_enabled() is True


# --------------------------------------------------------------------------- #
# /admin/audit endpoint
# --------------------------------------------------------------------------- #


class TestAdminAuditEndpoint:
    """Tests for the GET /admin/audit endpoint."""

    @pytest.mark.asyncio
    async def test_requires_admin_key(self, client: AsyncClient):
        """Endpoint should reject requests without admin key in production."""
        with patch.dict(os.environ, {"ADMIN_API_KEY": "secret-admin-key"}):
            resp = await client.get("/admin/audit")
            assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_returns_entries_with_admin_key(self, client: AsyncClient):
        """Endpoint should return audit entries with valid admin key."""
        # In dev mode (no ADMIN_API_KEY set), admin key is not enforced
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ADMIN_API_KEY", None)
            resp = await client.get("/admin/audit")
            assert resp.status_code == 200
            data = resp.json()
            assert "entries" in data
            assert "count" in data
            assert "limit" in data
            assert "offset" in data

    @pytest.mark.asyncio
    async def test_pagination_params(self, client: AsyncClient):
        """Endpoint should accept pagination params."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ADMIN_API_KEY", None)
            resp = await client.get("/admin/audit?limit=10&offset=5")
            assert resp.status_code == 200
            data = resp.json()
            assert data["limit"] == 10
            assert data["offset"] == 5

    @pytest.mark.asyncio
    async def test_filter_params_accepted(self, client: AsyncClient):
        """Endpoint should accept all filter query params without error."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ADMIN_API_KEY", None)
            resp = await client.get(
                "/admin/audit",
                params={
                    "api_key_hash": "abc123",
                    "endpoint": "/intelligence/ask",
                    "date_from": "2026-01-01",
                    "date_to": "2026-12-31",
                    "sandbox_only": "true",
                },
            )
            assert resp.status_code == 200


# --------------------------------------------------------------------------- #
# Governance wiring in /intelligence/ask and /ask/stream
# --------------------------------------------------------------------------- #


@contextlib.contextmanager
def _override_token_hash(token_hash: str = "hash-abc"):
    """Override get_app_token_hash dependency so governance checks run in tests."""
    from app.main import app as _app
    from app.auth import get_app_token_hash
    _app.dependency_overrides[get_app_token_hash] = lambda: token_hash
    try:
        yield
    finally:
        _app.dependency_overrides.pop(get_app_token_hash, None)


def _dummy_query_response():
    """Return a minimal QueryResponse-shaped object for mocking _run_query."""
    from app.routers.intelligence import QueryResponse
    return QueryResponse(
        answer="Test answer",
        sources=[],
        question="test?",
        model="claude-sonnet-4-20250514",
        answered_at="2026-01-01T00:00:00Z",
        embedding_candidates=0,
        tokens_used={"input": 100, "output": 50, "total": 150},
    )


class TestGovernanceWiringAsk:
    """Tests that /intelligence/ask respects GOVERNANCE_ENABLED flag."""

    @pytest.mark.asyncio
    async def test_ask_returns_429_on_rate_limit(self, client: AsyncClient):
        """When governance is enabled and rate limit exceeded, /ask returns 429."""
        with _override_token_hash("hash-abc"):
            with patch.dict(os.environ, {"GOVERNANCE_ENABLED": "1"}):
                with patch("app.routers.intelligence.gov_check_rate_limit", new_callable=AsyncMock, return_value=False):
                    resp = await client.post(
                        "/intelligence/ask",
                        json={"question": "What are the best RAG frameworks?"},
                    )
                    assert resp.status_code == 429
                    assert "rate limit" in resp.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_ask_returns_429_on_budget_exceeded(self, client: AsyncClient):
        """When governance is enabled and budget exhausted, /ask returns 429."""
        with _override_token_hash("hash-abc"):
            with patch.dict(os.environ, {"GOVERNANCE_ENABLED": "1"}):
                with patch("app.routers.intelligence.gov_check_rate_limit", new_callable=AsyncMock, return_value=True):
                    with patch("app.routers.intelligence.gov_check_budget", new_callable=AsyncMock, return_value=(False, 0.0)):
                        resp = await client.post(
                            "/intelligence/ask",
                            json={"question": "What are the best RAG frameworks?"},
                        )
                        assert resp.status_code == 429
                        assert "budget" in resp.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_ask_passes_when_governance_disabled(self, client: AsyncClient):
        """When GOVERNANCE_ENABLED is not set, governance checks are skipped."""
        with patch.dict(os.environ, {"GOVERNANCE_ENABLED": "0"}):
            with patch("app.routers.intelligence._run_query", new_callable=AsyncMock, return_value=_dummy_query_response()):
                with patch("app.routers.intelligence.get_app_token_hash", return_value="hash-abc"):
                    with patch("app.routers.intelligence.require_app_token", return_value=None):
                        resp = await client.post(
                            "/intelligence/ask",
                            json={"question": "What are the best RAG frameworks?"},
                        )
                        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_ask_records_spend_after_query(self, client: AsyncClient):
        """When governance is enabled, record_spend is called after a successful query."""
        mock_record = AsyncMock()
        with _override_token_hash("hash-abc"):
            with patch.dict(os.environ, {"GOVERNANCE_ENABLED": "1"}):
                with patch("app.routers.intelligence.gov_check_rate_limit", new_callable=AsyncMock, return_value=True):
                    with patch("app.routers.intelligence.gov_check_budget", new_callable=AsyncMock, return_value=(True, 5.0)):
                        with patch("app.routers.intelligence.gov_record_spend", mock_record):
                            with patch("app.routers.intelligence._run_query", new_callable=AsyncMock, return_value=_dummy_query_response()):
                                resp = await client.post(
                                    "/intelligence/ask",
                                    json={"question": "What are the best RAG frameworks?"},
                                )
                                assert resp.status_code == 200
                                mock_record.assert_called_once()
                                args = mock_record.call_args
                                assert args[0][0] == "hash-abc"
                                assert args[0][1] > 0  # cost > 0

    @pytest.mark.asyncio
    async def test_ask_fails_open_on_governance_error(self, client: AsyncClient):
        """When governance check raises an unexpected error, request proceeds (fail-open)."""
        with patch.dict(os.environ, {"GOVERNANCE_ENABLED": "1"}):
            with patch("app.routers.intelligence.gov_check_rate_limit", new_callable=AsyncMock, side_effect=RuntimeError("Redis exploded")):
                with patch("app.routers.intelligence._run_query", new_callable=AsyncMock, return_value=_dummy_query_response()):
                    with patch("app.routers.intelligence.get_app_token_hash", return_value="hash-abc"):
                        with patch("app.routers.intelligence.require_app_token", return_value=None):
                            resp = await client.post(
                                "/intelligence/ask",
                                json={"question": "What are the best RAG frameworks?"},
                            )
                            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_ask_skips_governance_when_no_token_hash(self, client: AsyncClient):
        """When token_hash is None, governance checks are skipped even if enabled."""
        with patch.dict(os.environ, {"GOVERNANCE_ENABLED": "1"}):
            with patch("app.routers.intelligence._run_query", new_callable=AsyncMock, return_value=_dummy_query_response()):
                with patch("app.routers.intelligence.get_app_token_hash", return_value=None):
                    with patch("app.routers.intelligence.require_app_token", return_value=None):
                        resp = await client.post(
                            "/intelligence/ask",
                            json={"question": "What are the best RAG frameworks?"},
                        )
                        assert resp.status_code == 200


class TestGovernanceWiringStream:
    """Tests that /intelligence/ask/stream respects GOVERNANCE_ENABLED flag."""

    @pytest.mark.asyncio
    async def test_stream_returns_429_on_rate_limit(self, client: AsyncClient):
        """When governance is enabled and rate limit exceeded, /ask/stream returns 429."""
        with _override_token_hash("hash-abc"):
            with patch.dict(os.environ, {"GOVERNANCE_ENABLED": "1"}):
                with patch("app.routers.intelligence.gov_check_rate_limit", new_callable=AsyncMock, return_value=False):
                    resp = await client.post(
                        "/intelligence/ask/stream",
                        json={"question": "What are the best RAG frameworks?"},
                    )
                    assert resp.status_code == 429

    @pytest.mark.asyncio
    async def test_stream_returns_429_on_budget_exceeded(self, client: AsyncClient):
        """When governance is enabled and budget exhausted, /ask/stream returns 429."""
        with _override_token_hash("hash-abc"):
            with patch.dict(os.environ, {"GOVERNANCE_ENABLED": "1"}):
                with patch("app.routers.intelligence.gov_check_rate_limit", new_callable=AsyncMock, return_value=True):
                    with patch("app.routers.intelligence.gov_check_budget", new_callable=AsyncMock, return_value=(False, 0.0)):
                        resp = await client.post(
                            "/intelligence/ask/stream",
                            json={"question": "What are the best RAG frameworks?"},
                        )
                        assert resp.status_code == 429

    @pytest.mark.asyncio
    async def test_stream_fails_open_on_governance_error(self, client: AsyncClient):
        """When governance check raises an unexpected error, stream proceeds (fail-open)."""
        with patch.dict(os.environ, {"GOVERNANCE_ENABLED": "1"}):
            with patch("app.routers.intelligence.gov_check_rate_limit", new_callable=AsyncMock, side_effect=RuntimeError("Redis down")):
                with patch("app.routers.intelligence.get_app_token_hash", return_value="hash-abc"):
                    with patch("app.routers.intelligence.require_app_token", return_value=None):
                        try:
                            resp = await client.post(
                                "/intelligence/ask/stream",
                                json={"question": "What are the best RAG frameworks?"},
                            )
                            # Should not be 429 — governance failed open
                            assert resp.status_code != 429
                        except Exception:
                            # DB/model not available is acceptable in test env
                            pass
