"""Tests for OpenTelemetry integration and /metrics/export endpoint."""

import os
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from tests.conftest import AUTH_HEADERS


# ---------------------------------------------------------------------------
# init_telemetry tests
# ---------------------------------------------------------------------------


def test_init_telemetry_noop_when_disabled():
    """OTEL_ENABLED=0 (default) should return None without touching the provider."""
    with patch.dict(os.environ, {"OTEL_ENABLED": "0"}, clear=False):
        from app.telemetry import init_telemetry

        result = init_telemetry()
        assert result is None


def test_init_telemetry_noop_when_unset():
    """When OTEL_ENABLED is not set at all, init_telemetry should still no-op."""
    env = os.environ.copy()
    env.pop("OTEL_ENABLED", None)
    with patch.dict(os.environ, env, clear=True):
        from app.telemetry import init_telemetry

        result = init_telemetry()
        assert result is None


def test_init_telemetry_creates_provider_when_enabled():
    """OTEL_ENABLED=1 should create a TracerProvider and set it globally."""
    with patch.dict(os.environ, {"OTEL_ENABLED": "1"}, clear=False):
        # Mock the GCP-specific imports so tests run without google-cloud deps.
        mock_exporter = MagicMock()
        mock_instrumentor = MagicMock()
        with (
            patch(
                "opentelemetry.exporter.cloud_trace.CloudTraceSpanExporter",
                return_value=mock_exporter,
            ),
            patch(
                "opentelemetry.instrumentation.fastapi.FastAPIInstrumentor",
                mock_instrumentor,
            ),
        ):
            from app.telemetry import init_telemetry

            provider = init_telemetry()
            assert provider is not None

            # Verify a TracerProvider was returned
            from opentelemetry.sdk.trace import TracerProvider

            assert isinstance(provider, TracerProvider)

            # Clean up: shut down the provider to avoid leaking span processors.
            provider.shutdown()


# ---------------------------------------------------------------------------
# /metrics/export endpoint tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_metrics_export_shape(client: AsyncClient):
    """GET /metrics/export should return JSON with the expected top-level keys."""
    resp = await client.get("/metrics/export", headers=AUTH_HEADERS)
    assert resp.status_code == 200

    data = resp.json()
    assert "generated_at" in data
    assert "slo" in data
    assert "spend" in data
    assert "embeddings" in data
    assert "revision" in data

    # SLO section
    slo = data["slo"]
    assert "window_seconds" in slo
    assert "routes" in slo

    # Spend section
    spend = data["spend"]
    assert "usd_24h" in spend
    assert "cache_hit_rate" in spend
    assert "status" in spend
    assert "daily_budget_usd" in spend

    # Embeddings section
    emb = data["embeddings"]
    assert "total_public_repos" in emb
    assert "repos_with_embeddings" in emb
    assert "coverage_percent" in emb

    # Revision section
    rev = data["revision"]
    assert "api_version" in rev
    assert "build_number" in rev
