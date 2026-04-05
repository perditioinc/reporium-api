"""Tests for the /metrics/slo endpoint and the underlying in-memory observer."""
import pytest

from app.slo_observer import SLOObserver, slo_observer


# ---------------------------------------------------------------------------
# Unit tests — SLOObserver
# ---------------------------------------------------------------------------


def test_observer_records_tracked_routes_only():
    obs = SLOObserver()
    obs.record("/health", 42.0, 200)
    obs.record("/some/untracked/route", 999.0, 200)

    snap = obs.snapshot()
    assert snap["/health"]["count"] == 1
    # Untracked routes must not appear in the snapshot at all
    assert "/some/untracked/route" not in snap


def test_observer_percentiles_and_error_rate():
    obs = SLOObserver()
    # 100 samples, latencies 1..100 ms, 5 of them 5xx
    for i in range(1, 101):
        status = 500 if i <= 5 else 200
        obs.record("/health", float(i), status)

    snap = obs.snapshot()["/health"]
    assert snap["count"] == 100
    # Nearest-rank p95 on [1..100] is ~95
    assert snap["p95_ms"] is not None
    assert 90 <= snap["p95_ms"] <= 100
    assert snap["error_rate"] == 0.05


def test_observer_empty_snapshot_has_stable_shape():
    obs = SLOObserver()
    snap = obs.snapshot()
    for route in ("/health", "/library/full", "/intelligence/ask", "/intelligence/nl-filter"):
        assert route in snap
        assert snap[route]["count"] == 0
        assert snap[route]["p95_ms"] is None
        assert snap[route]["error_rate"] is None


# ---------------------------------------------------------------------------
# Endpoint test — /metrics/slo
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_metrics_slo_endpoint_returns_expected_shape(client):
    # Reset the shared observer so this test is deterministic regardless of
    # test ordering.
    slo_observer.reset()

    # Inject a handful of samples so at least one route reports data.
    for latency in (100.0, 120.0, 140.0, 160.0, 200.0):
        slo_observer.record("/health", latency, 200)

    resp = await client.get("/metrics/slo")
    assert resp.status_code == 200
    data = resp.json()

    assert data["window_seconds"] == 24 * 60 * 60
    assert data["source"] == "in_memory_histogram"
    assert "generated_at" in data

    routes = data["routes"]
    for expected in ("/health", "/library/full", "/intelligence/ask", "/intelligence/nl-filter"):
        assert expected in routes
        entry = routes[expected]
        assert "target" in entry
        assert "observed" in entry
        assert "status" in entry
        assert "breaches" in entry

    # /health had 5 samples well under the 500ms target → ok
    health = routes["/health"]
    assert health["observed"]["count"] == 5
    assert health["status"] == "ok"
    assert health["breaches"] == []

    # Routes with no traffic must report no_data, not breach
    assert routes["/library/full"]["status"] == "no_data"


@pytest.mark.asyncio
async def test_metrics_slo_endpoint_flags_breaches(client):
    slo_observer.reset()

    # Record latencies well above the 500ms /health target
    for latency in (900.0, 1100.0, 1300.0, 1500.0, 1700.0):
        slo_observer.record("/health", latency, 200)

    resp = await client.get("/metrics/slo")
    assert resp.status_code == 200
    health = resp.json()["routes"]["/health"]
    assert health["status"] == "breach"
    assert any("p95" in b for b in health["breaches"])
