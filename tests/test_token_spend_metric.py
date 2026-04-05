"""
Tests for the token-spend observer, /metrics/spend endpoint, and budget
guardrail added in KAN-ask-spend.
"""
from __future__ import annotations

import threading

import pytest

from app.slo_observer import TokenSpendObserver, token_observer


# ---------------------------------------------------------------------------
# Unit tests — TokenSpendObserver
# ---------------------------------------------------------------------------


def test_observer_records_tokens_and_totals():
    obs = TokenSpendObserver()
    obs.record_tokens(
        route="/intelligence/ask",
        input_tokens=1000,
        output_tokens=200,
        usd_cost=0.012,
        model="claude-sonnet-4-20250514",
    )
    obs.record_tokens(
        route="/intelligence/ask",
        input_tokens=500,
        output_tokens=100,
        usd_cost=0.006,
        model="claude-sonnet-4-20250514",
    )

    snap = obs.get_spend_snapshot()
    route = snap["routes"]["/intelligence/ask"]
    assert route["input_tokens"] == 1500
    assert route["output_tokens"] == 300
    assert route["requests"] == 2
    assert route["usd"] == pytest.approx(0.018, rel=1e-6)

    total = snap["total"]
    assert total["input_tokens"] == 1500
    assert total["output_tokens"] == 300
    assert total["requests"] == 2
    assert total["usd"] == pytest.approx(0.018, rel=1e-6)


def test_observer_aggregates_across_routes():
    obs = TokenSpendObserver()
    obs.record_tokens("/intelligence/ask", 100, 50, 0.005, "claude-sonnet-4-20250514")
    obs.record_tokens("/intelligence/nl-filter", 200, 20, 0.0004, "claude-haiku-4-5")

    snap = obs.get_spend_snapshot()
    assert snap["total"]["requests"] == 2
    assert snap["total"]["input_tokens"] == 300
    assert snap["total"]["output_tokens"] == 70
    assert snap["total"]["usd"] == pytest.approx(0.0054, rel=1e-6)
    assert set(snap["routes"].keys()) == {"/intelligence/ask", "/intelligence/nl-filter"}


def test_cache_hit_rate_math():
    obs = TokenSpendObserver()
    # 3 real Claude calls, 1 cache hit -> hit rate 1/4 = 0.25
    for _ in range(3):
        obs.record_tokens("/intelligence/ask", 100, 50, 0.005, "claude-sonnet-4-20250514")
    obs.record_cache_hit("/intelligence/ask")

    snap = obs.get_spend_snapshot()
    route = snap["routes"]["/intelligence/ask"]
    assert route["cache_hits"] == 1
    assert route["requests"] == 3
    assert route["cache_hit_rate"] == pytest.approx(0.25, rel=1e-6)
    assert snap["total"]["cache_hit_rate"] == pytest.approx(0.25, rel=1e-6)


def test_cache_hit_rate_zero_traffic():
    obs = TokenSpendObserver()
    snap = obs.get_spend_snapshot()
    assert snap["total"]["cache_hit_rate"] == 0.0
    assert snap["routes"] == {}


def test_observer_thread_safety_concurrent_record_tokens():
    obs = TokenSpendObserver()
    threads = []
    per_thread = 200
    n_threads = 10

    def worker():
        for _ in range(per_thread):
            obs.record_tokens(
                "/intelligence/ask",
                input_tokens=10,
                output_tokens=5,
                usd_cost=0.0001,
                model="claude-sonnet-4-20250514",
            )
            obs.record_cache_hit("/intelligence/ask")

    for _ in range(n_threads):
        t = threading.Thread(target=worker)
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    snap = obs.get_spend_snapshot()
    route = snap["routes"]["/intelligence/ask"]
    expected_reqs = per_thread * n_threads
    assert route["requests"] == expected_reqs
    assert route["cache_hits"] == expected_reqs
    assert route["input_tokens"] == expected_reqs * 10
    assert route["output_tokens"] == expected_reqs * 5
    assert route["usd"] == pytest.approx(expected_reqs * 0.0001, rel=1e-6)


def test_observer_reset():
    obs = TokenSpendObserver()
    obs.record_tokens("/intelligence/ask", 100, 50, 0.005, "claude-sonnet-4-20250514")
    obs.record_cache_hit("/intelligence/ask")
    obs.reset()
    snap = obs.get_spend_snapshot()
    assert snap["routes"] == {}
    assert snap["total"]["requests"] == 0
    assert snap["total"]["cache_hits"] == 0


# ---------------------------------------------------------------------------
# Budget status transitions
# ---------------------------------------------------------------------------


def test_spend_status_thresholds():
    from app.routers.platform import _spend_status

    budget = 10.0
    # Under 80% -> ok
    assert _spend_status(0.0, budget) == "ok"
    assert _spend_status(5.0, budget) == "ok"
    assert _spend_status(7.99, budget) == "ok"
    # 80% - 100% -> warning
    assert _spend_status(8.0, budget) == "warning"
    assert _spend_status(9.5, budget) == "warning"
    # >= 100% -> breach
    assert _spend_status(10.0, budget) == "breach"
    assert _spend_status(15.0, budget) == "breach"


def test_spend_status_zero_budget_is_ok():
    from app.routers.platform import _spend_status

    # A misconfigured zero budget should not crash the endpoint.
    assert _spend_status(100.0, 0.0) == "ok"


# ---------------------------------------------------------------------------
# Endpoint tests — /metrics/spend
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_metrics_spend_endpoint_shape(client):
    token_observer.reset()

    token_observer.record_tokens(
        route="/intelligence/ask",
        input_tokens=1000,
        output_tokens=200,
        usd_cost=0.012,
        model="claude-sonnet-4-20250514",
    )
    token_observer.record_cache_hit("/intelligence/ask")

    resp = await client.get("/metrics/spend")
    assert resp.status_code == 200
    data = resp.json()

    assert data["window_seconds"] == 24 * 60 * 60
    assert data["source"] == "in_memory_accumulator"
    assert "generated_at" in data
    assert "daily_budget_usd" in data
    assert isinstance(data["daily_budget_usd"], float)

    total = data["total"]
    for key in ("input_tokens", "output_tokens", "usd", "requests", "cache_hits", "cache_hit_rate"):
        assert key in total
    assert total["input_tokens"] == 1000
    assert total["output_tokens"] == 200
    assert total["requests"] == 1
    assert total["cache_hits"] == 1
    assert total["cache_hit_rate"] == pytest.approx(0.5, rel=1e-6)

    assert "/intelligence/ask" in data["routes"]
    assert data["status"] in {"ok", "warning", "breach"}


@pytest.mark.asyncio
async def test_metrics_spend_status_ok(client):
    token_observer.reset()
    # Way under default $10 budget
    token_observer.record_tokens(
        "/intelligence/ask", 100, 20, 0.001, "claude-sonnet-4-20250514"
    )
    resp = await client.get("/metrics/spend")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_metrics_spend_status_warning(client):
    from app.config import settings

    token_observer.reset()
    # Push spend to ~85% of budget
    token_observer.record_tokens(
        "/intelligence/ask",
        input_tokens=1,
        output_tokens=1,
        usd_cost=settings.spend_daily_budget_usd * 0.85,
        model="claude-sonnet-4-20250514",
    )
    resp = await client.get("/metrics/spend")
    assert resp.status_code == 200
    assert resp.json()["status"] == "warning"


@pytest.mark.asyncio
async def test_metrics_spend_status_breach(client):
    from app.config import settings

    token_observer.reset()
    # Push spend above budget
    token_observer.record_tokens(
        "/intelligence/ask",
        input_tokens=1,
        output_tokens=1,
        usd_cost=settings.spend_daily_budget_usd * 1.25,
        model="claude-sonnet-4-20250514",
    )
    resp = await client.get("/metrics/spend")
    assert resp.status_code == 200
    assert resp.json()["status"] == "breach"


@pytest.mark.asyncio
async def test_metrics_slo_includes_spend_summary(client):
    """Grafana-style dashboards pulling /metrics/slo should get cost info for free."""
    token_observer.reset()
    token_observer.record_tokens(
        "/intelligence/ask", 100, 20, 0.003, "claude-sonnet-4-20250514"
    )
    token_observer.record_cache_hit("/intelligence/ask")

    resp = await client.get("/metrics/slo")
    assert resp.status_code == 200
    data = resp.json()
    assert "spend_summary" in data
    summary = data["spend_summary"]
    assert summary["usd_24h"] == pytest.approx(0.003, rel=1e-6)
    assert summary["cache_hit_rate"] == pytest.approx(0.5, rel=1e-6)
    assert summary["status"] in {"ok", "warning", "breach"}
