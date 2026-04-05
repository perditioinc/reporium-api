"""
Lightweight in-memory SLO histogram used by the /metrics/slo endpoint.

This is intentionally minimal — no Prometheus, no external dependency.
It keeps a rolling 24h window of (timestamp, latency_ms, status_code) triples
per route and computes p50/p95/p99 + availability on demand.

When a proper metrics backend is wired up (Cloud Monitoring / Prometheus) this
module will be replaced. Until then it gives us a single-process view that is
good enough for dashboard-style debugging and smoke-level SLO tracking.
"""
from __future__ import annotations

import bisect
import threading
import time
from dataclasses import dataclass
from typing import Iterable

# Rolling window length. 24h keeps the memory footprint small for a service
# that does < 1M req/day (we are currently < 50k req/day).
WINDOW_SECONDS = 24 * 60 * 60

# Routes we actively track against documented SLOs. Anything else is ignored
# so we don't balloon memory with one-off paths (/openapi.json, /docs, etc.).
TRACKED_ROUTES: frozenset[str] = frozenset(
    {
        "/health",
        "/library/full",
        "/intelligence/ask",
        "/intelligence/nl-filter",
    }
)


@dataclass
class _Sample:
    __slots__ = ("ts", "latency_ms", "status")
    ts: float
    latency_ms: float
    status: int


class SLOObserver:
    """Thread-safe rolling histogram keyed by route."""

    def __init__(self, window_seconds: int = WINDOW_SECONDS) -> None:
        self._window = window_seconds
        self._samples: dict[str, list[_Sample]] = {}
        self._lock = threading.Lock()

    def record(self, route: str, latency_ms: float, status: int) -> None:
        if route not in TRACKED_ROUTES:
            return
        now = time.time()
        with self._lock:
            bucket = self._samples.setdefault(route, [])
            bucket.append(_Sample(now, latency_ms, status))
            self._evict_locked(bucket, now)

    def _evict_locked(self, bucket: list[_Sample], now: float) -> None:
        cutoff = now - self._window
        # bucket is append-only in time order, so binary search is safe.
        idx = bisect.bisect_left([s.ts for s in bucket], cutoff)
        if idx > 0:
            del bucket[:idx]

    def snapshot(self) -> dict[str, dict]:
        """
        Return a dict of `{route: {count, p50, p95, p99, error_rate}}`.
        Empty buckets are returned with null percentiles so the shape is stable.
        """
        now = time.time()
        out: dict[str, dict] = {}
        with self._lock:
            for route in TRACKED_ROUTES:
                bucket = self._samples.get(route, [])
                self._evict_locked(bucket, now)
                out[route] = _summarize(bucket)
        return out

    def reset(self) -> None:
        """Testing helper — clears all buckets."""
        with self._lock:
            self._samples.clear()


def _percentile(sorted_values: list[float], pct: float) -> float | None:
    if not sorted_values:
        return None
    # Nearest-rank percentile — good enough for dashboard-level reporting.
    k = max(0, min(len(sorted_values) - 1, int(round(pct / 100.0 * (len(sorted_values) - 1)))))
    return round(sorted_values[k], 2)


def _summarize(samples: Iterable[_Sample]) -> dict:
    samples = list(samples)
    count = len(samples)
    if count == 0:
        return {
            "count": 0,
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "error_rate": None,
        }
    latencies = sorted(s.latency_ms for s in samples)
    errors = sum(1 for s in samples if s.status >= 500)
    return {
        "count": count,
        "p50_ms": _percentile(latencies, 50),
        "p95_ms": _percentile(latencies, 95),
        "p99_ms": _percentile(latencies, 99),
        "error_rate": round(errors / count, 4),
    }


# Module-level singleton used by the request-logging middleware + the
# /metrics/slo endpoint. Tests can call `slo_observer.reset()` between cases.
slo_observer = SLOObserver()


# ---------------------------------------------------------------------------
# KAN-ask-spend: Token / cost observer
# ---------------------------------------------------------------------------
#
# Parallel in-memory accumulator to SLOObserver — tracks LLM token spend per
# route over a 24h rolling window. Same design principles:
#   * single-process, in-memory, $0 infra
#   * thread-safe (single lock like SLOObserver)
#   * replaced by a proper metrics backend once we have one
#
# It also tracks cache hits per route (smart-route / Redis / semantic cache)
# so /metrics/spend can report a cache hit-rate, which is the single most
# useful cost-efficiency number we currently lack.


@dataclass
class _SpendSample:
    __slots__ = ("ts", "input_tokens", "output_tokens", "usd", "model")
    ts: float
    input_tokens: int
    output_tokens: int
    usd: float
    model: str


@dataclass
class _CacheSample:
    __slots__ = ("ts",)
    ts: float


class TokenSpendObserver:
    """
    Thread-safe rolling 24h accumulator for LLM token spend per route.

    Unlike SLOObserver we don't restrict to TRACKED_ROUTES — any route that
    actually spends tokens is worth tracking. In practice this is 2-3 routes
    (/intelligence/ask, /intelligence/query, /intelligence/nl-filter).
    """

    def __init__(self, window_seconds: int = WINDOW_SECONDS) -> None:
        self._window = window_seconds
        self._spend: dict[str, list[_SpendSample]] = {}
        self._cache_hits: dict[str, list[_CacheSample]] = {}
        self._lock = threading.Lock()

    # -- recording ----------------------------------------------------------

    def record_tokens(
        self,
        route: str,
        input_tokens: int,
        output_tokens: int,
        usd_cost: float,
        model: str,
    ) -> None:
        now = time.time()
        with self._lock:
            bucket = self._spend.setdefault(route, [])
            bucket.append(
                _SpendSample(
                    ts=now,
                    input_tokens=int(input_tokens or 0),
                    output_tokens=int(output_tokens or 0),
                    usd=float(usd_cost or 0.0),
                    model=model or "unknown",
                )
            )
            self._evict_spend_locked(bucket, now)

    def record_cache_hit(self, route: str) -> None:
        now = time.time()
        with self._lock:
            bucket = self._cache_hits.setdefault(route, [])
            bucket.append(_CacheSample(now))
            self._evict_cache_locked(bucket, now)

    # -- eviction -----------------------------------------------------------

    def _evict_spend_locked(self, bucket: list[_SpendSample], now: float) -> None:
        cutoff = now - self._window
        idx = bisect.bisect_left([s.ts for s in bucket], cutoff)
        if idx > 0:
            del bucket[:idx]

    def _evict_cache_locked(self, bucket: list[_CacheSample], now: float) -> None:
        cutoff = now - self._window
        idx = bisect.bisect_left([s.ts for s in bucket], cutoff)
        if idx > 0:
            del bucket[:idx]

    # -- snapshot -----------------------------------------------------------

    def get_spend_snapshot(self) -> dict:
        """
        Return a dict shaped like:
            {
              "routes": {
                 "<route>": {
                    "input_tokens": int,
                    "output_tokens": int,
                    "usd": float,
                    "requests": int,
                    "cache_hits": int,
                    "cache_hit_rate": float,  # hits / (hits + requests)
                 },
                 ...
              },
              "total": { same shape, aggregated across routes },
            }
        """
        now = time.time()
        routes_out: dict[str, dict] = {}
        tot_in = 0
        tot_out = 0
        tot_usd = 0.0
        tot_reqs = 0
        tot_hits = 0

        with self._lock:
            # Union of all routes we've ever seen (either spend or cache).
            seen_routes = set(self._spend.keys()) | set(self._cache_hits.keys())
            for route in seen_routes:
                spend_bucket = self._spend.get(route, [])
                cache_bucket = self._cache_hits.get(route, [])
                self._evict_spend_locked(spend_bucket, now)
                self._evict_cache_locked(cache_bucket, now)

                r_in = sum(s.input_tokens for s in spend_bucket)
                r_out = sum(s.output_tokens for s in spend_bucket)
                r_usd = sum(s.usd for s in spend_bucket)
                r_reqs = len(spend_bucket)
                r_hits = len(cache_bucket)
                denom = r_reqs + r_hits
                hit_rate = (r_hits / denom) if denom > 0 else 0.0

                routes_out[route] = {
                    "input_tokens": r_in,
                    "output_tokens": r_out,
                    "usd": round(r_usd, 6),
                    "requests": r_reqs,
                    "cache_hits": r_hits,
                    "cache_hit_rate": round(hit_rate, 4),
                }
                tot_in += r_in
                tot_out += r_out
                tot_usd += r_usd
                tot_reqs += r_reqs
                tot_hits += r_hits

        total_denom = tot_reqs + tot_hits
        total_hit_rate = (tot_hits / total_denom) if total_denom > 0 else 0.0
        return {
            "routes": routes_out,
            "total": {
                "input_tokens": tot_in,
                "output_tokens": tot_out,
                "usd": round(tot_usd, 6),
                "requests": tot_reqs,
                "cache_hits": tot_hits,
                "cache_hit_rate": round(total_hit_rate, 4),
            },
        }

    def reset(self) -> None:
        """Testing helper — clears all buckets."""
        with self._lock:
            self._spend.clear()
            self._cache_hits.clear()


# Module-level singleton. Like slo_observer, tests can call .reset() between cases.
token_observer = TokenSpendObserver()
