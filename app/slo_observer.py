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
