"""Prometheus-friendly request metrics for reporium-api."""

from __future__ import annotations

from collections import defaultdict

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

    HTTP_REQUESTS_TOTAL = Counter(
        "reporium_http_requests_total",
        "Total HTTP requests served by reporium-api.",
        ("route", "method", "status_code"),
    )

    HTTP_REQUEST_DURATION_SECONDS = Histogram(
        "reporium_http_request_duration_seconds",
        "HTTP request latency for reporium-api.",
        ("route", "method"),
        buckets=(0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
    )

    # Observability for the operator-runbook backfill endpoint
    # (POST /admin/backfill/primary_category_column). See KAN-API-OBS-BACKFILL.
    ADMIN_BACKFILL_RUNS_TOTAL = Counter(
        "admin_backfill_runs_total",
        "Total invocations of /admin/backfill/primary_category_column by terminal outcome.",
        ("outcome",),
    )

    ADMIN_BACKFILL_DURATION_SECONDS = Histogram(
        "admin_backfill_duration_seconds",
        "End-to-end duration of /admin/backfill/primary_category_column in seconds.",
        buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
    )

    def _render_latest() -> bytes:
        return generate_latest()

except ModuleNotFoundError:
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"

    class _MetricHandle:
        def __init__(self, sink: dict[tuple[tuple[str, str], ...], float], labels: dict[str, str]) -> None:
            self._sink = sink
            self._labels = tuple(sorted(labels.items()))

        def inc(self, amount: float = 1.0) -> None:
            self._sink[self._labels] += amount

        def observe(self, amount: float) -> None:
            self._sink[self._labels] += amount

    class _FallbackMetric:
        def __init__(self, name: str) -> None:
            self.name = name
            self.values: dict[tuple[tuple[str, str], ...], float] = defaultdict(float)

        def labels(self, **labels: str) -> _MetricHandle:
            return _MetricHandle(self.values, labels)

        def inc(self, amount: float = 1.0) -> None:
            # Unlabeled increment falls back to a single bucket keyed by ().
            self.values[tuple()] += amount

        def observe(self, amount: float) -> None:
            self.values[tuple()] += amount

    HTTP_REQUESTS_TOTAL = _FallbackMetric("reporium_http_requests_total")
    HTTP_REQUEST_DURATION_SECONDS = _FallbackMetric("reporium_http_request_duration_seconds")
    ADMIN_BACKFILL_RUNS_TOTAL = _FallbackMetric("admin_backfill_runs_total")
    ADMIN_BACKFILL_DURATION_SECONDS = _FallbackMetric("admin_backfill_duration_seconds")

    def _render_lines(metric: _FallbackMetric) -> list[str]:
        lines = [
            f"# HELP {metric.name} fallback metric",
            f"# TYPE {metric.name} gauge",
        ]
        for labels, value in metric.values.items():
            label_str = ",".join(f'{key}="{val}"' for key, val in labels)
            lines.append(f"{metric.name}{{{label_str}}} {value}")
        return lines

    def _render_latest() -> bytes:
        lines = _render_lines(HTTP_REQUESTS_TOTAL)
        lines.extend(_render_lines(HTTP_REQUEST_DURATION_SECONDS))
        lines.extend(_render_lines(ADMIN_BACKFILL_RUNS_TOTAL))
        lines.extend(_render_lines(ADMIN_BACKFILL_DURATION_SECONDS))
        lines.append("")
        return "\n".join(lines).encode("utf-8")


def normalize_route(path: str) -> str:
    """Collapse dynamic paths to low-cardinality labels for Prometheus."""
    if path.startswith("/repos/") and path.count("/") == 2:
        return "/repos/{name}"
    if path.startswith("/graph/subgraph/") and path.count("/") == 3:
        return "/graph/subgraph/{repo_name}"
    return path


def record_http_request(*, path: str, method: str, status_code: int, duration_ms: float) -> None:
    route = normalize_route(path)
    HTTP_REQUESTS_TOTAL.labels(
        route=route,
        method=method.upper(),
        status_code=str(status_code),
    ).inc()
    HTTP_REQUEST_DURATION_SECONDS.labels(
        route=route,
        method=method.upper(),
    ).observe(max(duration_ms, 0.0) / 1000.0)


def render_latest_metrics() -> tuple[bytes, str]:
    """Return Prometheus exposition payload and content type."""
    return _render_latest(), CONTENT_TYPE_LATEST
