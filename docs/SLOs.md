# Service Level Objectives - reporium-api

Status: draft

This document defines the current latency and availability objectives for the
read-heavy API surfaces that matter most after the dependency backfill and
knowledge graph rollout.

## Targets

| Route / Signal | Objective | Window | Measured by |
|---|---|---|---|
| `GET /health` | p95 < 500 ms, p99 < 1 s, availability > 99.9% | 30 d | Cloud Run + `/metrics/latency` |
| `GET /stats` | p95 < 200 ms, p99 < 500 ms | 30 d | Prometheus histogram + `/metrics/latency` |
| `GET /library` | p95 < 750 ms, p99 < 1.5 s | 30 d | Prometheus histogram + `/metrics/latency` |
| `GET /library/full` | p95 < 2 s, p99 < 4 s | 30 d | Prometheus histogram + `/metrics/latency` |
| `GET /graph/edges` | p95 < 200 ms, p99 < 500 ms | 30 d | Prometheus histogram + `/metrics/latency` |
| `GET /graph/edges/search` | p95 < 1.5 s, p99 < 3 s | 30 d | Prometheus histogram + `/metrics/latency` |
| `POST /intelligence/ask` | p95 < 15 s, p99 < 25 s | 30 d | Prometheus histogram + `/metrics/latency` |
| `POST /intelligence/nl-filter` | p95 < 3 s, p99 < 5 s | 30 d | Prometheus histogram + `/metrics/latency` |
| Error budget | 5xx rate < 1% | 30 d | Cloud Run + Prometheus counters |

## Measurement

The service now exposes three complementary observability layers:

1. Cloud Run built-in metrics for request counts, revision health, and 5xxs.
2. `GET /metrics/prometheus` for Prometheus/Grafana scraping of route-level
   histograms and request counters.
3. `GET /metrics/latency` and `GET /metrics/slo` for a lightweight JSON view
   derived from the in-memory rolling observer in `app/slo_observer.py`.

The JSON endpoints are intentionally operator-friendly and easy to inspect in
smoke tests, but they remain single-instance views. Prometheus is the source
for multi-instance dashboards and percentile panels.

## New metrics surfaces

| Endpoint | Purpose |
|---|---|
| `GET /metrics/latency` | p50/p95/p99 and error rate for tracked routes |
| `GET /metrics/slo` | Same latency data with SLO breach evaluation and spend summary |
| `GET /metrics/backfill` | Dependency backfill coverage, throughput, and ETA proxy |
| `GET /metrics/graph-quality` | DEPENDS_ON exact precision/recall plus edge coverage proxies |
| `GET /metrics/prometheus` | Prometheus exposition for Grafana |

## Operational notes

- `DEPENDS_ON` quality is measured exactly against the current
  `repo_dependencies` corpus.
- `ALTERNATIVE_TO`, `COMPATIBLE_WITH`, and `EXTENDS` currently use operational
  proxy metrics instead of human-labeled truth sets.
- Backfill ETA is an estimate based on recent `repo_dependencies.fetched_at`
  throughput and is most accurate when zero-dependency repos write sentinel
  rows.
