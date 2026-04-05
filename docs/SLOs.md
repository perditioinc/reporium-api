# Service Level Objectives — reporium-api

Status: **draft** (Phase 4 of the 2026-03 Reporium audit, tracked in issue #226)

This document defines the first formal SLOs for `reporium-api`. They are
deliberately loose enough that we are not paging on noise today, but tight
enough that a real regression shows up. The targets will tighten once we have
30 days of baseline data from Cloud Monitoring + the `/metrics/slo` endpoint.

## Targets

| # | Route / Signal | Objective | Window | Measured by |
|---|----------------|-----------|--------|-------------|
| 1 | `GET /health` | p95 latency < 500 ms, availability > 99.9% | 30 d | Cloud Run request metrics + [`app/main.py` `/health`](../app/main.py) (line ~263) |
| 2 | `GET /library/full` | p95 latency < 2 s (cached response), cache hit rate > 95% | 30 d | Cache metrics in [`app/routers/library_full.py`](../app/routers/library_full.py) (line ~1329) |
| 3 | `POST /intelligence/ask` | p95 latency < 15 s (warm), p99 < 25 s | 30 d | Request log latencies emitted by [`app/routers/intelligence.py`](../app/routers/intelligence.py) (`latency_ms` fields) |
| 4 | `POST /intelligence/nl-filter` | p95 latency < 3 s | 30 d | Request logs from [`app/routers/nl_filter.py`](../app/routers/nl_filter.py) (line ~141) |
| 5 | Error budget (all 5xx) | 5xx rate < 1% | 30 d | Cloud Run request metrics, broken out per route |
| 6 | Golden-set answer quality | score >= 0.7 on `tests/test_intelligence_quality.py` | per-PR | CI job in `.github/workflows/dev-test.yml` |

## Measurement

Until a full metrics backend is wired (Prometheus / Cloud Monitoring custom
metrics), live values come from two sources:

1. **Cloud Run built-in metrics** — request count, latency distribution, and
   5xx count per revision. Dashboards live under the `reporium-api` Cloud Run
   service in the `reporium-prod` GCP project.
2. **In-memory rolling histogram** — see [`app/slo_observer.py`](../app/slo_observer.py).
   The request-logging middleware in `app/main.py` records every request
   against a 24h rolling window keyed by route. Live values are exposed via
   `GET /metrics/slo`, which compares the current p50/p95/p99/error-rate for
   each tracked route against the targets in the `_SLO_TARGETS` dict in
   [`app/routers/platform.py`](../app/routers/platform.py).

   **Limitations**: single-process, single-instance. On Cloud Run with >1
   instance the values are per-instance only. Treat it as a smoke-level view,
   not a source of truth. A Prometheus / Cloud Monitoring backend is tracked
   separately (see Gaps below).

## Breach actions

| SLO | Breach threshold | Action |
|-----|------------------|--------|
| 1 — `/health` | Any 30-min window with p95 > 500 ms or availability < 99.9% | Page on-call via Cloud Monitoring alert policy. Check DB connection pool, Cloud Run revision status. |
| 2 — `/library/full` | p95 > 2 s or hit rate < 95% for 1 h | Slack `#reporium-alerts`. Investigate Redis health + cold cache after deploys. |
| 3 — `/intelligence/ask` | p95 > 15 s or p99 > 25 s for 1 h (warm traffic) | Slack `#reporium-alerts`. Check Anthropic API status, embedding model warm-up, DB pgvector query plan. |
| 4 — `/intelligence/nl-filter` | p95 > 3 s for 1 h | Slack `#reporium-alerts`. Check LLM fast-path latency, regex fallback rate. |
| 5 — Error budget | > 1% 5xx over rolling 30 d | Freeze feature deploys; reliability work takes priority until budget recovers. |
| 6 — Quality gate | Golden-set score < 0.7 on any PR | Block merge. The failing PR author investigates prompt / retrieval regression. |

## Sentry wiring — current state

**Status: NOT WIRED.** Issue [#24](https://github.com/perditioinc/reporium-api/issues/24)
was closed on 2026-03-24 with a comment stating *"Sentry initialized in
main.py; guarded by SENTRY_DSN env var"*, but as of this document:

- `sentry_sdk` does not appear in [`requirements.txt`](../requirements.txt).
- `sentry_sdk.init(...)` does not appear anywhere in `app/`.
- There is no `SENTRY_DSN` reference in the codebase or in `cloudbuild.yaml` /
  `deploy/`.

### What is needed to actually close #24

1. Add `sentry-sdk[fastapi]` to `requirements.txt`.
2. Initialize in `app/main.py` *before* `FastAPI(...)` is constructed:
   ```python
   import sentry_sdk
   from sentry_sdk.integrations.fastapi import FastApiIntegration
   from sentry_sdk.integrations.starlette import StarletteIntegration

   _sentry_dsn = os.environ.get("SENTRY_DSN")
   if _sentry_dsn:
       sentry_sdk.init(
           dsn=_sentry_dsn,
           environment=os.environ.get("ENVIRONMENT", "prod"),
           release=os.environ.get("APP_VERSION"),
           traces_sample_rate=0.1,
           profiles_sample_rate=0.1,
           integrations=[StarletteIntegration(), FastApiIntegration()],
       )
   ```
3. Plumb `SENTRY_DSN` through GCP Secret Manager (same pattern as
   `INGESTION_API_KEY` and `ANTHROPIC_API_KEY`) and expose it to the Cloud Run
   revision in `deploy/` / `cloudbuild.yaml`.
4. Configure a Cloud Monitoring uptime check on `/health` with a 2-failures
   alert policy pointing at on-call email + Slack webhook.

A separate PR will implement the above. This PR only documents the gap so
that the reopened issue has a clear definition of done.

## Golden-set quality gate — current state

Location: [`tests/test_intelligence_quality.py`](../tests/test_intelligence_quality.py)

**What the test suite asserts today:**

- Response structure: `answer`, `sources`, `model`, `tokens_used`,
  `question`, `answered_at` are all present; `tokens_used.total` equals
  input + output.
- `answer` is a non-empty string and matches the mocked Anthropic response
  text exactly.
- `sources` list is ordered by `relevance_score` descending.
- Each source has `name`, `owner`, `relevance_score` (float),
  `integration_tags` (list).
- `model` field contains the string "claude".
- Empty question → 422.
- No matching repos → still returns 200 with an answer.

**CI wiring:** `.github/workflows/dev-test.yml` runs `pytest tests/ -v` on
push-to-dev, and `.github/workflows/test.yml` runs it on PR-to-main. The
golden-set tests are part of `tests/` and run on every such trigger.

**Gap — quality scoring:** the suite does NOT yet compute a numeric quality
score, and there is no `score >= 0.7` assertion. It is a *structural* golden
set, not a *semantic* one. To close this gap we need to:

1. Add a small scored rubric (e.g. 0.4 for mentioning both langchain and
   llama_index, 0.3 for citing at least 2 sources, 0.3 for correct ordering).
2. Assert total score >= 0.7 and emit the score to the CI log for trending.
3. Optionally plug into an LLM-as-judge against a frozen reference answer.

This PR intentionally does **not** modify the test logic. The scoring rubric
should land as a follow-up once the rubric is reviewed.

**CI trigger gap:** both workflows currently run on `pull_request: branches
[main]`, not `[dev]`. PRs targeting `dev` (this one included) do not run the
quality gate. Adding `dev` to the PR trigger list is a 1-line change that
should ride along with the scoring rubric PR.

## Live endpoint — `GET /metrics/slo`

Returns a JSON document of the form:

```json
{
  "window_seconds": 86400,
  "source": "in_memory_histogram",
  "generated_at": "2026-04-04T12:34:56+00:00",
  "routes": {
    "/health": {
      "target": {"p95_ms": 500, "max_error_rate": 0.001},
      "observed": {"count": 1234, "p50_ms": 18.2, "p95_ms": 42.7, "p99_ms": 88.1, "error_rate": 0.0},
      "status": "ok",
      "breaches": []
    },
    ...
  }
}
```

`status` is one of `ok`, `breach`, or `no_data`. The endpoint is unauthenticated
(read-only, no PII). If that changes we should move it behind `verify_api_key`.

## Gaps (not in this PR)

- [ ] Sentry is not actually wired — #24 was closed prematurely.
- [ ] No Cloud Monitoring uptime check on `/health`.
- [ ] No alert policies — nothing pages on SLO breach today.
- [ ] `/metrics/slo` is per-instance; no cross-instance aggregation.
- [ ] Golden-set test has no numeric quality score or `>= 0.7` assertion.
- [ ] CI workflows do not run on PRs to `dev`.
- [ ] No 30-day baseline data yet — targets above may tighten once we have it.
