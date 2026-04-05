# Cloud Run Sizing (reporium-api)

## Current config

Set in `.github/workflows/deploy.yml`:

```
--memory=2Gi
--cpu=1
--min-instances=0
--max-instances=10
--concurrency=200
--timeout=60
```

Theoretical max concurrent requests: `max-instances * concurrency = 10 * 200 = 2000`
(previously `3 * 20 = 60`, so ~33x headroom; conservatively ~10x real throughput).

## Why concurrency=200 is safe

The hot path in this service is `/intelligence/ask` and related LLM endpoints,
which spend **5–15 s p95 inside `await anthropic.messages.create(...)`**. That
time is pure IO wait — the Python process is blocked on a socket, not burning
CPU. While one coroutine is awaiting Anthropic, the event loop happily services
hundreds of others on the same instance.

FastAPI + uvicorn on a single CPU can comfortably multiplex 200 concurrent
in-flight requests when the per-request CPU cost is small (JSON parse, DB
fetch, response serialization — all sub-10 ms). Memory stays flat because each
awaiting request holds only a small request object + the pending HTTPX future,
not a whole thread stack.

The previous `concurrency=20` was the Cloud Run default and was sized for
CPU-bound workloads. It was leaving ~90% of each instance's event-loop capacity
on the floor.

## Cost model: still $0/month idle

- `min-instances=0` → Cloud Run scales to zero when there is no traffic, so
  **idle cost remains $0**.
- Billing is per request-second of active container time. A wider concurrency
  means **fewer instances are spun up for the same traffic**, which is
  strictly cheaper, not more expensive.
- `max-instances=10` caps the blast radius: worst case we burn 10 instances
  worth of request-seconds, same unit cost as before.
- No new GCP resources are added. No memory/CPU bump. No min-instances.

## Observability

Watch the effect of this change via:

- `/metrics/slo` — in-app SLO endpoint (latency histograms, error rate)
- Cloud Run console → `reporium-api` → Metrics: `container/cpu/utilizations`,
  `request_count`, `instance_count`, `request_latencies`
- Alert if `container/cpu/utilizations` p95 > 0.8 sustained — that would mean
  the IO-bound assumption has broken (e.g. someone added a CPU-heavy codepath).

## How to revert

One-line rollback without a redeploy:

```
gcloud run services update reporium-api --region=us-central1 --concurrency=20 --max-instances=3
```

Then open a PR reverting this change in `deploy.yml` so the next deploy does
not re-apply the wider values.
