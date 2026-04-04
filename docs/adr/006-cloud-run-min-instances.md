# ADR-006: Set min-instances=1 on reporium-api

## Status

Accepted

## Context

Reporium.com experiences multi-second cold start delays when the Cloud Run
API hasn't received traffic recently. Cold start measured at 1,580 ms for
`/health` and up to 42 s for `/library/full`. This creates unacceptable UX
for live demos and first-time visitors.

## Options Evaluated

1. **Static JSON hardcoding** — Rejected: compromises data freshness, creates maintenance burden.
2. **Vercel edge caching only** — Rejected: doesn't help for `/ask`, `/graph`, `/intelligence` endpoints.
3. **Cron ping every 5 min** — Rejected: unreliable, doesn't guarantee warm instance, adds complexity.
4. **min-instances=1** — Accepted: deterministic, zero application code changes, eliminates cold starts.

## Decision

Set `min-instances=1` on the reporium-api Cloud Run service.

## Cost

$51.80/month (1 vCPU, 2 GiB, us-central1, after free tier deduction).
Previous: $0.00/month. This is a conscious, justified cost increase for a
production-quality user experience.

## Latency Impact

| Endpoint | Before (min=0) | After (min=1) | Change |
|----------|----------------|----------------|--------|
| `/health` first request | 1,580 ms | 263 ms | -83% |
| `/health` warm avg (x5) | 573 ms | 252 ms | -56% |
| `/graph/edges` | 660 ms | 967 ms | ~same |
| `/intelligence/ask` | 1,336 ms | 1,436 ms | ~same |
| `/library/full` | 42,344 ms | 6,511 ms | -85% |

Benchmark captured 2026-03-31, 5 minutes after applying `min-instances=1`.

## Consequences

- Monthly infrastructure cost increases from $0 to ~$52.
- Cold starts are eliminated for the first concurrent user.
- Max 3 instances still limits cost during traffic spikes.
- Future optimization: 3-year CUD commitment would reduce to ~$28/month.

## Agent Alignment

`min-instances=1` is the established standard for reporium-api. Must not be
reverted without a documented team decision referencing this ADR.
