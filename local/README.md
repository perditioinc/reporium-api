# Reporium $0 local-OSS substrate

> **REPORIUM-$0-01** (epic [#5], card [#6]). Run `reporium-api` and its whole
> dependency stack **locally at $0** by substituting every paid GCP dependency
> with an OSS / emulator equivalent. **Additive and local-only** — this does not
> touch production, change any live cloud resource, or affect the paid
> deployment. The application code is **unchanged**.

## Why

`reporium-api` runs in production on paid cloud (Cloud Run + Cloud SQL + GCP
Pub/Sub + Memorystore + paid embeddings). For dev and ingestion we want the
exact same app running with **zero spend**. This `local/` stack does that.

## Cloud → OSS substitution map

| Paid (production)        | $0 local substitute                                   | How it's transparent |
|--------------------------|-------------------------------------------------------|----------------------|
| Cloud SQL (Postgres)     | `pgvector/pgvector:pg16` container                    | `DATABASE_URL` env points at it; migrations run via alembic |
| Memorystore (Redis)      | `redis:7-alpine` container                            | `REDIS_URL` env points at it |
| GCP Pub/Sub              | Pub/Sub **emulator** (`thekevjames/gcloud-pubsub-emulator`) | `PUBSUB_EMULATOR_HOST` — the unmodified `google-cloud-pubsub` client auto-routes to the emulator. **No code change.** |
| Cloud Run (api)          | local docker container                                | same image, same `Dockerfile` |
| Paid embedding API       | `sentence-transformers` `all-MiniLM-L6-v2`, baked into the api image, CPU-only, offline | already in the prod `Dockerfile`; $0 |
| Secret Manager           | plain env vars in `local/docker-compose.yml`          | never contacted (`ENVIRONMENT=development`) |

No cloud credentials, no secrets, no paid services are used or required.

## Quick start

From the **repo root**:

```bash
make up      # build + start everything, wait until healthy
make smoke   # up -> health-check every service -> down -v   (the proof)
make down    # stop + delete volumes
make ps      # service status
make logs    # tail logs
make seed    # (re)apply DB migrations  (api `up` already does this)
```

Or call the substrate Makefile directly: `make -f local/Makefile up`.

Once up:

| Service  | Address                          | Notes |
|----------|----------------------------------|-------|
| API      | http://localhost:8080/health     | OpenAPI docs at `/docs` |
| Postgres | localhost:5432                   | `postgres` / `postgres`, db `reporium` |
| Redis    | localhost:6379                   | |
| Pub/Sub  | localhost:8681                   | emulator, project `reporium-local`, topic `reporium-events` |

## What `make up` does

1. Starts **Postgres (pgvector)**, **Redis**, and the **Pub/Sub emulator**, each
   with a healthcheck.
2. Runs a one-shot **`migrate`** service (`alembic upgrade head`) against the
   local Postgres once it is healthy.
3. Starts the **api** only after the DB is migrated and all brokers are healthy,
   then waits for its `/health` endpoint to report `db: ok`.

`make smoke` additionally asserts the Pub/Sub emulator pre-created the
`reporium-events` topic and tears everything down with `-v`.

## Scope / what this is NOT

- This is the **$0-01** local dev substrate for `reporium-api`. Wiring the
  *other* suite repos (frontend, ingestion, mcp) into one compose, and the
  permanent cloud→OSS substitution + cost-cap / free-tier guardrails, are
  tracked separately (card **$0-02 / #7**).
- The Pub/Sub *emulator* is the local broker substitution. Swapping the
  production broker (or making the publisher pluggable for a non-GCP broker
  like NATS) is a code change owned by **#7**, not this card.
- **$0 / OSS only.** Any CI added for this must use the repo's self-hosted
  runner, never GitHub-hosted runners.

[#5]: https://github.com/perditioinc/reporium-system-design/issues/5
[#6]: https://github.com/perditioinc/reporium-system-design/issues/6
