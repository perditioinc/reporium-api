# reporium-api
<!-- perditio-badges-start -->
[![Tests](https://github.com/perditioinc/reporium-api/actions/workflows/test.yml/badge.svg)](https://github.com/perditioinc/reporium-api/actions/workflows/test.yml)
![Last Commit](https://img.shields.io/github/last-commit/perditioinc/reporium-api)
![License](https://img.shields.io/github/license/perditioinc/reporium-api)
![python](https://img.shields.io/badge/python-3.11%2B-3776ab)
![suite](https://img.shields.io/badge/suite-Reporium-6e40c9)
![repos](https://img.shields.io/badge/repos-826-blue)
![deployed](https://img.shields.io/badge/deployed-Cloud%20Run-blue)
![docs](https://img.shields.io/badge/docs-%2Fdocs-blue)
<!-- perditio-badges-end -->

Backend API for Reporium. Handles all data reads and writes.
Public reads, authorized writes only.

## Quick Start (local)

```bash
docker-compose up
```

API available at http://localhost:8000
Docs at http://localhost:8000/docs

## Manual Setup

```bash
cp .env.example .env
# Edit .env with your values

pip install -r requirements.txt

# Run migrations
alembic upgrade head

# Start server
uvicorn app.main:app --reload
```

## Migrate from library.json

```bash
python scripts/migrate_from_json.py --json-path ../reporium/public/data/library.json
```

## Environment Variables

See `.env.example` for all options.

| Variable | Required | Description |
|---|---|---|
| `DATABASE_URL` | Yes | `postgresql+asyncpg://...` |
| `DATABASE_PROVIDER` | No | `local` \| `supabase` \| `gcp` |
| `INGESTION_API_KEY` | Yes | Secret key for write endpoints |
| `REDIS_URL` | No | Redis connection string — API works without it |
| `GH_USERNAME` | No | GitHub username |
| `GH_TOKEN` | No | GitHub token for direct API calls |
| `ENVIRONMENT` | No | `development` \| `production` |

## Deploy to Supabase + Railway

1. Create a Supabase project — copy the connection string
2. Set `DATABASE_URL` to your Supabase connection string
3. Set `DATABASE_PROVIDER=supabase`
4. Deploy to Railway, set all env vars
5. Run `alembic upgrade head` via Railway shell

## Deploy to GCP Cloud Run

1. Create a Cloud SQL PostgreSQL instance with pgvector enabled
2. Set `DATABASE_URL=postgresql+asyncpg://[user]:[pass]@/[db]?host=/cloudsql/[connection-name]`
3. Set `DATABASE_PROVIDER=gcp`
4. Build and push Docker image to Artifact Registry
5. Deploy to Cloud Run with the Cloud SQL connection

## Auth

Write endpoints require:
```
Authorization: Bearer {INGESTION_API_KEY}
```

## Endpoints

### Public
- `GET /health` — health check
- `GET /library` — full library data (cached 5 min)
- `GET /repos` — filterable repo list
- `GET /repos/{name}` — repo detail (cached 1 hr)
- `GET /search?q=` — text/semantic search
- `GET /trends` — latest trend signals
- `GET /gaps` — gap analysis
- `GET /stats` — library statistics
- `GET /wiki/skills/{skill}` — skill wiki page
- `GET /wiki/categories/{category}` — category wiki page

### Authorized (Bearer token required)
- `POST /ingest/repos` — batch upsert repos
- `POST /ingest/repos/{name}/enrich` — update enriched fields
- `POST /ingest/trends/snapshot` — commit trend snapshot
- `POST /ingest/gaps` — update gap analysis
- `POST /ingest/log` — update ingestion log
