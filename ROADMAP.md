# reporium-api Roadmap

## Current State (March 2026)

`reporium-api` is the live backend for the Reporium suite.

- 17 Alembic migrations are present under `migrations/versions`
- PostgreSQL + pgvector power semantic search, repo similarity, taxonomy assignment, and portfolio analytics
- The taxonomy system is open-ended and database-backed across 8 dimensions:
  - `skill_area`
  - `industry`
  - `use_case`
  - `modality`
  - `ai_trend`
  - `deployment_context`
  - `tags`
  - `maturity_level`
- Quality signals are stored on repos and exposed to the frontend and MCP layers
- Pub/Sub push handling is live through `POST /ingest/events/repo-ingested`
- Webhook handling is live through `POST /webhooks/github`
- Ingestion run history is stored and exposed through `/admin/runs`
- MCP-facing query endpoints are live through the companion `reporium-mcp` server

## Recent Platform Additions

- Semantic repo search and repo-to-repo similarity on precomputed embeddings
- Dynamic taxonomy bootstrap, embedding, assignment, and deduplication admin flows
- Portfolio insights and cross-dimension analytics endpoints
- Protected admin and ingest surfaces using bearer auth plus `X-Admin-Key` / `X-Ingest-Key`
- Quality signal computation from stored repo metadata
- Ingest run recording for dashboard and operator visibility

## What Is Next

- Cloud deployment hardening for the ingestion pipeline that feeds this API
- Nightly enrichment and maintenance jobs running without manual triggers
- Scale the full platform from the current corpus to 10K repos
- Public query UI rate limiting and abuse protection refinement
- Commit-stat refresh automation so recent activity stays current without manual rebuilds
