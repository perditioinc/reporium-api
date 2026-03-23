# Changelog

## [1.3.0] - 2026-03-22

### Security
- Fixed private repo exposure: 39 private repos were being returned by /library/full
- Added `is_private` column to repos table, backfilled from GitHub API
- Replaced hardcoded whitelist with proper `WHERE is_private = false` filter
- Added security unit tests to prevent regression
- Deleted all private repo data from database permanently

## [1.2.0] - 2026-03-21

### Changed
- Redis Memorystore cache enabled via VPC connector — /library/full cached 5 min, 2.4x faster responses
- /health now reports `"cache":"ok"` when Redis is connected
- Cloud Run revision 00011 with `--vpc-connector=forksync-connector` and `REDIS_URL` env var

## [1.1.0] - 2026-03-21

### Added
- `/library/full` endpoint — returns complete LibraryData matching frontend TypeScript interfaces (camelCase, nested objects, all 826 repos)
- `/intelligence/query` endpoint — semantic search + Claude-powered answers over the repo knowledge base ($0.01/query)
- SlowAPI rate limiting: 100/min public, 30/min search, 10/min ingest
- `X-RateLimit-Policy` header on all responses
- Scalar API docs at `/docs` with custom dark theme
- ANTHROPIC_API_KEY loaded from GCP Secret Manager with `\r\n` stripping
- sentence-transformers (all-MiniLM-L6-v2) for query embedding
- Dev branch workflow with PostgreSQL service container

### Changed
- Memory increased to 2Gi on Cloud Run for sentence-transformers model
- min-instances=1 to avoid cold start latency

## [1.0.0] - 2026-03-20

### Added
- Initial deployment to Cloud Run (826 repos, Neon PostgreSQL, pgvector)
- REST endpoints: /repos, /search, /stats, /health, /metrics/latest, /audit/status
- API key auth for ingest endpoints
- Alembic migrations for schema management
