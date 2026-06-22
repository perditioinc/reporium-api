# Changelog

## [Unreleased] - 2026-04-08

### Added
- **$0 local-OSS substrate** (`local/`, `Makefile`, `scripts/smoke.sh`): runs the full api stack locally at zero cloud spend by substituting paid GCP deps with OSS equivalents (Cloud SQL → pgvector, Memorystore → redis, GCP Pub/Sub → Pub/Sub emulator as the local broker, paid embeddings → baked-in sentence-transformers). `make up | down | smoke | seed`. App code unchanged; Pub/Sub substituted via `PUBSUB_EMULATOR_HOST` (no code change). Smoke proves up → healthy → `/health` db=ok → `down -v`. (REPORIUM-$0-01, system-design#6)
- **Similar repos fallback**: If `/intelligence/similar/{owner}/{name}` returns zero candidates (repo has no categories or tags), falls back to surfacing repos sharing the same `primary_category`, ordered by star count.
- **Edge type colors in 3D graph**: Knowledge graph edges now rendered in distinct colors per relationship type (amber=ALTERNATIVE_TO, green=COMPATIBLE_WITH, blue=DEPENDS_ON, violet=SIMILAR_TO, pink=EXTENDS).
- **`/graph/edges` now returns typed relationship edges**: After fetching pgvector similarity edges, also queries the `repo_edges` table for ALTERNATIVE_TO, COMPATIBLE_WITH, DEPENDS_ON, EXTENDS edges. Typed edges override SIMILAR_TO when the same (source, target) pair exists. Degrades gracefully if `repo_edges` table is missing. `edgeTypes` field in response now lists all distinct types present in the data.

### Fixed
- **Alembic clean-DB `upgrade head` on `dev`**: ported `migrations/env.py` to `transaction_per_migration=True` + `engine.connect()` (already present on `main`), so migration 038's `autocommit_block()` (CREATE INDEX CONCURRENTLY) no longer raises `AssertionError` on a fresh database. Restores dev↔main parity; required for the local substrate to migrate a clean DB.

### Changed
- **pgvector retrieval threshold**: Lowered from 0.45 → 0.40 in the adaptive top_k filter to surface more relevant repos per ask query.

## [2.0.0] - 2026-04-06

### Security
- **Off-topic domain boundary filter** — Pre-Claude regex filter rejects math, coding exercises,
  trivia, recipes, creative writing, medical/legal/financial advice, and utility tasks at $0 cost.
  Repo-signal override prevents false positives (AI/ML keywords bypass the filter).
- **System prompt hardened** with explicit domain boundary rules and strict off-topic refusal.
- **Prompt injection defense** consolidated to single compressed security instruction (~40 tokens saved).

### Added
- **Governance gateway** (`app/governance.py`): per-key rate limiting (Redis sliding window),
  per-key daily budget enforcement, auditable sandbox via `X-Sandbox: true` header.
- **Audit trail** (`app/models/audit_log.py`, migration 025): `audit_logs` table with
  `GET /admin/audit` endpoint for paginated, filterable audit log access.
- **OpenTelemetry + Cloud Trace** integration (`app/telemetry.py`): HTTP, Claude API, pgvector,
  Redis, and embedding spans. Feature-flagged via `OTEL_ENABLED=0`.
- **Golden-set expansion**: 18 → 53 test cases covering negation, temporal, multi-category,
  ambiguous, out-of-domain, injection edge cases, session follow-ups, and empty results.
- **Off-topic filter tests**: 56 unit tests covering rejected, allowed, and edge cases.
- **Graph optimization**: Redis caching (1hr TTL), semantic subgraph search at
  `GET /graph/edges/search`, embedding coverage metrics at `GET /metrics/embeddings`.
- **Source block compaction**: XML `<repo>` tags → compact numbered format (~165 tokens/request saved).
- **PII redaction** wired into `_log_query()` before DB insert.
- **Early-exit helper** `_should_early_exit()` extracted from duplicated similarity checks.
- **Enterprise roadmap** documentation at `docs/enterprise-roadmap.md`.

### Changed
- System prompt compressed from ~5 security restatements to 1 clear rule + domain boundary.
- Session history capping deduplicated — single `_MAX_SESSION_HISTORY_CHARS` constant.
- Dead `cosine_similarity()` function and unused `get_anthropic_key` import removed.
- Stale TODO comments updated to reflect implemented retention system.

### Fixed
- 20 stale injection tests updated to match current log-only defense strategy.
- `ask_sessions(created_at)` btree index added (migration 023) for retention purge performance.
- `repo_embeddings(repo_id)` btree index added (migration 024) for graph query performance.

## [Unreleased] - 2026-03-24

### Added
- Open-issues count support in schema, ingest, and `/library/full` responses.
- Dynamic taxonomy administration endpoints for rebuild, embedding, assignment, and noise-tag pruning.
- Semantic cache support for `/intelligence/ask` using stored query embeddings and cached answers.
- Cross-dimension analytics at `GET /analytics/cross-dimension`.
- Proactive portfolio insights feed at `GET /intelligence/portfolio-insights`.
- Pub/Sub-triggered ingestion refresh hook at `POST /ingest/events/repo-ingested`.
- Semantic repo search at `GET /search/semantic` using pgvector cosine similarity.

### Changed
- Taxonomy is now managed as an open, embedding-assigned system rather than fixed frontend-only skill lists.
- API docs and admin surfaces now cover taxonomy, analytics, intelligence refresh, and semantic search flows.

### Fixed
- Pytest collection no longer crashes on the FastAPI/Starlette router import mismatch.
- Alembic migration chain conflict resolved across the 010/011/012 feature wave.

## [1.5.0] - 2026-03-23

### Security
- **Prompt injection prevention** in `/intelligence/query`:
  - Input validation: `QueryRequest.question` now rejects known injection patterns
    (`ignore previous instructions`, `act as`, `you are now`, `<system>`, etc.) via Pydantic validator.
  - Structural isolation: repo context fields (description, readme_summary, problem_solved) are
    wrapped in `<repo index="N">...</repo>` XML delimiters and truncated to 400 chars each,
    preventing data fields from being interpreted as instructions.
  - System prompt hardened with explicit data/instruction boundary rules that cannot be
    overridden by content inside `<repo>` tags or the user question.
  - User prompt restructured to use `<question>` and `<repos>` delimiters for clear role separation.
- **SQL injection fix** in `/intelligence/query` knowledge graph edges query:
  - Replaced f-string UUID interpolation (`f"'{tid}'"` joined into raw SQL) with parameterized
    `ANY(CAST(:ids AS uuid[]))` binding. Previous code was injectable if UUIDs were attacker-controlled.

## [1.4.0] - 2026-03-23

### Fixed
- **AI Dev Coverage badges** — `/library/full` now returns `aiDevSkillStats` keyed by skill *group* names
  (`"Inference & Serving"`, `"RAG & Knowledge"`, etc.) matching the frontend taxonomy exactly.
  Previously returned individual tool names that the frontend couldn't look up, causing all badges to show ❌.
- **Builders section** — added `KNOWN_ORG_CATEGORIES` override table (35 orgs). `anthropics`, `huggingface`,
  `facebookresearch`, `langchain-ai`, `deepset-ai` and others were classified as `"individual"` in the DB and
  filtered out of the Builders UI. Now correctly classified with display names (e.g. "Anthropic", "Meta Research").
  Builders are now sorted by `repoCount` descending instead of `totalParentStars`.
- **Tag cloud** — system tags (`Active`, `Forked`, `Built by Me`, `Inactive`, `Archived`, `Popular`) are now
  filtered from `tagMetrics` at the source. These had 500–800+ repo counts and dominated the linear font scaling.
- **Fork timeline dates** — removed bad fallback that showed the ingestion date (Mar 2026) as
  `upstreamCreatedAt` for all forks. Field is now left empty until `backfill_fork_dates.py` populates real data.

### Added
- 32 unit tests in `tests/test_library_full.py` covering `_build_ai_dev_skill_stats`,
  `_build_builder_stats`, `_build_tag_metrics`, and `sanitize_repo` date fallback safety.

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
