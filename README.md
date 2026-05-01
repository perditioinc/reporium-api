# reporium-api
<!-- perditio-badges-start -->
[![Tests](https://github.com/perditioinc/reporium-api/actions/workflows/test.yml/badge.svg)](https://github.com/perditioinc/reporium-api/actions/workflows/test.yml)
![Last Commit](https://img.shields.io/github/last-commit/perditioinc/reporium-api)
![License](https://img.shields.io/github/license/perditioinc/reporium-api)
![python](https://img.shields.io/badge/python-3.11%2B-3776ab)
![suite](https://img.shields.io/badge/suite-Reporium-6e40c9)
![deployed](https://img.shields.io/badge/deployed-Cloud%20Run-blue)
![docs](https://img.shields.io/badge/docs-%2Fdocs-blue)
<!-- perditio-badges-end -->

Backend API for Reporium — the AI-native GitHub knowledge graph. Handles all data reads and writes, semantic search, 8-dimension dynamic taxonomy, portfolio intelligence, and a queryable MCP interface for Claude.

Public reads. Protected writes (X-Ingest-Key). Admin operations (X-Admin-Key). LLM endpoints (X-App-Token).

---

## Quick Start (local)

```bash
docker-compose up
```

API: http://localhost:8000
Docs: http://localhost:8000/docs

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

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `DATABASE_URL` | Yes | `postgresql+asyncpg://...` |
| `ADMIN_API_KEY` | No | X-Admin-Key header for admin endpoints. Unset = open in dev. |
| `INGEST_API_KEY` | No | X-Ingest-Key header for ingest endpoints. Unset = open in dev. |
| `APP_API_TOKEN` | No | X-App-Token header for LLM endpoints. Unset = open in dev; logged at CRITICAL on prod startup if missing. |
| `REDIS_URL` | No | Redis connection string — API works without it |
| `GRAPH_SNAPSHOT_BUCKET` | No | GCS bucket for the published graph snapshot artifact |
| `GRAPH_SNAPSHOT_OBJECT` | No | Object path for the graph snapshot artifact |
| `GRAPH_SNAPSHOT_LOCAL_PATH` | No | Local graph snapshot file for development/tests |
| `GH_USERNAME` | No | GitHub username |
| `GH_TOKEN` | No | GitHub token |
| `ENVIRONMENT` | No | `development` \| `production` |
| `PUBSUB_AUDIENCE` | No | GCP Pub/Sub push JWT audience for signature verification |
| `GCP_PROJECT_ID` | No | GCP project ID (default: `perditio-platform`) |

> Secrets are resolved from GCP Secret Manager in production (no `.env` needed on Cloud Run).
>
> `GET /graph/edges` serves from the published graph snapshot artifact before attempting live database queries so the graph UI remains available during primary DB incidents.

---

## Auth

| Header | Env Var | Applies To |
|--------|---------|-----------|
| `X-Ingest-Key` | `INGEST_API_KEY` | All `/ingest/*` routes |
| `X-Admin-Key` | `ADMIN_API_KEY` | All `/admin/*` routes (also `/metrics/*` when `METRICS_REQUIRE_AUTH=1`) |
| `X-App-Token` | `APP_API_TOKEN` | LLM-backed endpoints: `/intelligence/ask`, `/intelligence/ask/stream`, `/intelligence/nl-filter`, `/intelligence/compare` |

All three headers are optional when their env var is unset (dev-safe passthrough). In production, set them in Cloud Run environment variables or GCP Secret Manager.

### `X-App-Token` (LLM endpoints)

The `/api/intelligence/ask` family and other LLM-backed endpoints require the `X-App-Token` header — a shared secret protecting expensive Anthropic calls. Send it on every request:

```bash
curl https://api.reporium.com/intelligence/ask \
  -H 'Content-Type: application/json' \
  -H 'X-App-Token: <your-app-token>' \
  -d '{"question": "What are the top RAG repos?"}'
```

**Note:** the token goes in the `X-App-Token` header — **not** `Authorization: Bearer`. Sending a Bearer token to `/intelligence/ask` will return HTTP 403.

A request without the header (or with the wrong token) returns:

```http
HTTP/1.1 403 Forbidden
Content-Type: application/json

{"detail": "Missing X-App-Token header"}
```

Or, if a token was sent but doesn't match the configured `APP_API_TOKEN`:

```http
HTTP/1.1 403 Forbidden
Content-Type: application/json

{"detail": "Invalid X-App-Token header"}
```

The 403 response detail names the header explicitly so integrators can self-diagnose without reading the source. To obtain an app token in production, request one via the platform admin (it's stored in GCP Secret Manager under `APP_API_TOKEN`). For local development, leave `APP_API_TOKEN` unset — the auth dependency is a no-op outside production.

The same scheme is registered in `/openapi.json` as `AppToken` so it's self-documenting in the Scalar docs at `/docs`.

---

## Endpoints

### Public
| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check — DB, Redis, last ingestion time |
| `GET /library/full` | Complete LibraryData for the frontend (cached 5 min) |
| `GET /library` | Paginated library |
| `GET /repos` | Filterable repo list with full-text + taxonomy search |
| `GET /repos/{name}` | Repo detail with taxonomy, embeddings, similarity score (cached 1 hr) |
| `GET /repos/{name}/similar` | Semantically similar repos via pgvector cosine similarity |
| `GET /search?q=` | Full-text + semantic search with cosine similarity scores |
| `GET /taxonomy/dimensions` | List of all taxonomy dimensions with repo counts |
| `GET /taxonomy/{dimension}` | Values for a dimension, sorted by repo count |
| `GET /trends` | Latest trend signals |
| `GET /gaps` | Gap analysis — underrepresented taxonomy areas |
| `GET /stats` | Library statistics |
| `GET /docs` | Scalar API docs (dark theme) |

### Ingest (X-Ingest-Key required)
| Endpoint | Description |
|----------|-------------|
| `POST /ingest/repos` | Batch upsert repos with full taxonomy (100 max per request) |
| `POST /ingest/repos/{name}/enrich` | Update enriched fields for a single repo |
| `POST /ingest/trends/snapshot` | Commit trend snapshot |
| `POST /ingest/gaps` | Update gap analysis |
| `POST /ingest/log` | Update ingestion run log |
| `POST /ingest/events/repo-ingested` | GCP Pub/Sub push handler — triggers taxonomy + intelligence refresh |

### LLM endpoints (X-App-Token required)
| Endpoint | Description |
|----------|-------------|
| `POST /intelligence/ask` | Natural-language Q&A over the knowledge graph (Anthropic-backed) |
| `POST /intelligence/ask/stream` | Streaming variant of `/intelligence/ask` |
| `POST /intelligence/nl-filter` | Translate natural-language queries into structured filter params |
| `GET /intelligence/compare` | Side-by-side compare 2-5 repos with metrics + tags + HN mentions |
| `GET /intelligence/portfolio-insights` | Curated portfolio dashboard feed |

### Admin (X-Admin-Key required)
| Endpoint | Description |
|----------|-------------|
| `POST /admin/taxonomy/rebuild` | Aggregate raw_values from repo_taxonomy into taxonomy_values |
| `POST /admin/taxonomy/embed` | Generate embeddings for taxonomy_values missing vectors |
| `POST /admin/taxonomy/assign` | Assign taxonomy_values to repos via pgvector cosine similarity |
| `GET /admin/data-quality` | Data quality metrics |
| `POST /admin/tags/prune` | Remove low-signal tags |

---

## 8-Dimension Dynamic Taxonomy

Taxonomy is **fully data-driven** — no hardcoded lists. Values are generated freely by Claude during enrichment and grow automatically as new repos are added.

| Dimension | What it captures |
|-----------|-----------------|
| `skill_area` | Core AI/ML competency (e.g. RAG & Retrieval, Fine-tuning, Agents) |
| `industry` | Target vertical (e.g. Healthcare, Finance, DevTools) |
| `use_case` | Problem solved (e.g. Document Q&A, Code generation) |
| `modality` | Data type (Text, Vision, Audio, Multimodal) |
| `ai_trend` | Emerging trend (Agentic AI, Reasoning Models, Long Context) |
| `deployment_context` | Where it runs (Edge, Cloud, On-premise, Serverless) |
| `tags` | Cross-cutting labels (production-ready, research, benchmark) |
| `maturity_level` | Repo lifecycle stage (prototype, production, research) |

**Zero-re-enrichment expansion:** Adding a new taxonomy value costs one local embedding call (~$0.00001) and one SQL similarity query. Existing repos are matched automatically via pgvector — no Claude re-enrichment of any repo required.

---

## Semantic Search & Similarity

- **Embeddings:** `all-MiniLM-L6-v2` (384-dim), stored in `repo_embeddings.embedding_vec`
- **Index:** HNSW (`vector_cosine_ops`) for fast approximate nearest-neighbor
- **Cosine similarity scores** are returned in search results and repo detail responses
- **Taxonomy assignment** uses the same embeddings: `1 - (taxonomy_value.embedding_vec <=> repo.embedding_vec) >= 0.65`

---

## Pub/Sub Event Flow

```
reporium-ingestion
  └─ publishes repo.ingested → GCP Pub/Sub topic: repo-ingested
                                    │
                              push subscription
                                    │
                                    ▼
reporium-api  POST /ingest/events/repo-ingested
  ├─ embed new taxonomy_values (sentence-transformers)
  ├─ assign taxonomy via cosine similarity
  └─ invalidate portfolio intelligence cache
```

---

## MCP Server

Reporium exposes an MCP (Model Context Protocol) server via the separate `reporium-mcp` repo, giving Claude Code direct query access to the knowledge graph.

```bash
# Add to Claude Code
claude mcp add reporium -- python /path/to/reporium-mcp/mcp_server.py
```

10 tools available: `search_repos`, `search_repos_semantic`, `get_repo`, `find_similar_repos`, `list_taxonomy_dimensions`, `list_taxonomy_values`, `get_repos_by_taxonomy`, `ask_portfolio`, `get_portfolio_gaps`, `get_ai_trends`.

---

## Deploy to GCP Cloud Run

1. Create Cloud SQL PostgreSQL instance with pgvector extension enabled
2. Set `DATABASE_URL=postgresql+asyncpg://[user]:[pass]@/[db]?host=/cloudsql/[connection-name]`
3. Set `ADMIN_API_KEY` and `INGEST_API_KEY` in Cloud Run environment variables (or Secret Manager)
4. Build and push Docker image to Artifact Registry
5. Deploy to Cloud Run with the Cloud SQL connection
6. Run `alembic upgrade head` via Cloud Run Jobs or shell

---

## Tests

```bash
pip install pytest pytest-asyncio httpx
pytest tests/
```
