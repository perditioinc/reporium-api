# Reporium Multiagent Orchestration — Onboarding Guide

## Overview

Reporium is an AI-native platform for tracking, enriching, and querying 1,680+ AI/ML GitHub repositories. This document provides everything needed to onboard multiagent orchestration across the Reporium suite.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    REPORIUM SUITE (14 repos)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐    │
│  │  reporium    │   │ reporium-api │   │ reporium-ingestion│    │
│  │  (Next.js)   │──▶│  (FastAPI)   │◀──│  (Python scripts) │    │
│  │  Vercel      │   │  Cloud Run   │   │  Local/CI         │    │
│  └──────────────┘   └──────┬───────┘   └────────┬──────────┘    │
│                            │                     │               │
│                     ┌──────▼───────┐   ┌────────▼──────────┐    │
│                     │   Neon DB    │   │  Claude API       │    │
│                     │  PostgreSQL  │   │  (Haiku + Sonnet) │    │
│                     │  + pgvector  │   └───────────────────┘    │
│                     └──────────────┘                             │
│                                                                  │
│  Supporting Services:                                            │
│  ├─ reporium-db        (nightly index generation)               │
│  ├─ reporium-events    (GCP Pub/Sub event bus)                  │
│  ├─ reporium-metrics   (nightly platform metrics)               │
│  ├─ reporium-scoring   (repo quality scoring 0-100)             │
│  ├─ reporium-trending  (GitHub trending discovery)              │
│  ├─ reporium-mcp       (MCP server for Claude tools)            │
│  ├─ reporium-audit     (contract & schema validation)           │
│  ├─ reporium-dataset   (dataset export generation)              │
│  ├─ reporium-security  (security scanning)                      │
│  └─ reporium-system-design (architecture docs)                  │
└─────────────────────────────────────────────────────────────────┘
```

## AI Pipeline (Data Flow)

### Ingestion → Enrichment → Serving

| Phase | Tool | Model | Cost | Output |
|-------|------|-------|------|--------|
| 0. Discovery | reporium-trending | GitHub API | $0 | candidates.json |
| 1. Ingestion | reporium-ingestion | GitHub API | $0 | repos table (name, owner, language, etc.) |
| 2. Tagging | tagger.py | Keyword matching | $0 | integration_tags, repo_tags |
| 3. AI Enrichment | ai_enricher.py | Claude Sonnet | ~$0.006/repo | 8-dim taxonomy, summary, quality |
| 4. Embeddings | generate_embeddings.py | all-MiniLM-L6-v2 | $0 | 384-dim vectors in pgvector |
| 5. Graph Build | build_knowledge_graph.py | SQL/heuristics | $0 | 42,523 edges (4 types) |
| 6. Serving | reporium-api | Haiku/Sonnet | ~$0.01/query | /ask, /search, /similar |

### Knowledge Graph Edge Types

| Type | Count | How Generated |
|------|-------|---------------|
| COMPATIBLE_WITH | 15,000 | Repos sharing 2+ integration tags |
| ALTERNATIVE_TO | 15,000 | Repos in same category |
| SAME_CATEGORY | 7,523 | Category-based connections for unconnected repos |
| MAINTAINED_BY | 5,000 | Repos with same owner |
| **Total** | **42,523** | **0 repos unconnected** |

## API Endpoints for Agent Use

### Read Operations (no auth required)
| Endpoint | Purpose | Cost |
|----------|---------|------|
| `GET /health` | Service health check | $0 |
| `GET /library/full` | All public repos with enrichment data | $0 |
| `GET /repos/{name}` | Single repo detail | $0 |
| `GET /search/semantic?q=...` | Vector similarity search | $0 |
| `GET /graph/edges` | Knowledge graph edges | $0 |
| `GET /metrics/latest` | Platform metrics | $0 |

### Intelligence (rate-limited, optional auth)
| Endpoint | Purpose | Cost |
|----------|---------|------|
| `POST /intelligence/ask` | Natural language Q&A over repos | ~$0.01 |
| `GET /intelligence/similar/{name}` | Find similar repos | $0 |
| `GET /intelligence/recommended?seeds=...` | Multi-seed recommendations | $0 |

### Admin Operations (requires X-Admin-Key)
| Endpoint | Purpose |
|----------|---------|
| `POST /admin/taxonomy/bootstrap` | Assign taxonomy via pgvector similarity |
| `POST /admin/enrichment/trigger` | Trigger AI enrichment for new repos |
| `POST /admin/backfill/categories` | Rebuild categories from tags |
| `POST /admin/embeddings/backfill` | Generate missing embeddings |

## Database Schema (Key Tables)

| Table | Rows | Purpose |
|-------|------|---------|
| repos | 1,680 | Core repo metadata |
| repo_edges | 42,523 | Knowledge graph relationships |
| repo_categories | 17,199 | Repo ↔ category assignments |
| repo_taxonomy | 38,037+ | 6-dimension AI-generated taxonomy |
| repo_embeddings | 1,641 | 384-dim pgvector embeddings |
| repo_tags | ~1,667 | Keyword tags |
| repo_builders | ~1,641 | Known org/builder associations |
| query_log | varies | /ask query cache + cost tracking |

## Data Coverage (as of 2026-04-08)

| Metric | Coverage |
|--------|----------|
| Category | 100% (1,641/1,641 public) |
| Summary | 100% |
| Quality signals | 100% |
| Taxonomy | 100% (1,674 repos) |
| Embeddings | 100% |
| Graph edges | 100% (0 repos unconnected) |

## Environment Variables

### Required for API
```
DATABASE_URL          # PostgreSQL connection (Neon)
ANTHROPIC_API_KEY     # Claude API for /ask endpoint
INGESTION_API_KEY     # Auth for /ingest endpoints
```

### Optional
```
REDIS_URL             # Cache (graceful fallback without)
ADMIN_API_KEY         # Admin endpoint access
ENVIRONMENT           # "production" | "test"
SENTRY_DSN            # Error tracking (not yet wired)
IP_HASH_SECRET        # HMAC salt for IP hashing
DAILY_LLM_COST_CAP   # Budget cap for /ask (default $5)
```

## Cost Model

| Component | Monthly Cost (est.) |
|-----------|-------------------|
| Neon DB (free tier) | $0 |
| Cloud Run (min-instances=0) | ~$5-15 |
| Claude API (/ask queries) | ~$5-30 (depends on traffic) |
| AI Enrichment (one-time per repo) | ~$0.006/repo |
| Vercel (frontend) | $0 (free tier) |
| **Total** | **~$10-50/month** |

## Security Posture

- Auth: HMAC timing-safe comparison for all API keys
- CORS: Strict allowlist (reporium.com, Vercel previews)
- Rate limiting: IP-based (200/hr global, 6/min for /ask)
- Private repos: `is_private=false` filter on all public queries
- Secrets: GCP Secret Manager in production
- Error handling: Generic error messages (no internal details leaked)

## Multiagent Integration Points

1. **MCP Server** (reporium-mcp): Exposes repo search and details as Claude tool calls
2. **REST API**: All endpoints available for agent consumption
3. **Event Bus** (reporium-events): Pub/Sub topics for repo.ingested, repo.enriched events
4. **Knowledge Graph**: 42K+ typed edges for relationship reasoning
5. **Semantic Search**: pgvector-powered similarity for any text query
6. **Budget Governance**: Redis-backed daily cost caps prevent runaway agent spending

## Quick Start for Agents

```python
import httpx

API = "https://reporium-api-573778300586.us-central1.run.app"

# Search for repos
results = httpx.get(f"{API}/search/semantic?q=vector+database&limit=5").json()

# Get similar repos
similar = httpx.get(f"{API}/intelligence/similar/chromadb?limit=5").json()

# Ask a question
answer = httpx.post(f"{API}/intelligence/ask", json={
    "question": "What are the best RAG frameworks?",
    "limit": 10
}, headers={"X-App-Token": "..."}).json()
```
