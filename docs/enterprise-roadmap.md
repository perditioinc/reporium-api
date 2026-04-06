# Reporium Enterprise Roadmap — Tiered Investment Plan

*Generated 2026-04-05. All estimates assume current GCP infrastructure.*

---

## Tier 0: $0 — Free Improvements (Immediate)

### 0.1 Golden-Set Expansion (More Coverage)
**ROI: ★★★★★ | Risk: None | Effort: 2hr**

Current: 18 test cases in `tests/golden_set_ask.yaml` covering simple/medium/complex.

Missing patterns to add ($0 — hand-crafted, no API calls):
- **Negation queries:** "Which repos do NOT use PyTorch?"
- **Temporal queries:** "What's the newest RAG framework?"
- **Multi-category:** "Compare vector databases vs embedding models"
- **Ambiguous:** "What's the best tool?" (should gracefully ask for clarification)
- **Out-of-domain:** "What's the weather?" (should refuse politely)
- **Multi-language:** "¿Qué herramientas de IA hay?" (should handle or gracefully decline)
- **Long context:** 500+ char questions with background context
- **Follow-up:** Session-based "tell me more about that" queries
- **Empty result:** Questions about repos we don't track
- **Injection edge cases:** Unicode homoglyphs, RTL override chars

Target: 18 → 50+ test cases, covering all smart routes + edge cases.

### 0.2 GitHub Data Backfill (Free API Data)
**ROI: ★★★★★ | Risk: None | Effort: 3hr**

GitHub API provides for free (no enrichment API calls needed):
- `topics` (GitHub-native tags) — **already available in API, NOT stored**
- `license` (SPDX) — **column exists (`license_spdx`) but empty**
- `forks_count` — total forks
- `watchers_count` — distinct from stargazers
- `contributors_count` — number of contributors (requires separate API call per repo)

Backfill these via existing forksync or new admin endpoint. Improves:
- Graph edge quality (more text for embeddings)
- Smart routing (can answer "what license does X use?")
- /library/full response richness

### 0.3 Documentation Updates
**ROI: ★★★★ | Risk: None | Effort: 2hr**

Update stale docs:
- `docs/SLOs.md` — Mark Sentry as "not wired", add Cloud Trace as planned
- `docs/DEPLOYMENT.md` — Update revision history through 00147
- `docs/ROADMAP.md` — Reflect completed optimization work
- `README.md` — Add architecture diagram, endpoint summary
- `CHANGELOG.md` — Add rounds 1-4 entries
- NEW: `docs/governance.md` — Gateway pattern, rate limits, budgets

### 0.4 Cloud Run Service.yaml Alignment
**ROI: ★★★★ | Risk: Low | Effort: 30min**

`deploy/service.yaml` still shows `containerConcurrency: 20` and `maxScale: 3`.
Code ships `concurrency=200` and `max-instances=10` via gcloud CLI.
Align the YAML to match reality.

### 0.5 Metrics Dashboard JSON Export
**ROI: ★★★ | Risk: None | Effort: 1hr**

Add `GET /metrics/export` endpoint that returns ALL metrics in a single JSON payload:
- SLO snapshot
- Token spend
- Embedding coverage
- Rate limit counters
- Uptime since last restart

Consumable by external tools (Grafana, Looker, BigQuery) via simple HTTP poll.

---

## Tier 1: $0-5 — Minimal Spend, Maximum Impact

### 1.1 OpenTelemetry + Cloud Trace Integration
**Cost: $0 (Cloud Trace free tier: 2.5M spans/month) | ROI: ★★★★★ | Effort: 4hr**

Add `opentelemetry-sdk` + `opentelemetry-exporter-gcp-trace` to requirements.
Instrument:
- HTTP request spans (middleware)
- Claude API call spans (with model, tokens, cost as attributes)
- pgvector search spans (with similarity scores)
- Redis cache spans (hit/miss)
- Embedding generation spans

This gives:
- Distributed trace waterfall in Cloud Console
- Latency breakdown without custom logging
- Automatic correlation with Cloud Logging
- Export to BigQuery via Cloud Trace → BigQuery sink

### 1.2 BigQuery Trace Export Pipeline
**Cost: $0 (BigQuery free tier: 1TB query/month, 10GB storage) | ROI: ★★★★★ | Effort: 2hr**

Configure Cloud Trace → BigQuery export (GCP native, no code):
1. Create BigQuery dataset `reporium_traces`
2. Create Cloud Logging sink: `resource.type="cloud_run_revision" AND trace!=""` → BigQuery
3. Query in BigQuery for:
   - Adoption trends (unique users/day, questions/day)
   - ROI analysis (cost per query, cache savings)
   - Performance trends (p95 over time)
   - Popular question patterns

### 1.3 Looker Studio Dashboard
**Cost: $0 (Looker Studio free) | ROI: ★★★★ | Effort: 3hr**

Connect BigQuery dataset to Looker Studio:
- Daily active users trend
- Questions per day + cost per question
- Cache hit rate over time
- Top question categories
- Response quality scores
- Token spend breakdown by model

### 1.4 Golden-Set Auto-Generation via Haiku
**Cost: ~$0.50 | ROI: ★★★★★ | Effort: 1hr**

Use Haiku to generate 30+ diverse test cases from existing repo data:
- One API call with all category names → diverse question set
- Human review before committing to golden set
- Covers patterns humans wouldn't think of

---

## Tier 2: $5-20 — Strategic Investment

### 2.1 Vertex AI Integration (Gemini as Fallback/Cost Optimizer)
**Cost: $0-5/month (Gemini free tier generous) | ROI: ★★★★ | Effort: 8hr**

Why: Gemini 2.0 Flash on Vertex AI has a generous free tier and is excellent for:
- Simple question answering (replace Haiku for counting/listing)
- Embedding generation (text-embedding-005 is free up to quota)
- Batch enrichment (Vertex AI Batch API)

Architecture:
```
Question → Model Router → {
  simple → Smart Route (SQL, $0)
  medium → Gemini Flash (Vertex AI, ~$0)
  complex → Claude Haiku ($0.0004)
  very complex → Claude Sonnet ($0.004)
}
```

Implementation:
1. Add `google-cloud-aiplatform` to requirements
2. Create `app/vertex.py` with Gemini client (lazy init)
3. Add Gemini to `_select_model()` as lowest-cost tier
4. Feature-flag: `VERTEX_AI_ENABLED=1`

### 2.2 Graph Edge Evidence Generation
**Cost: ~$3 one-time | ROI: ★★★★ | Effort: 3hr**

For top-500 highest-similarity repo pairs, ask Haiku:
"Why are {repo_a} and {repo_b} related? One sentence."

Store in `edge_evidence` column. Frontend shows "Related because: ..."
Pre-compute + cache permanently. Re-run monthly.

### 2.3 Pre-Computed Comparison Cache
**Cost: ~$2 one-time | ROI: ★★★★ | Effort: 2hr**

For top-100 most-compared repo pairs (from query logs):
- Generate "X vs Y" comparison via Haiku
- Store in Redis permanently
- When user asks "compare X and Y", serve instantly (zero Claude call)

### 2.4 Auditable Sandbox Framework
**Cost: $0 infrastructure | ROI: ★★★★ | Effort: 6hr**

Enterprise requirement for AI governance. Implement:
- **Sandbox mode:** `X-Sandbox: true` header → all Claude calls logged with full prompt/response
- **Audit trail:** `audit_log` table with: timestamp, user, endpoint, prompt, response, model, tokens, cost
- **Replay:** `GET /admin/audit/{id}` to inspect any past AI interaction
- **Budget enforcement:** Per-key daily spend limits (not just global)

### 2.5 API Gateway Pattern
**Cost: $0 | ROI: ★★★★★ | Effort: 4hr**

Implement proper gateway controls:
```python
# Middleware chain:
1. Rate limiter (per-key, not just per-IP)
2. Budget checker (per-key daily spend limit)
3. Sandbox logger (if sandbox header set)
4. Auth validator (timing-safe)
5. Request validator (schema, size limits)
6. Route handler
7. Response logger (audit trail)
8. Cost recorder (token spend)
```

Currently missing: per-key rate limiting, per-key budgets, request size limits.

---

## Tier 3: $20-100 — Enterprise Features

### 3.1 Full Vertex AI Pipeline
**Cost: ~$20-50/month | ROI: ★★★★ | Effort: 16hr**

- Vertex AI Pipelines for batch enrichment (scheduled, managed)
- Vertex AI Model Garden for embedding model serving (if scale demands)
- Vertex AI Feature Store for repo feature vectors
- Vertex AI Matching Engine as pgvector alternative (if >100K repos)

### 3.2 Cloud Monitoring + Alerting
**Cost: ~$5/month | ROI: ★★★★★ | Effort: 4hr**

- Uptime checks on /health (5-min interval, global)
- Alert policies: SLO breach → PagerDuty/Slack
- Custom metrics from /metrics/slo → Cloud Monitoring
- Dashboard in Cloud Console

### 3.3 Cluster Labeling + Topic Modeling
**Cost: ~$5 one-time | ROI: ★★★ | Effort: 4hr**

- Run Louvain community detection on graph ($0, Python networkx)
- Label each cluster via Haiku (~$0.50)
- Store cluster assignments
- Frontend: color-coded graph regions with topic names

### 3.4 Multi-Model Eval Framework
**Cost: ~$20 one-time | ROI: ★★★★ | Effort: 8hr**

- 200-question eval set
- Run through Claude Sonnet (gold standard), Haiku, Gemini Flash
- Score each on: accuracy, relevance, cost, latency
- Identify optimal routing rules
- Output: cost-accuracy Pareto frontier

---

## ROI Summary Table

| Item | Cost | ROI | Priority | Effort |
|------|------|-----|----------|--------|
| Golden-set expansion | $0 | ★★★★★ | P0 | 2hr |
| GitHub data backfill | $0 | ★★★★★ | P0 | 3hr |
| Documentation updates | $0 | ★★★★ | P0 | 2hr |
| service.yaml alignment | $0 | ★★★★ | P0 | 30min |
| Metrics export endpoint | $0 | ★★★ | P1 | 1hr |
| OpenTelemetry + Cloud Trace | $0 | ★★★★★ | P1 | 4hr |
| BigQuery trace export | $0 | ★★★★★ | P1 | 2hr |
| Looker Studio dashboard | $0 | ★★★★ | P1 | 3hr |
| Golden-set auto-gen (Haiku) | $0.50 | ★★★★★ | P1 | 1hr |
| Vertex AI (Gemini fallback) | $0-5/mo | ★★★★ | P2 | 8hr |
| Graph edge evidence | $3 | ★★★★ | P2 | 3hr |
| Pre-computed comparisons | $2 | ★★★★ | P2 | 2hr |
| Auditable sandbox | $0 | ★★★★ | P2 | 6hr |
| API gateway pattern | $0 | ★★★★★ | P2 | 4hr |
| Cloud Monitoring + alerts | $5/mo | ★★★★★ | P2 | 4hr |
| Full Vertex AI pipeline | $20-50/mo | ★★★★ | P3 | 16hr |
| Multi-model eval | $20 | ★★★★ | P3 | 8hr |
| Cluster labeling | $5 | ★★★ | P3 | 4hr |

---

## Governance Architecture

### Gateway Pattern (Server-Side Execution Boundaries)

```
┌─────────────────────────────────────────────────┐
│                  API GATEWAY                     │
│                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │ Auth     │→│ Rate     │→│ Budget       │  │
│  │ Validate │  │ Limiter  │  │ Enforcer     │  │
│  └──────────┘  └──────────┘  └──────────────┘  │
│       ↓              ↓              ↓            │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │ Sandbox  │→│ Request  │→│ Route        │  │
│  │ Logger   │  │ Validator│  │ Handler      │  │
│  └──────────┘  └──────────┘  └──────────────┘  │
│       ↓              ↓              ↓            │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │ Response │→│ Cost     │→│ Audit        │  │
│  │ Logger   │  │ Recorder │  │ Trail        │  │
│  └──────────┘  └──────────┘  └──────────────┘  │
└─────────────────────────────────────────────────┘
```

### Per-Key Controls
- Daily spend limit (configurable per API key)
- Rate limit (requests/minute per key)
- Model access tier (which models this key can invoke)
- Sandbox mode (full prompt/response logging)
- Audit retention (how long to keep audit trail)

### Execution Boundaries
- Max input length per request (chars)
- Max output tokens per request
- Max requests per session
- Max concurrent requests per key
- Blocked model list (prevent key from using Sonnet)

---

## Observability Pipeline

```
Cloud Run (app)
    │
    ├─→ stdout (JSON logs) ─→ Cloud Logging
    │                              │
    ├─→ OpenTelemetry spans ─→ Cloud Trace
    │                              │
    └─→ /metrics/* endpoints       │
                                   ↓
                            BigQuery Sink
                                   │
                            ┌──────┴──────┐
                            │             │
                       Looker Studio   SQL Queries
                            │             │
                       Dashboards    Ad-hoc Analysis
                            │
                       ┌────┴────┐
                       │         │
                    Adoption   ROI
                    Trends    Analysis
```
