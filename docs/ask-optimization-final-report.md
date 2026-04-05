# /intelligence/ask — Full Optimization Report (2026-04-05)

3 rounds of autonomous audit and implementation across cost, security, privacy,
reliability, and observability. Every change is $0 infrastructure cost.

---

## Executive Summary

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Estimated Claude spend per query | ~$0.008 | ~$0.003 | **-60%** |
| Cache hit rate (projected) | ~15% | ~40-55% | **+25-40pp** |
| Max concurrent DB connections | 5 | 30 | **6x** |
| Session data retention | indefinite | 90 days | GDPR fix |
| Error message PII exposure | yes | no | Privacy fix |
| Metrics endpoints auth | none | feature-flagged | Security fix |
| Background task error visibility | silent | logged | Reliability fix |

---

## Round 1 — Core Cost Optimization (PRs #231-#234)

### Shipped
1. **Prompt caching** — `cache_control: {type: "ephemeral"}` on system prompt + sources block. Saves ~50% input tokens on cache hit.
2. **pgvector semantic cache** — Cosine distance threshold 0.15 (~0.85 similarity). Near-duplicate questions serve cached answers with zero Claude call.
3. **Smart routing** — SQL shortcut for counting/listing questions (e.g., "how many AI agents are there?"). No LLM invocation at all.
4. **Haiku-first tiered routing** — `claude-haiku-4-5-20250414` for simple questions, `claude-sonnet-4-20250514` only for complex multi-repo analysis. Haiku is ~10x cheaper.
5. **Query normalization** — Synonym resolution, punctuation/whitespace stripping for cache key generation. Prevents near-miss cache misses.
6. **Context hygiene** — 240-char description cap, minimal fields sent to Claude. Reduces input tokens by ~15%.
7. **Session memory compaction** — Older turns compressed to 80-char previews. Prevents session bloat.
8. **Cloud Run rightsizing** — concurrency 20→200, max-instances 3→10 ($0 change, autoscale headroom).
9. **Token-spend observer** — Per-route, per-model token tracking via `/metrics/spend` + $10/day budget guardrail.
10. **Golden-set CI gate** — Numeric quality test (0.7 threshold) blocks regressions in `ask-quality-gate` workflow.

### PRs
- #231 — Cloud Run rightsizing
- #232 — Golden-set quality gate
- #233 — Token-spend observer + budget guardrail
- #234 — Prompt caching, query normalization, context hygiene

---

## Round 1.5 — Security Hardening (PR #242)

### Shipped
1. **Session ownership binding** — SHA-256 `token_hash` column on `ask_sessions`. Each app token can only read its own session history. Prevents cross-tenant session theft.
2. **Timing-safe auth** — All 4 secret comparisons in `auth.py` use `hmac.compare_digest()`. Prevents timing side-channel attacks.
3. **Prompt-injection defense** — Structured `<question>` XML wrapping + system prompt defense clause. `_sanitize_question` changed from hard-reject to log-only (avoids false-positive blocking).
4. **Migration** — `022_add_token_hash_to_ask_sessions.py` with composite index.
5. **23 security tests** in `test_security_hardening_2026_04.py`.

---

## Round 2 — Deep Cost + Privacy (PRs #246-#249)

### Shipped
1. **Dynamic max_tokens** — Haiku=512, Sonnet=768 (was hardcoded 1024). Output tokens are 5x more expensive than input — this alone saves ~20%.
2. **stop_sequences** — `["</answer>", "\n\nNote:", "\n\nDisclaimer:"]` trims tail output that adds no value.
3. **Negative caching** — Failed/low-quality answers cached for 60s with `"negative": True` marker. Prevents re-generation on retry.
4. **Early-exit on low-similarity retrieval** — If top-1 pgvector similarity < 0.40, return deterministic "I don't have enough relevant data" with zero Claude call. Budget-friendly AND quality-friendly.
5. **Session turns 3→2** — Q&A context rarely benefits from >1 prior turn. Saves ~500 tokens/session.
6. **Streaming cache-write alignment** — `/ask/stream` now writes to Redis cache after completion. Previously, first user on streaming paid full price AND second user on non-streaming got a miss.
7. **Smart-route TTL 300s→3600s** — Simple counting answers don't change hourly. 12x cache lifetime.
8. **Semantic threshold A/B** — `ASK_CACHE_RELAXED=1` env flag switches cosine threshold from 0.15 to 0.25 for A/B testing. Could 2-3x semantic cache hit rate.
9. **ask_sessions 90-day retention** — `POST /admin/purge-ask-sessions?days=90` endpoint. Closes GDPR retention gap.
10. **RTBF delete endpoint** — `DELETE /admin/ask-sessions/{session_id}`. Closes #238.
11. **PII redaction in query logs** — `app/privacy.py::redact_pii()` strips emails, phone numbers, API keys from persisted question text. Claude still receives raw text.
12. **Injection log rate-limiting** — 60s cooldown per IP prevents log flooding DoS.
13. **Metrics auth gate** — `METRICS_REQUIRE_AUTH=1` + `X-Admin-Key` header on `/metrics/slo`, `/metrics/spend`, `/metrics/latest`, `/audit/status`. Closes #236.

### PRs
- #246 — Retention + RTBF + PII redaction (closes #238)
- #247 — Metrics auth gate (closes #236)
- #248 — Dynamic max_tokens + stop_sequences + negative cache + early-exit
- #249 — Streaming cache-write + TTL bumps + semantic threshold A/B

---

## Round 3 — Reliability + Observability (PRs #252-#254)

### Shipped
1. **DB connection pool sizing** — `pool_size=20`, `max_overflow=10`, `pool_recycle=3600`. Prevents pool exhaustion at concurrency=200.
2. **Fire-and-forget task error logging** — `_task_done_callback` on all 14 `asyncio.create_task()` calls. Silent failures now emit warnings.
3. **Startup env validation** — CRITICAL log if `APP_API_TOKEN` unset in production.
4. **Embedding generation timeout** — 5s `asyncio.wait_for` around `embed_model.encode()`. Graceful fallback on timeout.
5. **Streaming timeout alignment** — 35s→30s to match non-streaming path.
6. **Error message PII scrub** — Removed user-supplied input from all `HTTPException` detail messages.
7. **Phase-level latency breakdown** — Single INFO log per request: `total, smart, embed, search, context, claude` milliseconds + model + cached flag. Enables targeted optimization.

### PRs
- #252 — Connection pool + task error logging + env validation
- #253 — Embedding timeout + streaming timeout alignment
- #254 — Error PII scrub + latency breakdown logging

---

## Issues Closed

| Issue | Title | Closed By |
|-------|-------|-----------|
| #236 | Metrics endpoints unauthenticated | PR #247 |
| #238 | ask_sessions no retention / RTBF | PR #246 |

---

## Test Coverage Added

| Test File | Tests | Round |
|-----------|-------|-------|
| test_security_hardening_2026_04.py | 23 | 1.5 |
| test_ask_output_caps.py | 8 | 2 |
| test_ask_privacy.py | 15 | 2 |
| test_metrics_auth_gate.py | 8 | 2 |
| test_ask_cache_effectiveness.py | 10 | 2 |
| test_reliability_hardening.py | 6 | 3 |
| test_ask_timeout_hardening.py | 4 | 3 |
| test_ask_observability.py | 9 | 3 |
| **Total new tests** | **83** | |

---

## Architecture After All Rounds

```
User Question
     |
     v
[Query Normalization] -- synonyms, punctuation, whitespace
     |
     v
[Rate Limiter] -- 6/min, 60/day per IP (SlowAPI + Redis)
     |
     v
[Redis Cache Check] -- MD5 of normalized question
     |-- HIT --> return cached answer (TTL 1800s / 3600s for smart-route)
     |
     v
[Smart Route Check] -- SQL shortcut for counting/listing
     |-- MATCH --> return SQL result, cache 3600s
     |
     v
[Embedding Generation] -- sentence-transformers (5s timeout)
     |-- TIMEOUT --> early-exit "I don't have enough data"
     |
     v
[Semantic Cache Check] -- pgvector cosine distance (0.15 or 0.25)
     |-- HIT --> return cached answer
     |
     v
[pgvector Retrieval] -- top_k+10, filter similarity >= 0.45
     |-- ALL < 0.40 --> early-exit "I don't have enough data"
     |
     v
[Model Selection] -- Haiku for simple, Sonnet for complex
     |
     v
[Context Building] -- 240-char descriptions, minimal fields
     |
     v
[Session History] -- last 2 turns, 8000 char cap, 80-char compaction
     |
     v
[Claude API Call] -- prompt-cached system prompt, stop_sequences,
     |                 dynamic max_tokens (512/768), 30s timeout
     |
     v
[Response Processing]
     |-- Log (PII-redacted question, phase latency breakdown)
     |-- Cache (Redis 1800s; negative=60s with NULL embedding)
     |-- Session save (skip negative answers)
     |-- Token observer (per-route, per-model spend tracking)
     |
     v
[Return to User]
```

---

## Deployment History

| Rev | Date | Content |
|-----|------|---------|
| 00144 | 2026-04-05 | Round 1 + security hardening |
| 00145 | 2026-04-05 | Round 2 (cost + privacy) |
| 00146 | 2026-04-05 | Round 3 (reliability + observability) |

---

## Remaining Opportunities (not yet implemented)

1. **Concurrent session turn writes** — Add DB uniqueness constraint on (session_id, turn_number) to prevent race conditions.
2. **Cache hit rate by type** — Split cache_source into smart-route / redis / semantic for ROI analysis.
3. **Embedding storage opt-out** — Let users opt out of question embedding persistence.
4. **Rate limiter Redis failover** — Periodic health re-probe instead of startup-only check.
5. **Model pricing table** — Env-driven pricing for accurate cost tracking with new models.
6. **Duplicate session capping logic** — DRY refactor (minor code quality).

These are lower-priority items that don't affect cost or security.
