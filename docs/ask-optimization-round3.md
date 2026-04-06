# /intelligence/ask Optimization — Round 3 Audit (2026-04-05)

Third-pass audit after rounds 1+2 shipped (PRs #231–#234, #242, #246–#249).
Focus: reliability, observability, remaining privacy gaps, and code quality.

## Current State (post-round-2, pending rev 00145 deploy)

Shipped in round 1:
- Prompt caching (system + sources ephemeral breakpoints)
- pgvector semantic cache (0.15 cosine distance)
- Smart routing (SQL shortcut, no LLM for simple questions)
- Haiku-first tiered model routing
- Query normalization for cache keys
- Context hygiene (240-char descriptions, minimal fields)
- Session memory compaction (older turns to 80-char preview)
- top_k=5 default
- Token-spend observer + /metrics/spend + $10/day budget guardrail
- Golden-set CI gate (0.7 threshold)
- Cloud Run concurrency=200, max-instances=10
- Session ownership via token_hash
- Timing-safe auth + prompt-injection defense

Shipped in round 2:
- Dynamic max_tokens (Haiku=512, Sonnet=768) + stop_sequences
- Streaming endpoint cache-write alignment
- Smart-route TTL 300s → 3600s
- Semantic threshold A/B flag (ASK_CACHE_RELAXED)
- Negative caching (60s TTL for low-quality answers)
- Early-exit on low-similarity retrieval (<0.40)
- Session turns 3 → 2
- ask_sessions 90-day retention purge + RTBF endpoint
- PII redaction in query_logs
- Metrics auth gate (feature-flagged)
- Injection log rate-limiting

## Round-3 Findings

### Reliability

1. **Database connection pool undersized** — `create_async_engine` uses SQLAlchemy default (pool_size=5). With concurrency=200, this will exhaust under load.
   - Fix: pool_size=20, max_overflow=10, pool_recycle=3600

2. **Fire-and-forget task exceptions silently swallowed** — All `asyncio.create_task()` calls for `_log_query`, `cache.set`, `_save_session_turn` have no error handler.
   - Fix: Add `_task_done_callback` that logs warnings on failure.

3. **No timeout on embedding generation** — `embed_model.encode()` has no timeout. If model stalls, request hangs indefinitely.
   - Fix: Wrap in `asyncio.wait_for(..., timeout=5.0)` with graceful fallback.

4. **Streaming timeout asymmetry** — Streaming uses 35s timeout vs non-streaming 30s. Should be aligned.
   - Fix: Both paths → 30s.

5. **Startup env validation missing** — If APP_API_TOKEN is unset in production, /ask silently rejects all requests with 500.
   - Fix: Log CRITICAL on startup if env=production and token missing.

### Privacy / Security

6. **User input echoed in error messages** — `detail=f"Repo '{repo_name}' not found"` leaks search terms in HTTP responses.
   - Fix: Use generic messages like "Repo not found".

7. **Concurrent session turn writes race condition** — Two concurrent requests to same session can produce duplicate turn_numbers.
   - Fix: Add DB uniqueness constraint on (session_id, turn_number).

### Observability

8. **No latency phase breakdown** — Total latency logged but not split by phase (smart-route, embed, search, context, claude).
   - Fix: Add per-phase timing in `_run_query` with single INFO log.

9. **No cache hit rate by type** — Cache hits undifferentiated between smart-route, Redis, and semantic.
   - Fix: Add cache_type field to hit recording.

### Code Quality

10. **Duplicate session capping logic** — Session history capped identically in two places.
    - Fix: Remove redundant cap; trust `_load_session_turns`.

11. **Hardcoded magic numbers** — Description cap, answer truncate, session char cap scattered inline.
    - Fix: Extract to named module-level constants.

12. **Unused import `get_anthropic_key`** — Dead import in intelligence.py.
    - Fix: Remove.

## Priority Order (by ROI at $0)

| # | Change | Impact | Risk |
|---|--------|--------|------|
| 1 | Connection pool sizing | Prevents 503s under load | very low |
| 2 | Task error logging | Catches silent failures | very low |
| 3 | Embedding timeout guard | Prevents request hangs | low |
| 4 | Error message PII scrub | Privacy compliance | very low |
| 5 | Latency phase breakdown | Debug slow requests | very low |
| 6 | Streaming timeout alignment | Consistency | very low |
| 7 | Startup env validation | Prevent silent misconfig | very low |

## Agent Assignments

- Agent K: Items 1, 2, 5 (reliability hardening)
- Agent L: Items 4, 8 (privacy + observability)
- Agent M: Items 3, 6 (timeout hardening)
