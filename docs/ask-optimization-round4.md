# /intelligence/ask Optimization — Round 4 Audit (2026-04-05)

Fourth-pass deep audit after rounds 1-3 shipped (rev 00146).
Focus: token-level waste, code hygiene, test gaps, schema improvements.

## Round-4 Findings

### Cost — Token-Level Optimization

1. **Source block XML overhead (~165 tokens/request)** — `<repo index="N">...</repo>` tags add ~33 tokens per repo × top_k=5. Compact numbered format eliminates this.
   - Fix: Replace XML with `1. owner/repo (1.2k★, category): Description`

2. **System prompt security section bloat (~40 tokens)** — 5 near-identical injection defense restatements. Consolidable to 1 clear rule.
   - Fix: Consolidate to single security instruction.

3. **Duplicate session history capping** — 8000-char cap applied in BOTH `_load_session_turns()` AND `_prepare_query()`. Redundant CPU + DRY violation.
   - Fix: Keep one, extract to `_MAX_SESSION_HISTORY_CHARS = 8000`.

### Privacy

4. **PII not redacted before query_log INSERT** — `redact_pii()` exists in `app/privacy.py` but is not called in `_log_query()`. Raw user questions (potentially containing emails, API keys) stored verbatim.
   - Fix: Apply `redact_pii(question)` before DB insert.

### Code Quality

5. **Dead function: `cosine_similarity()`** — Defined but never called. pgvector uses SQL `<=>` operator directly.
   - Fix: Delete.

6. **Unused import: `get_anthropic_key`** — Imported but never referenced.
   - Fix: Remove.

7. **Stale TODO comments** — Comments about "add periodic cleanup" for sessions, but retention IS implemented in `app/retention.py`.
   - Fix: Remove or update.

8. **Early-exit guard duplicated** — Same similarity check in `_run_query` and `event_generator`. Should be `_should_early_exit()` helper.
   - Fix: Extract helper.

9. **Hardcoded model strings in multiple places** — Model names appear in constants, pricing dict, and comments.
   - Fix: Consider enum (deferred — low ROI).

### Schema

10. **Missing index: `ask_sessions(created_at)`** — Retention purge query does full table scan.
    - Fix: Migration 023 adds index.

### Testing

11. **20 pre-existing injection test failures** — Tests expect 422 rejection but `_sanitize_question` is now log-only.
    - Fix: Update tests to match current defense strategy.

12. **Missing embedding warm-up timing** — Startup log doesn't show how long model loading takes.
    - Fix: Add `time.monotonic()` timing.

### Test Coverage Gaps (documented, not all addressed this round)

13. `_run_query` end-to-end with mocked Claude — no test
14. Streaming `event_generator` — no test
15. Smart-route SQL handlers — no test
16. `_select_model()` patterns — no test
17. `_validate_query_embedding()` — no test
18. Circuit breaker behavior — no test
19. Redis failure during cache ops — no test

## Agent Assignments

| Agent | Task | Branch |
|-------|------|--------|
| N | Source block compaction | `claude/feature/KAN-ask-source-compaction` |
| O | PII redaction + dead code + warm-up timing | `claude/feature/KAN-ask-code-hygiene` |
| P | Fix injection tests + DB index | `claude/feature/KAN-ask-test-fixes` |
| Q | System prompt compression + DRY refactors | `claude/feature/KAN-ask-prompt-optimization` |

## Estimated Impact

- Source compaction: ~165 tokens/request = ~$1.32/day at 10k queries
- System prompt compression: ~40 tokens/request = ~$0.24/day
- Combined with rounds 1-3: **~65% total Claude spend reduction**
