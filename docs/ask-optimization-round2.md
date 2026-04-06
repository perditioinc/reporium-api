# /intelligence/ask Optimization — Round 2 Audit (2026-04-05)

Second-pass audit after round-1 improvements shipped (PRs #231–#234, #242).
Focus: $0 / free-tier-only improvements the first round missed.

## Current State (rev 00144-t6b)

Already shipped:
- Prompt caching on system prompt + sources block (ephemeral)
- pgvector semantic cache (threshold 0.15)
- Smart routing for simple questions (SQL shortcut, no LLM)
- Haiku-first tiered routing (Sonnet only for complex patterns)
- Query normalization for cache keys (synonyms, punctuation, whitespace)
- Context hygiene (240-char description cap, minimal fields)
- Session memory compaction (older turns to 80-char preview)
- top_k=5 default
- Token-spend observer + `/metrics/spend` + $10/day budget guardrail
- Numeric golden-set CI gate (0.7 threshold)
- Cloud Run concurrency=200, max-instances=10 ($0)
- Session ownership binding via `ask_sessions.token_hash`
- Timing-safe auth comparisons
- Prompt-injection structured wrapping + system-prompt defense

## Round-2 Findings

### Cost — hardcoded-ceiling bloat
1. **`max_tokens=1024`** at both Claude call sites. Golden-set avg answer = ~280 tokens. This ceiling is hit by runaway generations and directly wastes output tokens (5× input price). Fix: dynamic cap — Haiku=512, Sonnet=768.
2. **No `stop_sequences`** passed to the Claude call. Models can emit XML trailers or disclaimers. Adding `["</answer>", "\n\nNote:", "\n\nDisclaimer:"]` trims tail output.
3. **Session memory default** _MAX_SESSION_TURNS=3 → cap at 8000 chars. Sessions rarely benefit from >1 prior turn in a Q&A context. Drop to 2.

### Cost — cache effectiveness
4. **Semantic cache threshold 0.15** (cosine distance) ≈ 0.85 similarity. Tight. At 0.25 (≈0.75) plus normalization, we'd likely 2–3× hit rate. Ship behind a feature flag `ASK_CACHE_RELAXED=1` for A/B.
5. **Smart-route Redis TTL 300s** for simple counting questions that change hourly at most. Bump to 3600s.
6. **Streaming endpoint does NOT write to Redis cache after answering.** First user hitting a new question via `/ask/stream` pays full price, second user hitting `/ask` gets a miss. Align both.
7. **No negative caching.** Failed or low-quality answers are re-generated each retry. Cache empty-answer for 60s.

### Cost — early exit
8. **No similarity-based early exit.** If top-1 pgvector similarity < 0.4, the answer will be low quality. Return a deterministic "I don't have enough relevant data" response with zero Claude call. Budget-friendly AND quality-friendly.

### Cost — observability reuse
9. **Cache invalidation on taxonomy rebuild is not wired.** Nightly rebuilds change repo context but semantic cache serves stale answers for up to 24h. Free fix: on rebuild, UPDATE ask_cache expires_at = NOW() (if such a table exists) or bump via TTL field.

### Security / Privacy

10. **#236** `/metrics/slo`, `/metrics/spend`, `/metrics/latest` still unauthenticated. Add an optional admin-key gate that falls back open in dev.
11. **#238** `ask_sessions` has no retention job. GDPR RTBF gap. Add a nightly DELETE WHERE created_at < NOW() - INTERVAL '90 days' task.
12. **Question text logged in plaintext** to `query_logs` table. GDPR minor risk for PII inside questions. Hash or redact sensitive-looking substrings (emails, phone numbers, API keys).
13. **`_sanitize_question` logs every injection attempt** unbounded. DoS amplifier for log volume. Rate-limit log emissions by hashed IP.

## Priority Order (by ROI at $0)

| # | Change | Rough savings | Risk |
|---|--------|---------------|------|
| 1 | Dynamic max_tokens + stop_sequences | –20% output tokens | very low |
| 2 | Streaming cache-write + TTL bumps | –15% LLM calls | very low |
| 3 | Negative caching + early-exit low-sim | –10% LLM calls, better UX | low |
| 4 | Semantic threshold A/B (0.15→0.25 flag) | –20% LLM calls if hit lands | medium (quality) |
| 5 | Session turns 3→2 default | –500 tokens/session | very low |
| 6 | ask_sessions retention job | n/a (privacy) | very low |
| 7 | Metrics auth gate | n/a (security) | very low |
| 8 | Question text PII redaction | n/a (privacy) | low |

Cumulative estimated savings: **30–45% additional reduction on Claude spend** on top of round-1.
