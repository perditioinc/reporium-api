# Ask Endpoint Improvement Roadmap

**Status:** Draft for review · **Owner:** ASK workstream · **Last updated:** 2026-04-21

This roadmap enumerates concrete, evidence-backed improvements to the
`/intelligence/ask` pipeline. Every item cites the specific file and line
where the current behavior lives, names the trust vector it affects, and
carries an effort / risk estimate. Items are prioritized by expected
impact-per-effort, not by what's easiest.

No implementation is proposed here — this document is the "what to work on
next" input for Sprint 2+ planning.

---

## Trust vectors (refresher)

Four vectors the ASK answers need to earn:

| Vector | Question the user is really asking |
|---|---|
| **Provenance** | "Where did this claim come from — is it grounded in a repo I can verify?" |
| **Verifiability** | "Can I click through to the source and reproduce the reasoning?" |
| **Freshness** | "Is this from the last ingest, not from a stale cache or an old snapshot?" |
| **Agreement** | "Does the retrieval agree with the answer, or did the model overrule it?" |

Every improvement below maps to at least one of these.

---

## Priority 1 — Tune `_MIN_RETRIEVAL_SIMILARITY` based on golden set data

**Where:** [app/routers/intelligence.py:152](app/routers/intelligence.py#L152)

```python
_MIN_RETRIEVAL_SIMILARITY = 0.40
```

**Current behavior:** when no retrieved source has similarity ≥ 0.40, the
router skips the LLM call and returns `_EARLY_EXIT_ANSWER` (a canned
"not enough info" reply). Saves ~100% of token cost on junk queries
([intelligence.py:155](app/routers/intelligence.py#L155)).

**Why this is Priority 1:** 0.40 is a single magic number with no recorded
measurement behind the choice. Two failure modes:

1. **Too low** → LLM is called on weak retrievals, producing confabulated
   answers that erode provenance.
2. **Too high** → legitimate queries are early-exited, hurting recall and
   giving users a frustrating dead-end.

The Sprint 1 golden set expansion (PR #407) added `empty_result` (Q113-Q120)
and `out_of_graph` (Q078-Q087) cases that specifically exercise this
threshold. Once the eval harness runs against staging, we will have the
first real distribution of `max(sources.similarity)` across:

- Queries with good retrieval (`count`, `lookup`, `compare` categories)
- Queries that should bounce (`empty_result`, `out_of_graph`)
- Queries in the grey zone (`ambiguous`)

**Proposed work:** ship telemetry that logs the retrieval max-similarity
for every `/ask` request, then pick the threshold from the actual
bimodal distribution (the 5th percentile of good retrievals, or the
95th of bad, whichever is tighter).

- **Trust vectors:** provenance, agreement
- **Effort:** S (telemetry already exists in `token_observer`; add one
  metric + a query against `query_log`)
- **Risk:** low (pure measurement; threshold change is a config flip)

---

## Priority 2 — Tune `_SEMANTIC_CACHE_DISTANCE_THRESHOLD` on real hit-rate data

**Where:** [app/routers/intelligence.py:87-95](app/routers/intelligence.py#L87)

```python
_SEMANTIC_CACHE_DISTANCE_THRESHOLD = 0.15  # ~85% similarity
# ASK_CACHE_RELAXED=1 widens to 0.25 (~75%)
```

**Current behavior:** the semantic cache looks for a row in `query_log`
whose embedding is within 0.15 cosine distance of the new question.
A feature flag (`ASK_CACHE_RELAXED`) bumps this to 0.25 for live A/B.

**Why this matters:** the threshold is the main lever on the cache
hit-rate vs. false-hit-rate tradeoff. A false hit — serving a
cached answer to a subtly different question — is the single biggest
**agreement** risk in the whole pipeline: the user sees a confident
answer that's about something else.

**Concrete gap:** `test_ask_cache_effectiveness.py` exists but does not
record which historical questions produced which semantic-cache hits on
the golden set. Without that, we are tuning blind.

**Proposed work:**
1. Run the full 120-question golden set (PR #407) twice, back-to-back,
   against staging. The second run's semantic-cache hits tell us the
   baseline self-hit rate at threshold=0.15.
2. Repeat with `ASK_CACHE_RELAXED=1`.
3. For any "hit" where the cached question is semantically distinct
   (manual label), count as a false hit. Pick the threshold that
   maximizes `(true_hits - N * false_hits)` for some N aligned with
   the cost-of-a-bad-answer we're willing to pay.

- **Trust vectors:** agreement, freshness (stale cache entries)
- **Effort:** M (two full eval runs + manual labeling)
- **Risk:** low (threshold is already runtime-flippable)

---

## Priority 3 — Measure route accuracy of the smart-router

**Where:** [app/routers/intelligence.py:201-297](app/routers/intelligence.py#L201)
(17 route regexes) and [app/routers/intelligence.py:489](app/routers/intelligence.py#L489)
(`_try_smart_route_inner`)

**Current behavior:** 17 compiled regex patterns (`_ROUTE_COUNT`,
`_ROUTE_COUNT_CATEGORY`, `_ROUTE_LIST_CATEGORIES`, ...) try to answer
factual questions from SQL and skip the LLM. Order-of-evaluation and
overlap between patterns is not currently tested for correctness —
e.g. `_ROUTE_COUNT_CATEGORY` at L205 and `_ROUTE_COUNT_LANGUAGE` at
L226 could both match "how many repos use Python" depending on order.

**Why this matters:** the smart router is the single largest **cost**
lever. A false smart-route (regex matches but the SQL handler returns
empty/wrong) sends the user to an unhelpful canned answer with no LLM
fallback. A missed smart-route (should have matched but didn't) pays
the full LLM cost unnecessarily.

The Sprint 1 golden set now carries `expected_route` as a second-class
field (not asserted in Sprint 0 per the comment at
[tests/golden/ask_questions.yaml:10](tests/golden/ask_questions.yaml#L10)).
This roadmap proposes **promoting route accuracy to an assertable
metric in Sprint 2**.

**Proposed work:**
1. Add `actual_route` to the eval harness output
   ([tests/golden/test_ask_eval.py:241-254](tests/golden/test_ask_eval.py#L241))
   by capturing the route field already present in the smart-route
   response shape (`{"answer", "sources", "route"}`).
2. Compute per-category precision / recall vs. `expected_route`.
3. File a ticket per regex pattern with recall < 80% on the golden set.

- **Trust vectors:** agreement, verifiability (route is surfaced to client)
- **Effort:** M (harness change + Sprint 2 scope bump)
- **Risk:** low (measurement only)

---

## Priority 4 — Harden `_sanitize_question` beyond log-only

**Where:** [app/routers/intelligence.py:1149-1172](app/routers/intelligence.py#L1149)

**Current behavior:** `_sanitize_question` runs a regex denylist
(`_INJECTION_PATTERNS` at L63) and **logs a warning** but returns the
question unchanged. The comment at L1156-1166 explicitly records that
hard-blocking was abandoned in favor of structural mitigations
(delimiter-based system prompts, `<question>` tags).

**Why this still matters:** the Sprint 1 adversarial golden entries
(Q051-Q065, PR #407) will all pass straight through `_sanitize_question`
by design. Without a **measurement** of how often these patterns actually
trigger LLM compliance, we have no way to know whether the structural
mitigation is sufficient.

**Concrete gap:** `test_off_topic_filter.py` tests that adversarial
inputs are rejected by `_is_off_topic`, but doesn't track what happens
to probes that *pass* `_is_off_topic` (repo-signal bypass) and reach the
LLM.

**Proposed work:**
1. When `_sanitize_question` flags a probe, increment a Prometheus
   counter by pattern class (`instruction_override`, `role_override`,
   `exfiltration`, `jailbreak`).
2. On every `/ask` response, inspect the answer with a small second
   regex for refusal markers. Ratio of flagged-probes to clean
   refusals is the **provenance leakage rate**.
3. If leakage > 1%, promote the affected pattern from log-only to
   hard-block.

- **Trust vectors:** provenance, agreement
- **Effort:** M (instrumentation + weekly review cadence)
- **Risk:** medium (adding hard blocks later risks new false positives —
  must be driven by real rates, not intuition)

---

## Priority 5 — Reduce session history compaction loss

**Where:** [app/routers/intelligence.py:1500-1523](app/routers/intelligence.py#L1500)

```python
# oldest turns collapse the assistant answer to an 80-char preview
preview = a.strip().replace("\n", " ")[:80]
```

**Current behavior:** all but the most recent prior turn have the
assistant answer truncated to 80 chars and prefixed with `[prior: ...]`.
Cuts session history size by 60-70% at the cost of discarding any
detailed reasoning that happened earlier in the conversation.

**Why this matters:** the Sprint 1 follow-up entries (Q103-Q112, PR #407)
specifically exercise multi-turn chains. Q105 ("What license does the
third one use?") needs the ORDER of the list from Q103 to remain
recoverable. An 80-char preview will almost certainly truncate that
list before the third item.

**Proposed work:**
1. Instrument the follow-up entries: measure `must_mention` hit-rate
   on follow-up turns as a function of session-history char budget
   (128, 256, 512 chars per prior turn).
2. Check whether the current `_MAX_SESSION_HISTORY_CHARS=8000` cap is
   actually hit by realistic traffic (probably not — 2-3 turns at
   ~500 chars each is well under 8k).
3. If budget allows, raise the per-turn preview from 80 to 256-512 chars
   and keep the most-recent turn verbatim.

- **Trust vectors:** agreement (continuation is a form of agreement
  with prior state)
- **Effort:** S (two numbers + one config change)
- **Risk:** low-medium (more context = more token cost per follow-up;
  must be measured)

---

## Priority 6 — Freshness signal in the answer itself

**Where:** no current implementation; gap identified against trust vectors

**Current behavior:** the ASK response includes `sources[].stars` and
`sources[].description` but no `last_ingested_at` or `last_github_push`
timestamp. The user has no way to tell whether the answer reflects the
state of the world as of today or as of two weeks ago.

**Why this matters:** freshness is the one trust vector that cannot be
recovered from the output alone. A month-stale cached answer about an
actively-moving repo (e.g. LangChain releases every few days) can be
confidently wrong.

**Proposed work:**
1. Add `last_ingested_at` to the `SourceRepo` response model
   ([intelligence.py:1649](app/routers/intelligence.py#L1649)).
2. Surface it in the `sources` array; render it in the frontend next
   to the repo name as "updated 3 days ago".
3. On the semantic-cache hit path, also surface `cache_age_hours` so
   users know whether the answer was re-used from an older run.

- **Trust vector:** freshness
- **Effort:** S for the API side, M for the frontend rendering
- **Risk:** very low (additive field)

---

## Priority 7 — Route accuracy for streaming responses

**Where:** [app/routers/intelligence.py:1672-1689](app/routers/intelligence.py#L1672)
(`StreamEvent` model) and the `/ask/stream` endpoint further down

**Current behavior:** the streaming endpoint emits `sources`, `token`,
`done`, `error` events. No unit tests cover the streaming path.

**Why this matters:** if the non-streaming path diverges from the
streaming path (e.g. different early-exit behavior, different model
selection), users get different answers depending on which endpoint
their client uses. That's a silent **agreement** failure.

**Proposed work:** add a single parametrized test that calls both
endpoints with the same question and asserts the terminal answers match
(mocked Anthropic response). Deferred to Sprint 3 — lower priority than
the measurement-driven items above.

- **Trust vector:** agreement
- **Effort:** M (mock infrastructure + fixture)
- **Risk:** low

---

## Deferred items (tracked but not scheduled)

These were identified but do not meet the bar for Sprint 2-3 inclusion:

- **`_MODEL_PRICING` drift** ([intelligence.py:1367-1369](app/routers/intelligence.py#L1367)).
  Hard-coded per-1M-token prices will go stale. Fix: pull from a single
  `app/pricing.py` that is updated when Anthropic publishes changes.
- **`_format_stars` lacks an `M` tier.** Today 10M stars render as
  `"10000.0k"`. Cosmetic; low-traffic.
- **`_ROUTE_COMPARISON` regex is naive** ([intelligence.py:254](app/routers/intelligence.py#L254)):
  matches `\S+\s+(?:and|vs)\s+\S+`, so `"Python and Rust"` will try to
  look up repos named "Python" and "Rust" against
  `_REPO_INFO_BLACKLIST`. Fix requires NER, out of scope for now.
- **Session token-hash check** ([intelligence.py:1478-1494](app/routers/intelligence.py#L1478))
  silently falls back to NULL-only rows when no token is presented.
  Intentional per Issue #235 comment, but should be re-reviewed after
  token governance rolls out.

---

## Non-goals

- **Rewriting the router to LLM-based intent classification.** The
  17-regex smart router is slow to maintain but fast at runtime and
  cheap to debug. Replacement is a Sprint 5+ conversation.
- **Fine-tuning.** Out of scope for the Anthropic-only stack we ship on.
- **Adding new product features (e.g. code generation).** `_is_off_topic`
  explicitly bounces these; keep it that way.

---

## Appendix — evidence-gathering before Sprint 2

Before any of the above are scoped, run these measurement tasks:

1. Full 120-question eval against staging — produces the P1 and P2
   distributions.
2. Run twice back-to-back to populate semantic-cache self-hit data.
3. Repeat with `ASK_CACHE_RELAXED=1`.
4. Export `query_log` sample (last 7 days) → histogram of
   `max(sources.similarity)`. Sanity check P1 against real traffic.

All four are gated behind `RUN_GOLDEN_EVAL=1` + a working staging API
token. No changes to application code required.
