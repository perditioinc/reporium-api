# Ask Eval Harness

Baseline measurement infrastructure for the Reporium `/intelligence/ask`
endpoint. Introduced in **Sprint 0** so later sprints (router layer, semantic
cache, evidence packs) have a numerical reference to beat.

## What this is

- A hand-curated **golden set** of 50 natural-language questions spanning the
  query shapes users actually ask (count, lookup, compare, recommend,
  taxonomy, trend, graph, health, synthesis).
- A **pytest runner** that fires each question at a running API, records
  latency, response size, `must_mention` hit rate, and dumps a JSON artifact
  under `.results/latest.json` for downstream diffing.
- A stable schema so future sprints can add assertions (route accuracy, cache
  hit rate, cost regression) without rewriting the dataset.

**All 50 questions are currently `source: synthetic`.** Replace them with
redacted real traffic from `query_log` as it becomes available.

## 🔒 Privacy boundary (READ BEFORE ADDING QUESTIONS)

This golden set is **synthetic-only** and safe for public CI.

**Do NOT add to this file**:
- Production logs or real user questions
- PII, IP addresses, session identifiers, or any user-derived strings
- API tokens, keys, or any credential material
- Customer-specific prompts or proprietary query patterns

Sensitive or user-derived eval examples belong in the private overlay repo
**`perditioinc/reporium-evals`**, which is reserved for exactly that purpose.
The runner in this repo will be extended in a later sprint to merge the
public base with the private overlay at load time.

If you are unsure whether a question belongs here or in the private overlay,
default to the private overlay.

## How to run

The eval is **not** part of the default `pytest` run because it makes real
network + LLM calls. Opt in with an env var:

```bash
RUN_GOLDEN_EVAL=1 \
ASK_EVAL_BASE_URL=http://localhost:8000 \
ASK_EVAL_APP_TOKEN=<your X-App-Token> \
    pytest tests/golden/test_ask_eval.py -v -s
```

- `ASK_EVAL_BASE_URL` defaults to `http://localhost:8000`.
- `ASK_EVAL_APP_TOKEN` is the `X-App-Token` header value (`APP_API_TOKEN` on
  the server side). Required in non-dev environments.
- `-s` is recommended so the per-question table prints inline.

Results are written to `tests/golden/.results/latest.json` (gitignored).

### Rate limit pacing

The `/intelligence/ask` endpoint is rate-limited at **6 requests/minute and
60 requests/day per IP**. Firing all 50 questions in a tight loop will trip
the limiter and record false failures (and burn the daily quota on noise).
The runner exposes two env vars to pace itself:

| env var | default | purpose |
| --- | --- | --- |
| `ASK_EVAL_SLEEP_SECONDS` | `11` | Seconds to sleep between requests. `11` keeps us safely under the 6/min ceiling. Set to `0` to opt out entirely (e.g. local runs against a mock). |
| `ASK_EVAL_MAX_QUESTIONS` | unset (unlimited) | Cap on how many questions the run sends. Useful for a cheap smoke pass before spending the full daily quota. |

Recommended **first baseline** against a live API — short pass to confirm
the harness is wired correctly before committing 50 questions of quota:

```bash
ASK_EVAL_MAX_QUESTIONS=10 \
ASK_EVAL_SLEEP_SECONDS=12 \
RUN_GOLDEN_EVAL=1 \
ASK_EVAL_BASE_URL=http://localhost:8000 \
ASK_EVAL_APP_TOKEN=<your X-App-Token> \
    pytest tests/golden/test_ask_eval.py -v -s
```

A full 50-question run at the default 11-second pacing takes ~9 minutes of
wall clock and uses 50 of the 60 daily requests — plan accordingly.

## Question schema

```yaml
- id: Q001                         # stable opaque identifier
  question: "..."                  # the user-facing question
  category: recommend              # count|lookup|compare|recommend|taxonomy
                                   # trend|graph|health|synthesis
  expected_route: recommend        # what Sprint 1's router should pick
                                   # (NOT asserted yet — placeholder)
  must_mention: [lowercased]       # substrings required in answer.lower()
  must_not_mention: []             # substrings that must NOT appear
  expected_source_ids: []          # optional "owner/name" repo slugs
  source: synthetic                # synthetic | user_drafted
  notes: "rationale / origin"
```

Field-by-field:

| field | asserted in Sprint 0? | notes |
| --- | --- | --- |
| `id` | no | Stable across sprints; never reused. |
| `question` | yes (sent to API) | |
| `category` | no | Used for grouping in summary output. |
| `expected_route` | **no** | Router doesn't exist yet — Sprint 1 enables this. |
| `must_mention` | measured, not gated | Aggregated into `mention_hit_rate`. |
| `must_not_mention` | measured, not gated | Violations counted, not fatal. |
| `expected_source_ids` | no | For Sprint 2 precision checks. |
| `source` | no | Provenance tag; replace synthetic with real over time. |
| `notes` | no | Freeform. |

## Adding a question

Append to `ask_questions.yaml`:

```yaml
- id: Q051
  question: "Your new question here."
  category: lookup
  expected_route: lookup
  must_mention: ["grounding", "substring"]
  must_not_mention: []
  expected_source_ids: []
  source: user_drafted
  notes: "Where this came from."
```

Keep `must_mention` lowercased — the runner lowercases the answer before
checking membership.

## Roadmap

| Sprint | What this harness gains |
| --- | --- |
| 0 (this PR) | Baseline latency, response size, mention-hit rate. No quality floor. |
| 1 | Enable `expected_route` assertion once router lands. Set minimum route accuracy threshold off Sprint 0 data. |
| 2 | Assert `expected_source_ids` precision once evidence packs exist. |
| 3 | Add cost ceiling assertion using `tokens_used` from response body. |

## Why isn't this in CI?

Two reasons:

1. **Cost.** Every run bills real Anthropic tokens — 50 questions times the
   Sonnet/Haiku mix is non-trivial.
2. **Flakiness.** Model nondeterminism + network latency means a blind CI
   integration would flap. The right home is a manual trigger or a nightly
   workflow with budget alerts; that's a Sprint 1+ decision.
