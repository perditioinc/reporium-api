# /intelligence/ask — Privacy, Retention, and RTBF

This document covers the privacy posture of the public `/intelligence/ask`
endpoint: what PII we persist, for how long, how to run the retention job,
and how to service a GDPR right-to-be-forgotten (RTBF) request. Closes #238.

## What we store

### `ask_sessions` (conversational memory)

Every turn of an `/intelligence/ask` session writes one row:

| column        | contents                                                   |
| ------------- | ---------------------------------------------------------- |
| `session_id`  | UUID supplied by the client (opaque session identifier)    |
| `turn_number` | 0-indexed position of the turn in the session              |
| `question`    | User's question, verbatim                                  |
| `answer`      | Assistant's reply, verbatim                                |
| `token_hash`  | SHA-256 of the caller's `X-App-Token` (ownership binding)  |
| `created_at`  | Row insertion timestamp                                    |

Retention window: **90 days** (configurable per invocation, bounded 7–365).
Rows are read only when the caller presents the same `X-App-Token` (or a
`NULL`-token legacy row), which prevents cross-tenant session reads — see
PR #242 for the ownership-binding work and `app.auth.hash_app_token`.

### `query_log` (analytics + cost tracking)

One row per completed `/ask` (and `/query`) call. The `question` column is
**PII-redacted** before insert by `app.privacy.redact_pii`:

- Email addresses → `[REDACTED_EMAIL]`
- Digit runs of 10+ → `[REDACTED_NUMBER]` (phones, SSNs, card numbers)
- API keys (`sk-…`, `ghp_…`, `xoxb-…`, etc.) → `[REDACTED_KEY]`

Redaction applies only to the persisted copy — the original text is still
sent to Claude so answer quality is unchanged. `query_log` has its own 90-day
retention purge (`POST /admin/retention/purge-query-logs`).

## Running the retention purge

`ask_sessions` retention is exposed as a callable admin endpoint. We do not
run an in-process scheduler (keeps infra cost at $0). Call the endpoint from
any external cron — Cloud Scheduler, GitHub Actions cron, a cron-like job
in a neighbouring service — on a **daily** cadence.

```bash
curl -X POST "https://reporium-api.example.com/admin/purge-ask-sessions?days=90" \
     -H "X-Admin-Key: ${ADMIN_API_KEY}"
# {"purged": 142, "max_age_days": 90}
```

`days` is clamped to `[7, 365]`. A daily cron with `days=90` is the
recommended default.

## Handling an RTBF request

To honour a GDPR right-to-be-forgotten request for a specific session:

```bash
curl -X DELETE "https://reporium-api.example.com/admin/ask-sessions/${SESSION_ID}" \
     -H "X-Admin-Key: ${ADMIN_API_KEY}"
# {"deleted": 3, "session_id": "…"}
```

The endpoint deletes every `ask_sessions` row matching `session_id` and is
idempotent — a second call returns `{"deleted": 0}`. `session_id` must be a
UUID; invalid values yield a 400.

If the user also wants their `query_log` rows removed, run the corresponding
SQL against `query_log` by `hashed_ip` (IPs are SHA-256 hashed at write
time; the caller must supply the IP for hashing).

## Session ownership binding (PR #242 recap)

Each `ask_sessions` row carries the SHA-256 hex of the `X-App-Token` that
created it. `_load_session_turns` filters by this hash so a different token
cannot read another session's turns even if it guesses the `session_id`.
Rows written before the migration (legacy) have `NULL` in `token_hash` and
remain readable for backward compatibility.
