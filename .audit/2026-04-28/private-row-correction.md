# Private-row correction — bleed-stop runbook

**Incident:** 2026-04-27 ~07:15 UTC. `perditioinc/hippo-harvest-assignment` (a
GitHub-private repo) was ingested with `is_private = false` and surfaced on
every public read endpoint of reporium-api (and was baked into the static
`reporium.com/data/library.json` artifact).

**Lane:** This document covers Lane 1 only — corrective DB write to flip the
single bad row to `is_private = true` and invalidate caches. Static-artifact
regeneration (Lane 2), ingestion RCA (Lane 3), and audit-contract fix (Lane 4)
are tracked separately.

## What this PR ships

A single new admin endpoint:

```
POST /admin/repos/mark-private
Headers: X-Admin-Key: <ADMIN_API_KEY>
Body:    {"owner": "perditioinc", "name": "hippo-harvest-assignment",
          "dry_run": true}
```

- **Dry-run** (default) returns match info — id, owner, name,
  `current_is_private`, `ingested_at`, and the cache prefixes that *would* be
  invalidated. Read-only.
- **Apply** (`dry_run: false`) sets `is_private = true`, invalidates 11 cache
  prefixes via `redis_cache.clear_prefix()`, writes an `AuditLog` row, and
  returns `applied: true`.
- **Idempotent** — applying to an already-private row is a no-op success.
- **Defensive** — refuses to mutate if more than one row matches
  (`repos.name` is `UNIQUE`, so this is impossible-but-guarded).

## Why a new endpoint and not the existing /ingest/repos

`/ingest/repos` *can* mark a row private via the sticky logic introduced in
PR #414, but only when called with the *full* repo payload — re-fetching that
from production for incident response is awkward and slow. The new endpoint
takes only `owner+name`, supports dry-run preview, and bundles the cache
invalidation that `/ingest/repos` does not. Pattern is reusable for future
incidents (Cloud SQL is private-IP only, so direct `UPDATE` from the
operator host is not possible — see `reference_cloud_sql_private_ip.md`).

## Cache prefixes invalidated on apply

Every prefix is matched via `redis_cache.clear_prefix()` (Redis SCAN — bounded
I/O even on production-sized key spaces). Defined in
`app/routers/admin_visibility.py::INVALIDATION_PREFIXES`:

| Prefix             | Surfaces it covers |
|--------------------|----|
| `library:`         | `/library`, `/library/full` paginated pages |
| `repos:`           | `/repos` list + `/repos/{name}` detail |
| `graph_`           | `/graph/edges`, `/graph/subgraph`, `/graph/clusters`, `/graph/search` |
| `trending:`        | `/intelligence/trending` |
| `ecosystem:`       | `/intelligence/ecosystem/{name}` |
| `intelligence:`    | portfolio insights, category momentum, similar |
| `signals:`         | taxonomy gaps, stale repos, velocity leaders |
| `compare:`         | `/compare` results |
| `similar:`         | `/intelligence/similar/{name}` |
| `smart_route:`     | LLM router cache (may cite the repo) |
| `llm_response:`    | LLM answer cache (same) |

Per-repo `graph_edges:{rid}` keys (set in `intelligence.py` via the in-memory
`cache`) are cleared via the `graph_` prefix sweep when both layers point
to the same Redis instance, which is the production configuration.

## Bleed-stop runbook

Run from any host that can hit production (`reporium-api-573778300586.us-central1.run.app`).

**Step 1 — Wait for deploy.** This PR must be merged and Cloud Run must roll
out before the endpoint exists. Verify with:

```bash
curl -I -X POST https://reporium-api-573778300586.us-central1.run.app/admin/repos/mark-private
# expect 422 (validation error — no body), NOT 404
```

**Step 2 — Dry-run preview.**

```bash
curl -sS -X POST \
  -H "X-Admin-Key: $REPORIUM_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"owner": "perditioinc", "name": "hippo-harvest-assignment", "dry_run": true}' \
  https://reporium-api-573778300586.us-central1.run.app/admin/repos/mark-private | jq .
```

Expected response shape:

```json
{
  "match": {
    "id": "a01e40b2-0997-4a27-8b82-9f52c6a0fd81",
    "owner": "perditioinc",
    "name": "hippo-harvest-assignment",
    "current_is_private": false,
    "ingested_at": "2026-04-27T07:15:14+00:00"
  },
  "match_count": 1,
  "applied": false,
  "would_invalidate_prefixes": [...]
}
```

**Confirm before proceeding:**
- `match_count == 1` — exactly one row, no broad delete possible
- `match.id == "a01e40b2-0997-4a27-8b82-9f52c6a0fd81"` (cross-check against
  the production probe captured during the prior reconnaissance session)
- `match.current_is_private == false` (confirms this is the bad row)

If any check fails, **stop**. Do not run apply.

**Step 3 — Apply.**

```bash
curl -sS -X POST \
  -H "X-Admin-Key: $REPORIUM_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"owner": "perditioinc", "name": "hippo-harvest-assignment", "dry_run": false}' \
  https://reporium-api-573778300586.us-central1.run.app/admin/repos/mark-private | jq .
```

Expected response: `applied: true`, `match.current_is_private: true`,
`invalidated_prefixes` listing the 11 prefixes from the table above.

**Step 4 — Verify production no longer leaks.**

```bash
# Detail endpoint — must 404
curl -sS -o /dev/null -w "%{http_code}\n" \
  https://reporium-api-573778300586.us-central1.run.app/repos/hippo-harvest-assignment
# expect: 404

curl -sS -o /dev/null -w "%{http_code}\n" \
  https://reporium-api-573778300586.us-central1.run.app/repos/perditioinc/hippo-harvest-assignment
# expect: 404

# Search must not return it
curl -sS "https://reporium-api-573778300586.us-central1.run.app/search?q=hippo" | jq '. | length'
# expect: prior count minus 1

# /library/full must not list it
curl -sS "https://reporium-api-573778300586.us-central1.run.app/library/full?page=1&page_size=2000" | \
  jq '.repos[] | select(.name == "hippo-harvest-assignment")'
# expect: empty (no output)
```

**Step 5 — Confirm in DB via existing audit endpoint.**

```bash
curl -sS -H "X-Admin-Key: $REPORIUM_ADMIN_KEY" \
  "https://reporium-api-573778300586.us-central1.run.app/admin/audit?endpoint=admin.mark_private&limit=5" | jq .
```

Look for the row with `request_summary` containing
`name=hippo-harvest-assignment dry_run=False prior_is_private=true`.

## Out of scope (Lane 2+)

- **Static artifact** at `reporium.com/data/library.json` is **frozen** until
  the next frontend rebuild + deploy. The API fix does not touch the baked
  JSON. Lane 2 (reporium repo) blocks the artifact from emitting private
  rows in the first place.
- **Ingestion RCA** — why was the row inserted with `is_private=false` for a
  private GitHub repo? Lane 3 (reporium-ingestion repo).
- **Audit contract** — why didn't `reporium-audit/checks/contract.py`
  detect this in 22 hours? Lane 4 (reporium-audit repo).

## Files in this PR

- `app/routers/admin_visibility.py` — new admin router (110 lines)
- `app/main.py` — `+1` import, `+1` `include_router`
- `tests/test_admin_mark_private.py` — 9 tests (1 route registration, 8 DB-
  dependent: dry-run preview, apply mutation, cache invalidation calls,
  idempotency, audit-log row, dry-run-does-not-invalidate, missing
  X-Admin-Key returns 403, 404 on unknown owner+name)

## Verification status (this PR, pre-deploy)

- [x] Tests written first; route-registration test went red, then green.
- [x] Lint clean (`ruff check`) on changed files.
- [x] Full test suite passes: 576 passed, 254 skipped (DB-dependent), 0 failed.
- [ ] DB-dependent tests run in CI (will fire on PR open).
- [ ] Endpoint exercised against production after deploy (Step 2-5 above).

## After Lane 1

Lane 1 alone does NOT close the visible leak. Even with the row corrected:
- The static artifact at reporium.com still serves the old JSON.
- The next ingestion run could re-introduce the row if the source bug
  persists (it cannot flip is_private back to false because of PR #414's
  sticky logic, but it can still surface the row in other ways).

Lane 2 + Lane 3 must land before declaring the incident closed.
