# Private-leak hotfix — integration closeout

Reconciles three concurrent reporium-api branches into one PR-ready branch:
`claude/hotfix/private-leak-integrated-2026-04-28`.

## Source branches reviewed

| Branch | Worktree | Status before reconciliation |
|---|---|---|
| `hotfix/2026-04-28-api-private-and-fork` | `C:\DEV\PERDITIO_PLATFORM\reporium-api` (primary) | 2 commits ahead of main (already merged content), 11 modified files, 3 new files (`db_filters.py`, `source_canonical.py`, `tests/test_private_and_fork_hotfix.py`), 1 audit doc — all uncommitted. |
| `claude/hotfix/private-repo-centralized-filter` | `.worktrees/reporium-api-private-leak-2026-04-28` | 0 commits, 4 modified routers, 2 new files (`visibility.py`, `tests/test_no_private_leak.py`) — all uncommitted. |
| `claude/hotfix/private-row-correction` | `.worktrees/reporium-api-private-row-correction-2026-04-28` | 0 commits, 1 modified file (`main.py`), 3 new files (`admin_visibility.py`, test, audit) — all uncommitted. |

**Nothing was discarded.** All three source worktrees and the primary
checkout's working tree are unchanged. Reconciliation was a strict
file-copy + targeted-edit workflow into a fresh worktree based on
`origin/main @ 9067c3c`.

## Single coherent PR story

0. **`/library/full` exposes `isPrivate` on every wire repo (camelCase, bool).**
   Required by Lane 2's `validate-privacy.ts` (frontend `prebuild` gate)
   and Lane 4's `reporium-audit` `check_contract` so downstream checks
   have a structural signal to assert against. Field is the inverse of
   the prior (#414) shape — present-and-`False` instead of stripped.
   Updates in `app/routers/library_full.py` (`_build_enriched_repo`,
   `_fetch_page_repos` SELECT, `/forks` shape) plus a refreshed
   `tests/test_library_full.py` privacy contract that pins
   "every repo carries `isPrivate: false`" instead of the old
   "field is absent".

1. **Public APIs do not emit private repos.** Every router that reads from
   the `repos` table now goes through `app.db_filters.public_repo_filter()`
   (replaces `Repo.is_private == False` literals scattered across 10
   routers; new endpoints can no longer skip the predicate by accident
   because the static guards in `tests/test_no_private_leak.py` will
   refuse to merge a regression).

2. **`/repos/{owner}/{name}` and `/forks` no longer leak.** Two endpoints
   the prior hotfixes missed are now filtered (the `/repos/{owner}/{name}`
   leak is the one that surfaced
   `perditioinc/hippo-harvest-assignment` live to unauthenticated callers
   on 2026-04-28).

3. **ASK related-edge hydration does not leak private repos.** The
   `/intelligence/ask` related-edges SQL had three privacy gaps: the JOINs
   on `r1` and `r2`, plus the lateral `repo_embeddings` subquery that
   could pick a private row as nearest-neighbour and only filter at the
   outer JOIN, distorting `LIMIT` semantics. All three are now guarded.

4. **ASK source canonicalization (Lane 5).** Smart-route SQL now selects
   `forked_from` and routes through `_build_smart_route_source` /
   `app.source_canonical.canonical_owner_name()` so that
   `perditioinc/markitdown` is cited as `microsoft/markitdown` (the actual
   project the user would clone). Pure-function helper with rigorous
   fallback semantics — never invents a parent.

5. **Operator can correct a leaked row safely.** New
   `POST /admin/repos/mark-private` endpoint (admin-key-gated) supports
   dry-run preview, exact-one-match guard, post-mutation cache
   invalidation across 11 prefixes, and `audit_logs` row insertion.

6. **`recommendations.py` defense-in-depth.** `_SIMILAR_SQL` now requires
   both `seed_r.is_private = false` AND `r.is_private = false` so the
   join-level filter is enforced even if a future name-lookup change
   skips the pre-flight visibility check.

## File-level reconciliation map

### From primary `hotfix/2026-04-28-api-private-and-fork`

| File | Why kept |
|---|---|
| `app/db_filters.py` (renamed merge target — see below) | Centralized predicate. Lazy `Repo` import avoids cycles. |
| `app/source_canonical.py` | Pure function for fork canonicalization. Clean, well-tested. |
| `app/routers/repos.py` | Migration to `public_repo_filter()`. |
| `app/routers/library.py` | Migration. |
| `app/routers/library_full.py` | Adds `is_private = false` to `/forks` (NEW privacy fix; not in any other branch). |
| `app/routers/compare.py` | Migration. |
| `app/routers/search.py` | Migration. |
| `app/routers/trends.py` | Migration. |
| `app/routers/wiki.py` | Migration. |
| `app/routers/dependencies.py` | UUID-oracle fix on `/repos/{id}/dependencies` AND `IS_PUBLIC` filter on `/dependencies/dependents`. |
| `app/routers/mentions.py` | UUID-oracle fix on `/repos/{id}/mentions`. |
| `app/routers/intelligence.py` | Fork canonicalization in smart-route SQL handlers (selects + uses `_build_smart_route_source`). |
| `tests/test_private_and_fork_hotfix.py` | 6 unit tests for `canonical_owner_name`; integration tests for /repos, /search, /library, /forks, ASK fork canonicalization. |
| `.audit/2026-04-28/api-private-and-fork-hotfix.md` | Original investigation that discovered the live `/repos/{owner}/{name}` leak. |

### From `claude/hotfix/private-repo-centralized-filter` (Lane 1)

| File | Why kept |
|---|---|
| Layered onto `app/routers/intelligence.py` (lines 2823-2826) | Related-edges hydration JOINs on `r1`/`r2` + lateral subquery `r_inner` — three privacy filters the user's branch did NOT include. Without these, the LLM "Related repos:" context can include private repo names even after row-level correction. |
| `app/routers/recommendations.py` | `seed_r.is_private = false` defense-in-depth on `_SIMILAR_SQL`. Unique to Lane 1. |
| `tests/test_no_private_leak.py` | Static guards (regex audits of source files for the privacy filters) + direct SQL probe of the related-edges hydration query. The static guards run *without* a DB and turn the test suite red the moment a future edit drops a filter. |
| Folded into `app/db_filters.py`: `sql_public_filter(alias)`, `public_repos_select()` | Aliased SQL helper with identifier-validation injection guard. Augments the user's `db_filters.py` (not a separate file). |

### From `claude/hotfix/private-row-correction` (Lane data-correction)

| File | Why kept |
|---|---|
| `app/routers/admin_visibility.py` | New admin endpoint with dry-run + apply + cache invalidation + audit logging. |
| `app/main.py` | Wires `admin_visibility.router`. |
| `tests/test_admin_mark_private.py` | Route registration test (no DB) + 8 DB-dependent behavioural tests. |
| `.audit/2026-04-28/private-row-correction.md` | Operator runbook for executing the mark-private call after deploy. |

### Dropped intentionally

| File | Why dropped |
|---|---|
| `app/visibility.py` (Lane 1) | Replaced by `app/db_filters.py` — same purpose, user's name kept. The Lane 1 module's `IS_PUBLIC` / `public_repos_select` / `sql_public_filter` were folded into `db_filters.py` so callers get one canonical name. |

### Decisions noted

- **Module name**: chose `app.db_filters` (user's name) over `app.visibility`
  (Lane 1's name). Reason: more files in the repo already import the
  user's name, the migration of 10 routers points at it, and the user's
  audit doc references it.
- **API style**: kept the user's `public_repo_filter()` *function* (lazy
  import) rather than Lane 1's `IS_PUBLIC` *constant* (eager import).
  Reason: the user's lazy pattern avoids potential circular-import edge
  cases. Functionally equivalent at the call site.
- **ASK fork canonicalization**: included (Lane 5). The user's directive
  said "keep only if already implemented cleanly" — it is. `canonical_owner_name`
  is a pure function with explicit fallback semantics that never invents
  data. Held-back features (Lane 6 frontend card-click, Lane 2 static
  artifact, Lane 3 ingestion RCA, Lane 4 audit fix) are outside this PR.

## Verification

| Check | Status |
|---|---|
| `python -m pytest tests/ --ignore=tests/load --ignore=tests/golden -q` | **588 passed, 279 skipped, 0 failed.** Up from Lane 1's 581 (added: 6 canonical-owner tests, 1 admin route registration). |
| Targeted: `tests/test_no_private_leak.py` + `test_admin_mark_private.py` + `test_private_and_fork_hotfix.py` | **13 passed, 33 skipped (DB-dependent), 0 failed.** |
| `ruff check` on changed files | **clean.** Pre-existing warnings in `intelligence.py` (unused imports, f-string) are upstream from primary; this reconciliation does not introduce new lint. |
| Module imports from a clean Python session | **all ok**: `db_filters`, `source_canonical`, `admin_visibility`, every modified router. |
| `app.db_filters.public_repo_filter()` returns a SQL predicate | `repos.is_private = false` ✓ |
| `app.db_filters.sql_public_filter("r1")` returns the aliased fragment | `r1.is_private = false` ✓ |
| `source_canonical.canonical_owner_name(forked_from="m/x", own_owner="p", own_name="x")` | `("m", "x")` ✓ |
| Static guard: `intelligence.py` related-edges has the three filters | passes (test asserts the regex match) ✓ |

## Deploy / runbook order (after PR merge)

1. Merge this PR. Cloud Run rolls out automatically.
2. Verify the new endpoint exists with `curl -I -X POST .../admin/repos/mark-private` (expect 422 — body required, NOT 404).
3. Run dry-run from `.audit/2026-04-28/private-row-correction.md` Step 2.
   Confirm `match_count == 1` and the repo identity.
4. Run apply (Step 3 of that runbook).
5. Verify production endpoints (Step 4-5). The hippo row should now 404 on
   `/repos/...` and be absent from `/library/full`, `/search?q=hippo`,
   etc.
6. **Lane 2 must still ship** — the static `reporium.com/data/library.json`
   artifact will keep serving the old JSON until the frontend rebuild.
   Even with the row corrected via this PR, the visible website leak is
   not fully closed until the frontend artifact gate (Lane 2) lands and
   the site is rebuilt.

## Out of scope (deferred to other lanes)

- **Lane 2** — `reporium` static-artifact privacy gate (block emission of
  private rows in `library.json`).
- **Lane 3** — `reporium-ingestion` RCA: why was a private GitHub repo
  ingested with `is_private=false`?
- **Lane 4** — `reporium-audit`: the no-op `isPrivate` contract check
  passed during the leak.
- **Lane 6** — `reporium` frontend repo-card click regression.

The frontend bleed is NOT fully closed by this API PR alone. The static
artifact at `reporium.com/data/library.json` will continue to serve the
old (leaked) JSON until Lane 2 ships and the frontend is rebuilt.

## Files in this PR

```
NEW:
  app/db_filters.py
  app/source_canonical.py
  app/routers/admin_visibility.py
  tests/test_no_private_leak.py
  tests/test_private_and_fork_hotfix.py
  tests/test_admin_mark_private.py
  .audit/2026-04-28/api-private-and-fork-hotfix.md
  .audit/2026-04-28/private-row-correction.md
  .audit/2026-04-28/private-leak-integrated-closeout.md   (this file)

MODIFIED:
  app/main.py                       (wire admin_visibility.router)
  app/routers/compare.py            (migration)
  app/routers/dependencies.py       (UUID-oracle fix + dependents filter)
  app/routers/intelligence.py       (fork canonicalization + related-edges)
  app/routers/library.py            (migration)
  app/routers/library_full.py       (/forks privacy fix + migration)
  app/routers/mentions.py           (UUID-oracle fix)
  app/routers/recommendations.py    (seed_r filter)
  app/routers/repos.py              (migration; closes /repos/{owner}/{name} leak path)
  app/routers/search.py             (migration)
  app/routers/trends.py             (migration)
  app/routers/wiki.py               (migration)
```
