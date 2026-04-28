# Private-leak integrated branch recovery — 2026-04-28

**Branch:** `claude/hotfix/private-leak-integrated-2026-04-28`
**Worktree:** `C:\DEV\PERDITIO_PLATFORM\.worktrees\reporium-api-integrated-2026-04-28`
**Base:** `origin/main`

This memo documents the recovery of a damaged worktree that previously held the
consolidated 22-file private-leak hotfix. A prior agent ran
`git checkout main -- <12 files>` to "compare" against main and silently
overwrote those files' working-tree modifications. This recovery restored the
intended PR shape file-by-file from the audit closeout, the simpler reference
worktree, and the surviving new modules.

## Damage scope

The previous agent's `git checkout main -- <files>` overwrote the working-tree
content of 12 files that were tracked but uncommitted. Specifically:

```
app/main.py
app/routers/compare.py
app/routers/dependencies.py
app/routers/intelligence.py
app/routers/library.py
app/routers/library_full.py
app/routers/mentions.py
app/routers/recommendations.py
app/routers/repos.py
app/routers/search.py
app/routers/trends.py
app/routers/wiki.py
```

Untracked / `A`-staged files survived (this is what `git checkout main -- <path>`
leaves alone):

```
app/db_filters.py                                   ✓ surviving
app/source_canonical.py                             ✓ surviving
app/routers/admin_visibility.py                     ✓ surviving
tests/test_no_private_leak.py                       ✓ surviving
tests/test_admin_mark_private.py                    ✓ surviving
tests/test_private_and_fork_hotfix.py               ✓ surviving
.audit/2026-04-28/api-private-and-fork-hotfix.md    ✓ surviving
.audit/2026-04-28/private-leak-integrated-closeout.md ✓ surviving
.audit/2026-04-28/private-row-correction.md         ✓ surviving
```

The 12 routers / `main.py` content was lost (never staged, never committed —
no reflog entry). Recovery had to be re-derived from:

1. The surviving audit closeout doc (Lane-by-lane file-level intent).
2. The simpler reference worktree
   `.worktrees/reporium-api-private-leak-2026-04-28` for the 4 routers it
   modified (`dependencies`, `intelligence`, `mentions`, `recommendations`)
   and the wiring pattern.
3. The surviving new modules' public API
   (`db_filters.public_repo_filter()`, `source_canonical.canonical_owner_name()`,
   `admin_visibility.router`).

## Reference files used

| Source | Used for |
|---|---|
| `.worktrees/reporium-api-integrated-2026-04-28/.audit/2026-04-28/private-leak-integrated-closeout.md` | File-level intent (which routers to migrate, what changes per file) |
| `.worktrees/reporium-api-integrated-2026-04-28/.audit/2026-04-28/api-private-and-fork-hotfix.md` | Specific call-site list for `public_repo_filter()` and the `_build_smart_route_source` helper |
| `.worktrees/reporium-api-integrated-2026-04-28/.audit/2026-04-28/private-row-correction.md` | `admin_visibility` runbook + cache invalidation contract |
| `.worktrees/reporium-api-private-leak-2026-04-28/app/visibility.py` | Wiring template only — superseded by surviving `app/db_filters.py` |
| `.worktrees/reporium-api-private-leak-2026-04-28/app/routers/{dependencies,intelligence,mentions,recommendations}.py` | Diff-against-main pattern for the 4 routers |

`app/visibility.py` from the simpler reference was NOT copied — the surviving
`app/db_filters.py` supersedes it (same purpose, user's name retained per the
audit closeout decision).

## Recovery actions

### `app/main.py`
- Added `admin_visibility` to the `from app.routers import ...` line.
- Added `app.include_router(admin_visibility.router)` after `admin.router`.

### Router migrations to `public_repo_filter()`
The following routers had their inline `Repo.is_private == False` predicates
replaced with `public_repo_filter()` calls (and gained
`from app.db_filters import public_repo_filter`):

- `app/routers/repos.py` — `list_repos`, `cross_category_repos`, `repo_health`,
  `repo_evaluation` (UUID + name branches), `get_repo`, `get_repo_by_owner`.
  All 6 call sites migrated. `Repo.is_private == False` no longer appears.
- `app/routers/library.py` — `get_library` `stmt` plus `public_repos`
  variable used for total/total_forks/language counts.
- `app/routers/search.py` — `search_repos` + `_full_text_fallback`.
- `app/routers/compare.py` — `compare_repos`.
- `app/routers/wiki.py` — `get_skill_wiki` (ai_dev branch + pm_skill branch),
  `get_category_wiki`.
- `app/routers/trends.py` — `get_stats` `public_repos` variable.

### UUID-oracle fixes
- `app/routers/dependencies.py` — `get_repo_dependencies` switched from
  `db.get(Repo, repo_id)` (no predicate) to
  `select(Repo).where(Repo.id == repo_id, public_repo_filter())`. Also added
  `public_repo_filter()` to `get_dependents` so the dependents list cannot
  return private repos using a queried package.
- `app/routers/mentions.py` — same pattern: `get_repo_mentions` switched
  from `db.get(Repo, repo_id)` to filtered SELECT.

### `app/routers/intelligence.py`
- Added `from app.source_canonical import canonical_owner_name` and a
  module-level `_build_smart_route_source` helper that calls
  `canonical_owner_name` to pivot fork rows to upstream owner/name.
- 18 smart-route SQL handlers in `_try_smart_route_inner` re-shape source
  dicts via `_build_smart_route_source` (was hardcoding
  `forked_from: None`).
- `_prepare_query` related-edges hydration SQL — added
  `r1.is_private = false` AND `r2.is_private = false` to the outer JOINs,
  AND inserted `JOIN repos r_inner ON r_inner.id = e2_inner.repo_id AND
  r_inner.is_private = false` inside the lateral subquery so private rows
  cannot be picked as nearest neighbours and only filtered at the outer
  JOIN (which would distort `LIMIT` semantics).

### `app/routers/recommendations.py`
- `_SIMILAR_SQL` gained `seed_r.is_private = false` (defense-in-depth — a
  private seed cannot leak public neighbours back to the caller). The
  pre-existing `r.is_private = false` is retained.

### `/forks` and PR #278 schema coupling
- `app/routers/library_full.py::list_forks` — added `is_private = false` to
  both the SELECT `WHERE` clause and the COUNT query.
- `app/routers/library_full.py::_build_enriched_repo` — added
  `"isPrivate": bool(repo.get("is_private", False))` to the returned dict.
- `app/routers/library_full.py::_fetch_page_repos` — added `is_private` to
  the SQL SELECT projection so the column is available to the response
  shaper.
- `app/routers/library_full.py::list_forks` — re-emits the `is_private`
  column as `isPrivate` in the response shape, matching the camelCase
  contract used by `/library/full`.

## PR #278 coupling — privacy field name

PR `perditioinc/reporium#278` (static-artifact privacy gate) ships a
fail-closed validator that requires every repo in `library.json` to expose
**one** of:

```
isPrivate     | boolean
private       | boolean
visibility    | "public" | "private"
```

This recovery uses `isPrivate` (camelCase, boolean — matches the existing
`isFork` shape on the same payload). Coverage:

- `/library/full` — every repo carries `"isPrivate": false`. (Always false
  because the SQL filter excludes private rows. The point of the field is
  the **presence**, not the value — its presence is what lets the validator
  fail-closed if a future regression drops the WHERE clause.)
- `/forks` — same treatment.

`/library` (the snake_case endpoint, distinct from `/library/full`) is NOT
modified. The audit closeout doesn't require it (PR #278 reads the static
artifact, which is built from `/library/full`), and adding `is_private` to
the snake_case `RepoSummary` schema would be a wider blast radius than this
recovery PR should take. If the validator turns out to read `/library`
output too, that's a one-line follow-up: add `is_private: bool = False` to
`RepoSummary` and surface it from `_repo_to_summary`.

## Verification

Targeted tests (no DB):

```
$ python -m pytest tests/test_no_private_leak.py tests/test_admin_mark_private.py tests/test_private_and_fork_hotfix.py -x -v
13 passed, 33 skipped, 3 warnings in 0.14s
```

The 33 skips are all DB-dependent tests that require a live test Postgres
(`HAS_TEST_DB=1`). Matches the audit closeout's expected counts (13 passed,
33 skipped). Static guards in `test_no_private_leak.py` all pass —
confirming:

- `intelligence.py` related-edges has the three `is_private = false`
  filters (r1, r2, r_inner).
- `recommendations.py` `_SIMILAR_SQL` has both `seed_r.is_private = false`
  and `r.is_private = false`.
- `mentions.py` and `dependencies.py` import `public_repo_filter` from
  `app.db_filters` and use it (≥2 call sites in dependencies.py).
- `db_filters` module exposes `public_repo_filter`, `public_repos_select`,
  `sql_public_filter`, `PUBLIC_REPO_SQL_PREDICATE` — all working.
- `sql_public_filter` validates aliases against SQL injection.
- Admin route `POST /admin/repos/mark-private` is registered on the app.

Ruff:

```
$ python -m ruff check app/db_filters.py app/source_canonical.py app/routers/admin_visibility.py
All checks passed!
```

Pre-existing ruff violations on the migrated routers (15 total: unused
imports in `compare.py`, `library.py`, `repos.py`, `wiki.py`, plus
`f-string without placeholders`, `unused local`) are inherited from `main`
and not caused by this recovery. They are unrelated to the privacy fix and
should be cleaned up in a separate hygiene PR.

Module imports:

```
$ python -c "from app import db_filters, source_canonical; from app.routers import admin_visibility; print('imports OK')"
imports OK
```

Route registration:

```
$ python -c "from app.main import app; ..."
mark-private wired: True
```

Full `git status --short --branch` matches the 22-file changeset from the
closeout doc (3 audit memos + this recovery memo = 4, 3 new app modules,
12 routers including `main.py`, 3 tests).

## Deferred / not in scope

- Full test suite (`python -m pytest -x`) was NOT run — most tests are
  DB-dependent and need a Postgres reachable at `DATABASE_URL` to provide
  meaningful signal. Targeted privacy tests are green; CI will exercise the
  full DB-dependent set on PR open.
- The smart-route fork canonicalization across all 18 handlers is wired
  (every smart-route handler calls `_build_smart_route_source`), but only
  one handler (`_ROUTE_REPO_INFO`) is covered by integration test
  `test_ask_smart_route_source_canonicalizes_fork`. The other 17 are
  defended by the pure-function tests on `canonical_owner_name` plus the
  shared helper. If a regression slips through to one of the 17, it
  surfaces via the LLM context block, not the structured response —
  detection will rely on production observability.
- Pre-existing ruff hygiene (15 violations on inherited code) is left for a
  separate cleanup PR.
- The `/library` (snake_case) endpoint does NOT yet emit `is_private` —
  see PR #278 coupling section above for rationale and follow-up.

## Files in this PR

```
NEW (8):
  app/db_filters.py
  app/source_canonical.py
  app/routers/admin_visibility.py
  tests/test_no_private_leak.py
  tests/test_admin_mark_private.py
  tests/test_private_and_fork_hotfix.py
  .audit/2026-04-28/api-private-and-fork-hotfix.md
  .audit/2026-04-28/private-row-correction.md
  .audit/2026-04-28/private-leak-integrated-closeout.md

MODIFIED (12):
  app/main.py                       — wire admin_visibility.router
  app/routers/compare.py            — public_repo_filter() migration
  app/routers/dependencies.py       — UUID-oracle fix + dependents filter
  app/routers/intelligence.py       — fork canonicalization + related-edges
  app/routers/library.py            — public_repo_filter() migration
  app/routers/library_full.py       — /forks privacy + isPrivate exposure
  app/routers/mentions.py           — UUID-oracle fix
  app/routers/recommendations.py    — seed_r.is_private = false defense
  app/routers/repos.py              — public_repo_filter() migration
  app/routers/search.py             — public_repo_filter() migration
  app/routers/trends.py             — public_repo_filter() migration
  app/routers/wiki.py               — public_repo_filter() migration

NEW (4 audit memos):
  .audit/2026-04-28/private-leak-integrated-recovery.md   ← this file
```

Total: 22 file changes (3 audit memos surviving + 1 new recovery memo + 3
new app modules + 3 new tests + 12 router/main edits).
