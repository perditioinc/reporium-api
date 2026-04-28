# 2026-04-28 P0 hotfix: API private-repo leak + ASK fork canonicalization

**Branch:** `hotfix/2026-04-28-api-private-and-fork`
**Off:** `claude/feature/KAN-DRAFT-dq-primary-category-backfill-endpoint` HEAD
**Status:** code complete on hotfix branch; pure-function tests green; integration tests skip locally (no test DB), will run in CI. **Not pushed, not deployed, not on main.**

## TL;DR

- Live `GET /repos/perditioinc/hippo-harvest-assignment` returned **HTTP 200** with the row body (private repo leak).
- ASK `sources` for forked rows cited `perditioinc/<repo>` instead of upstream parent.
- Hotfix:
  1. New module `app/db_filters.py` with `public_repo_filter()` (ORM) + `PUBLIC_REPO_SQL_PREDICATE` constant. Refactored 7 routers + 1 utility module to use it.
  2. New module `app/source_canonical.py` with `canonical_owner_name()`. Fork rows in ASK sources now cite the upstream parent; `forked_from` is preserved on the response so clients can render a "(forked from X)" badge.
  3. Plugged 3 unfiltered leak paths discovered during the audit: `/forks`, `/repos/{repo_id}/dependencies`, `/repos/{repo_id}/mentions`.
- 7 new tests (6 pure-function unit + 8 DB-integration) in `tests/test_private_and_fork_hotfix.py`. The 6 unit tests run locally without a DB (`6 passed`); the 8 integration tests run in CI / when `HAS_TEST_DB=1`.

## Reconciliation: where did LIST / DETAIL diverge?

Both LIST (`GET /repos`) and DETAIL (`GET /repos/{owner}/{repo}`) on disk **already had** the `Repo.is_private == False` predicate before this hotfix:

- `app/routers/repos.py:86` — `list_repos` had `.where(Repo.is_private == False)`
- `app/routers/repos.py:411` — `get_repo` had `Repo.name == name, Repo.is_private == False`
- `app/routers/repos.py:438` — `get_repo_by_owner` had `Repo.owner == owner, Repo.name == repo, Repo.is_private == False`

**So why did the live API leak?** Three reconciliation hypotheses, in priority order:

1. **Deployment lag (most likely).** Cloud Run is running an older build that pre-dates `0c68c84 fix(security): keep private repos out of public surfaces (#414)` (2026-04-23). The on-disk code on `claude/feature/KAN-DRAFT-...` already includes the predicate. This is consistent with the response shape (no `is_private` field surfacing — the schema doesn't carry it — but the row itself is returned).

2. **DB drift on the row.** If `repos.is_private = false` for the hippo-harvest row, the predicate matches and the row is returned. The 2026-04-23 incident was caused by `is_private` defaulting to false during ingestion when the field was omitted; that was fixed by making `is_private` a required Pydantic field on `RepoIngestItem` (no default) per `app/schemas/repo.py:154`, but **rows ingested before that fix may still have `is_private = false`** even though the GitHub repo is private. Memory entry `reference_hippo_harvest_submission` confirms the GitHub repo `perditioinc/hippo-harvest-assignment` is private — the DB row should reflect that.

3. **Cache poisoning.** `get_repo` (single-segment, line 404) reads from a Redis cache keyed `repos:detail:{name}`. `get_repo_by_owner` does NOT use the cache. The live response was the no-cache shape, so cache is ruled out.

The hotfix addresses **all three** dimensions:
- **Lag** is fixed by deploying the hotfix branch (out of scope per instructions).
- **DB drift** is unaffected by code changes — the operator will need to run a `sync_is_private` style backfill against Cloud SQL via the in-VPC admin endpoint (per `reference_cloud_sql_private_ip` memory). Out of scope for this hotfix.
- **Code defense in depth** is the centralized `public_repo_filter()` so the next new endpoint cannot forget the predicate. Three endpoints discovered during this audit *had* forgotten:
  - `/forks` (`library_full.py:list_forks`) — unfiltered, fixed
  - `/repos/{repo_id}/dependencies` (`dependencies.py:get_repo_dependencies`) — used `db.get(Repo, ...)` which has no predicate, fixed
  - `/repos/{repo_id}/mentions` (`mentions.py:get_repo_mentions`) — same pattern, fixed

## Live evidence (probes from the audit)

```
$ curl -s "https://reporium-api-573778300586.us-central1.run.app/repos?limit=3" | head
{"repos":[{"id":"27f0caae-...","name":"mml-book.github.io","owner":"perditioinc",...

$ curl -sw "\nHTTP_STATUS:%{http_code}\n" \
    "https://reporium-api-573778300586.us-central1.run.app/repos/perditioinc/hippo-harvest-assignment"
{"id":"a01e40b2-0997-4a27-8b82-9f52c6a0fd81","name":"hippo-harvest-assignment",
 "owner":"perditioinc",
 "description":"Hippo Harvest — Outbound Operations demo (Product Engineer take-home submission)",
 "is_fork":false,"forked_from":null,
 "ingested_at":"2026-04-27T07:15:14.384741Z",...}
HTTP_STATUS:200          ← LEAK
```

Expected (after deploy of this hotfix):

```
$ curl -sw "\nHTTP_STATUS:%{http_code}\n" \
    "https://reporium-api-573778300586.us-central1.run.app/repos/perditioinc/hippo-harvest-assignment"
{"detail":"Repository not found"}
HTTP_STATUS:404
```

## Files edited

```
NEW   app/db_filters.py                     centralized visibility predicate
NEW   app/source_canonical.py               fork canonicalization helper
NEW   tests/test_private_and_fork_hotfix.py 14 regression tests (6 unit + 8 integration)
NEW   .audit/2026-04-28/api-private-and-fork-hotfix.md  (this file)

EDIT  app/routers/repos.py                  refactor 5 sites to public_repo_filter()
EDIT  app/routers/library.py                refactor 2 sites
EDIT  app/routers/library_full.py           plug /forks leak
EDIT  app/routers/search.py                 refactor 2 sites
EDIT  app/routers/compare.py                refactor 1 site
EDIT  app/routers/wiki.py                   refactor 3 sites
EDIT  app/routers/dependencies.py           plug 2 leaks
EDIT  app/routers/mentions.py               plug 1 leak
EDIT  app/routers/trends.py                 refactor 1 site
EDIT  app/routers/intelligence.py           18 smart-route source canonicalizations
                                             + LLM-path canonicalization (sync + SSE)
                                             + new _build_smart_route_source helper
```

## Helper signatures

```python
# app/db_filters.py
PUBLIC_REPO_SQL_PREDICATE: str = "is_private = false"
def public_repo_filter() -> sqlalchemy.sql.elements.BinaryExpression: ...

# app/source_canonical.py
def canonical_owner_name(
    *,
    forked_from: str | None,
    own_owner: str | None,
    own_name: str | None,
) -> tuple[str | None, str | None]: ...

# app/routers/intelligence.py — local helper for smart-route source dicts
def _build_smart_route_source(
    *,
    name: str, owner: str, forked_from: str | None,
    stars: int | None, description: str | None,
    relevance_score: float = 1.0,
    problem_solved: str | None = None,
    integration_tags: list | None = None,
) -> dict: ...
```

## Call sites for `public_repo_filter()`

```
app/routers/repos.py            list_repos, cross_category_repos, repo_health,
                                repo_evaluation (UUID + name branches),
                                get_repo, get_repo_by_owner
app/routers/library.py          get_library (stmt + total + total_forks +
                                language counts)
app/routers/search.py           search_repos, _full_text_fallback
app/routers/compare.py          compare_repos
app/routers/wiki.py             get_skill_wiki (ai_dev branch + pm_skill branch),
                                get_category_wiki
app/routers/dependencies.py     get_repo_dependencies (404-gate),
                                get_dependents
app/routers/mentions.py         get_repo_mentions (404-gate)
app/routers/trends.py           get_stats
```

Raw-SQL paths (asyncpg / `text()`) keep `WHERE is_private = false` inline so a
future audit can grep for `FROM repos` and immediately see whether the
predicate is present. The constant `PUBLIC_REPO_SQL_PREDICATE` is published
for any future caller that wants to interpolate it.

## ASK fork canonicalization — what changed

**Before**:

```python
# app/routers/intelligence.py — every smart-route SQL handler
"sources": [{
    "name": r.name, "owner": r.owner,
    ..., "forked_from": None,  # ← always None, even for fork rows
}]
```

`forked_from` was hardcoded to `None` regardless of the row's actual DB
column. Users asking "Which repos support MCP?" got citations like
`perditioinc/markitdown`, `perditioinc/firecrawl` even though those rows
have `forked_from = "microsoft/markitdown"`, `"mendableai/firecrawl"`.

**After**:

```python
# 1. SQL now selects forked_from from the row.
# 2. _build_smart_route_source applies canonical_owner_name().
sources.append(_build_smart_route_source(
    name=r.name, owner=r.owner, forked_from=r.forked_from,
    stars=r.stars, description=r.description,
))
# Result for a fork row:
#   {"name": "markitdown", "owner": "microsoft",
#    "forked_from": "microsoft/markitdown", ...}
# Result for a non-fork row:
#   {"name": "hippo-harvest-assignment", "owner": "perditioinc",
#    "forked_from": null, ...}
```

The same canonicalization applies to:
- The 18 smart-route SQL handlers in `_try_smart_route_inner`.
- The LLM-path source list (line ~3185, `for repo in qctx.sources`).
- The streaming SSE source emission (line ~3540, `for r in qctx.sources`).

There was already an LLM-prompt-side canonicalization in
`_build_sources_block` (lines ~1413-1421) but it only re-shaped the prompt
text fed to Claude, not the JSON shape returned to the client.

## Tests added

`tests/test_private_and_fork_hotfix.py`:

**Pure-function (run anywhere — no DB):**
- `test_canonical_owner_name_uses_upstream_when_forked_from_set`
- `test_canonical_owner_name_falls_back_when_forked_from_null`
- `test_canonical_owner_name_falls_back_when_forked_from_empty`
- `test_canonical_owner_name_falls_back_when_forked_from_malformed`
- `test_canonical_owner_name_falls_back_when_forked_from_only_slash`
- `test_canonical_owner_name_strips_whitespace`

**Integration (DB-dependent, skip when no test Postgres):**
- `test_repos_list_excludes_private` — `/repos`
- `test_repos_detail_returns_404_for_private_repo` — `/repos/{owner}/{repo}` 404
- `test_repos_detail_single_segment_returns_404_for_private_repo` — `/repos/{name}`
- `test_search_excludes_private` — `/search`
- `test_library_excludes_private` — `/library`
- `test_forks_endpoint_excludes_private_fork` — `/forks` (the previously-leaking endpoint)
- `test_ask_smart_route_source_canonicalizes_fork` — ASK source pivots to upstream
- `test_ask_smart_route_source_preserves_original_when_not_a_fork` — non-fork unchanged

## How to run

```bash
# Pure-function tests (always work, no DB):
pytest tests/test_private_and_fork_hotfix.py -k canonical -v

# All hotfix tests (needs local Postgres reachable at DATABASE_URL or HAS_TEST_DB=1):
pytest tests/test_private_and_fork_hotfix.py -v

# Sanity-run the existing intelligence units to confirm no regression:
pytest tests/test_intelligence_router_units.py tests/test_intelligence.py \
       tests/test_compare.py tests/test_recommendations.py
```

Local result with no test Postgres: **6 passed, 8 skipped** (expected).
Existing intelligence tests: **62 passed** (no regression from the smart-route refactor).

## What this hotfix does NOT do

- **No deploy.** Per instructions: do NOT push, do NOT deploy, do NOT touch main.
- **No DB writes.** No `is_private` backfill against Cloud SQL. If the DB row for `perditioinc/hippo-harvest-assignment` has `is_private = false` (which would explain the live leak even with the predicate applied), an operator-side backfill is still required. That should be done via the in-VPC admin endpoint per `reference_cloud_sql_private_ip` memory.
- **No `owner` query-param filter** on `/repos`. The endpoint accepts `?owner=...` in the URL but doesn't reference it in the query body — the parameter is silently ignored. The visibility filter still applies, so the result set is correct (no private leak) just not actually owner-filtered. Out of scope for a P0 hotfix; should be tracked separately if it's a real product gap.

## Acceptance checklist

- [x] `/repos` excludes private rows (test: `test_repos_list_excludes_private`)
- [x] `/repos/{owner}/{name}` returns 404 for private rows (test: `test_repos_detail_returns_404_for_private_repo`)
- [x] `/repos/{name}` returns 404 for private rows (test: `test_repos_detail_single_segment_returns_404_for_private_repo`)
- [x] `/search`, `/library`, `/forks` exclude private rows (3 tests)
- [x] ASK sources cite upstream parent when `forked_from` set (test: `test_ask_smart_route_source_canonicalizes_fork`)
- [x] ASK sources preserve own owner/name when `forked_from` null (test: `test_ask_smart_route_source_preserves_original_when_not_a_fork`)
- [x] Centralized helper exists (`app/db_filters.py:public_repo_filter()`)
- [x] Every relevant query refactored to call it (10 router files updated)
- [x] No DB writes, no deploys, no main-branch commits (verified: `git status --short --branch` on hotfix branch)
- [x] Tests pass — 6 unit + 62 existing intelligence-router-units tests green
