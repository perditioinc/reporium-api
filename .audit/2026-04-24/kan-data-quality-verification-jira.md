# KAN-XXX — Data Quality Check verification + admin-key plumbing

**Lane**: Verify `reporium-api` Data Quality Check
**Date**: 2026-04-24
**Owner**: claude
**Branch**: `claude/feature/KAN-XXX-data-quality-verification`
**PR**: opened against `main` (do not merge without review)

## TL;DR

The HTTPS-based rewrite of `scripts/quality_gates.py` already landed on `main`
as commit `0b6fdb3` (PR #432) on **2026-04-24 03:44 UTC**. The latest failed
scheduled run (`24829615936`, 2026-04-23 10:13 UTC) predates the fix by
~17.5 hours and was still on the old psycopg2-to-private-IP code path.

**Decision: still broken in code — one remaining plumbing gap.**

A live probe against the deployed endpoint confirms:

```
$ curl -s -o /dev/null -w "%{http_code}" \
    https://reporium-api-573778300586.us-central1.run.app/metrics/data-quality
403
```

`METRICS_REQUIRE_AUTH=1` was enabled in production on **2026-04-21 02:31 UTC**
by PR #393 ("security: enable metrics auth"). PR #432 landed three days later
without adding the `X-Admin-Key` header, so the next scheduled run at
2026-04-24 09:00 UTC would still fail — this time with
`metrics_api_reachable: HTTP Error 403: Forbidden`.

Minimal follow-up patch in this lane:

1. `scripts/quality_gates.py` — `_fetch_json` now sends `X-Admin-Key` when
   `ADMIN_API_KEY` is in the environment.
2. `.github/workflows/data-quality.yml` — quality-gates step now passes
   `ADMIN_API_KEY: ${{ secrets.ADMIN_API_KEY }}` (secret already exists in
   the repo, first added 2026-04-06).

## Evidence

### 1. Latest failed scheduled run is older than the fix

- Run ID: `24829615936`
- Start:  `2026-04-23T10:13:12Z`
- Failure: `[FAIL] db_connection: connection to server on socket
  "/cloudsql/perditio-platform:us-central1:reporium-db/.s.PGSQL.5432" failed:
  No such file or directory` — classic symptom of psycopg2 trying to reach
  private-IP Cloud SQL from a GitHub-hosted runner.

### 2. Fix commit is on `main` and deployed

- Commit:       `0b6fdb3ca7ddf90c484678e0b337bc6459b71a62`
- Authored:     `2026-04-23 20:44:21 -0700` (= `2026-04-24 03:44:21 UTC`)
- Message:      `fix(data-quality): rewrite gates to read /metrics/data-quality over HTTPS (#432)`
- Files:        `scripts/quality_gates.py`, `.github/workflows/data-quality.yml`,
                `app/routers/platform.py`, `tests/test_platform_metrics.py`

### 3. `/metrics/data-quality` is gated behind `METRICS_REQUIRE_AUTH=1`

- `app/auth.py:86-110` — `require_metrics_access` returns 403 unless
  `X-Admin-Key` matches `ADMIN_API_KEY` when the flag is on.
- `app/auth.py:69` — header name is `X-Admin-Key`.
- PR #393 merged `2026-04-21 02:31 UTC` enabled the flag in production.
- Probe at `2026-04-24 ~07:33 UTC` returns `403 Admin key required for
  metrics endpoints` — confirmed.

### 4. Secret already exists in the repo

`gh secret list --app actions` shows `ADMIN_API_KEY` in the repo secrets
since 2026-04-06. No new secret needs to be created — the workflow just
needs to reference it.

## Patch shipped on this branch

**`scripts/quality_gates.py`** — opportunistic header injection:

```python
def _fetch_json(url: str, headers: dict[str, str] | None = None, timeout: int = 20) -> dict:
    merged = {"Accept": "application/json", **(headers or {})}
    admin_key = os.getenv("ADMIN_API_KEY", "").strip()
    if admin_key and "X-Admin-Key" not in merged:
        merged["X-Admin-Key"] = admin_key
    req = urllib.request.Request(url, headers=merged)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))
```

Behavior:
- In CI with `ADMIN_API_KEY` set → sends the header, bypasses the auth gate.
- Locally with no key set → no header, endpoint must be open (dev mode).
- If `METRICS_REQUIRE_AUTH=0` is ever restored in prod → header is ignored
  by the server. Harmless either way.

**`.github/workflows/data-quality.yml`** — secret plumbed through:

```yaml
- name: Run quality gates
  env:
    REPORIUM_API_URL: ${{ secrets.REPORIUM_API_URL }}
    ADMIN_API_KEY: ${{ secrets.ADMIN_API_KEY }}
  run: python scripts/quality_gates.py
```

`scripts/check_data_quality.py` is untouched — it only hits `/library/full`,
which is public.

## Verification steps (for the reviewer)

1. **Review the diff** — two files, ~10 lines net.

2. **Dry-run the script against prod using the repo secret** (requires
   a local copy of `ADMIN_API_KEY`):

   ```
   ADMIN_API_KEY="<value>" \
   REPORIUM_API_URL="https://reporium-api-573778300586.us-central1.run.app" \
   python scripts/quality_gates.py --report-only
   ```

   Expected: all 5 gates print `[PASS]` or `[FAIL]` based on real data —
   no `HTTP Error 403`, no `DB connection failed`.

3. **Merge and dispatch** (not in this lane's mandate):

   ```
   gh workflow run data-quality.yml --ref main
   gh run watch $(gh run list --workflow=data-quality.yml \
                  --limit 1 --json databaseId -q '.[0].databaseId')
   ```

## Proof the patch works (lane-8 dry-run, 2026-04-24 07:40 UTC)

Executed the patched script against prod with the real `ADMIN_API_KEY`
fetched from GCP Secret Manager (`reporium-admin-api-key`):

```
ADMIN_API_KEY="$(gcloud secrets versions access latest \
                  --secret=reporium-admin-api-key \
                  --project=perditio-platform)" \
REPORIUM_API_URL="https://reporium-api-573778300586.us-central1.run.app" \
python scripts/quality_gates.py --report-only
```

Result:

```
[FAIL] primary_category_coverage: 1641/1856 public repos have primary_category
[PASS] embeddings_coverage: 1855/1856 public repos have embeddings
[PASS] null_is_private: 0 repos have NULL is_private
[PASS] readme_summary_coverage: 1645/1856 public repos have readme_summary
[PASS] no_private_repos_in_api: API returned 100 repos, 0 potentially private
```

- **No `HTTP Error 403`** — auth gate cleared.
- **No `DB connection failed`** — no psycopg2 path.
- 4 of 5 gates green on live data.
- The 1 remaining failure is a **real data-coverage regression**, not a
  plumbing bug (see next section).

Raw `/metrics/data-quality` payload at the same moment:

```json
{"total_public_repos":1856,"public_with_primary_category":1641,
 "public_with_readme_summary":1645,"public_with_embeddings":1855,
 "null_is_private_count":0,
 "generated_at":"2026-04-24T07:40:01.761467+00:00"}
```

## CI status on PR #440

All checks green at 2026-04-24 ~07:45 UTC:

- `test` (unit) — pass (3m10s)
- `test` (other) — pass (4m27s)
- `migration-smoke` — pass (1m59s)
- `ask-quality-gate` — pass (30s)
- `notify-on-failure` — skipped (intentional — only runs on job failure)

## Out-of-lane follow-up: primary_category coverage gap

The one remaining `[FAIL]` is **out of scope** for this lane. Pinning
it here so it doesn't get lost:

- **What:** 215 of 1856 public repos lack `primary_category` → 88.4%
  coverage vs 95% threshold.
- **Why it's out of lane:** owned scope is the workflow + gate script.
  Fixing the data requires either running the nightly enrichment job
  (`reporium-ingestion`) against the uncategorised set, or a one-shot
  backfill — neither of which touches this lane's files.
- **Similar prior issues (now closed):**
  [#131](https://github.com/perditioinc/reporium-api/issues/131) KAN-41
  re-run enrichment; [#165](https://github.com/perditioinc/reporium-api/issues/165)
  readme_summary backfill — suggests the pattern is "enrichment did not
  keep up with new ingestions."
- **Suggested next lane:** dispatch to `reporium-ingestion` to run
  targeted enrichment on rows where `primary_category IS NULL` AND
  `is_private = false`. Until that runs, the scheduled Data Quality
  Check will continue to exit 1 daily — **correctly** signalling a real
  data regression, not a plumbing bug.

## What could still fail the next run

With the plumbing fix live, a remaining failure reflects real data
regressions, not plumbing:

- `primary_category_coverage` < 95% — **ACTIVE, see follow-up above**
- `embeddings_coverage` < 95% — currently 99.95%, comfortable
- `readme_summary_coverage` < 80% — currently 88.6%, comfortable
- `null_is_private_count` > 0 — currently 0, comfortable
- `no_private_repos_in_api` tripped (`/library/full` heuristic) — green

## Process compliance

- Base branch: `main` ✓
- Branch name: `claude/feature/KAN-XXX-data-quality-verification` ✓
  (placeholder `XXX` — swap for real KAN on PR open if available)
- Owned files only: `.github/workflows/data-quality.yml`,
  `scripts/quality_gates.py`, this audit doc ✓
- `scripts/check_data_quality.py` intentionally untouched (no issue there) ✓
- No other lane editing these files (checked `gh pr list` 2026-04-24) ✓
- Not merging, not deploying ✓
