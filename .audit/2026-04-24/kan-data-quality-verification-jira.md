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

## What could still fail the next run

After this patch lands, a remaining failure would reflect real data
quality regressions, not plumbing:

- `primary_category_coverage` < 95%
- `embeddings_coverage` < 95%
- `readme_summary_coverage` < 80%
- `null_is_private_count` > 0
- `no_private_repos_in_api` tripped (`/library/full` heuristic)

None of these are plumbing bugs.

## Process compliance

- Base branch: `main` ✓
- Branch name: `claude/feature/KAN-XXX-data-quality-verification` ✓
  (placeholder `XXX` — swap for real KAN on PR open if available)
- Owned files only: `.github/workflows/data-quality.yml`,
  `scripts/quality_gates.py`, this audit doc ✓
- `scripts/check_data_quality.py` intentionally untouched (no issue there) ✓
- No other lane editing these files (checked `gh pr list` 2026-04-24) ✓
- Not merging, not deploying ✓
