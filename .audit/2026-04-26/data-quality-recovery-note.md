# Data Quality Recovery Lane — Final Note

**Date:** 2026-04-25 PDT (UTC date in repo paths)
**Scope:** Recover from the only remaining `Data Quality Check` failure on `main`.
**Workspace:** `C:\DEV\PERDITIO_PLATFORM\reporium-api`

## Live failure shape (revalidated)

- Workflow: `Data Quality Check` (`.github/workflows/data-quality.yml`)
- Latest run on `main`: id `24944014155`, status `completed`, conclusion `failure`
- Failing gate: `primary_category_coverage`
  - Numerator: 1641 public repos with `primary_category`
  - Denominator: 1856 public repos
  - Coverage: 88.4%
  - Threshold: 95%
  - Shortfall: **123 repos** (need to reach 1764 / 1856 to pass)
- All other gates pass: `embeddings_coverage`, `null_is_private`, `readme_summary_coverage`, `no_private_repos_in_api`
- Auth/403 from the prior lane is fully gone — `X-Admin-Key` injection works.

## Root cause

The data deficit itself (215 public repos with `primary_category IS NULL`) lives outside this repo. It is fed by the `reporium-ingestion` enrichment Cloud Run Job. Memory entry `project_ask_sprint1_apr22.md` already assigns enrichment ownership to that lane.

This repo's contribution to the failure is **insufficient diagnostics** — the gate today reports only `1641/1856 public repos have primary_category` and exits 1, giving the operator no way to know which repos to enrich, by how many, or where to act.

## Repo-local fix (this lane)

Branch: `claude/feature/KAN-DRAFT-data-quality-recovery`
PR: https://github.com/perditioinc/reporium-api/pull/442

Two small changes:

1. **`app/routers/platform.py`** — `/metrics/data-quality` now returns an additional field, `missing_primary_category_sample`: a list (capped at 10) of the most-recently-ingested public repos that are still missing `primary_category`, each shaped `{name: "owner/name", ingested_at: "ISO-8601"}`. Cheap extra `SELECT ... ORDER BY ingested_at DESC LIMIT 10`. Private repos are still excluded.
2. **`scripts/quality_gates.py`** — when `primary_category_coverage` fails, the script now prints:
   - the absolute gap to the threshold ("need N more enriched repos to reach 95%")
   - the sample list of missing repos with their `ingested_at` timestamps
   - a one-line pointer to the fix location (`reporium-ingestion` enrichment Cloud Run Job)

   Implementation detail: each gate result dict gained an `extra: list[str]` field; `main()` indents and prints those lines below `[FAIL]`. Pass-path output is unchanged.
3. **`tests/test_platform_metrics.py`** — extended the existing `test_metrics_data_quality_reports_public_only_coverage` to assert the new sample field exists, includes the inserted `dq-pub-bare` (NULL primary_category, public), is capped at 10, and never leaks the private `dq-priv` row.

## Proof of behavior

A local end-to-end smoke test pointed `quality_gates.py` at a mock HTTP server returning a payload that mirrors the live failure (1641/1856 + 3 sample repos). Output:

```
[FAIL] primary_category_coverage: 1641/1856 public repos have primary_category
        need 123 more enriched repos to reach 95% threshold
        most-recent public repos missing primary_category (top 3):
          - foo/recent-1  (ingested_at=2026-04-25T10:00:00+00:00)
          - bar/recent-2  (ingested_at=2026-04-25T09:00:00+00:00)
          - baz/recent-3  (ingested_at=2026-04-25T08:00:00+00:00)
        fix in: reporium-ingestion enrichment Cloud Run Job — re-run the nightly enrichment workflow against the missing repo IDs.
```

`_shortfall(1641, 1856, 95.0) == 123` confirmed against the live numbers.

The integration test against the real `/metrics/data-quality` endpoint requires Postgres on `localhost:5432`, which is not available on the workspace machine. CI on the PR will run it.

## What this lane does NOT change

- Threshold remains 95% (lowering it without fixing the underlying data would mask the problem).
- The data fix itself stays with the **ingestion lane** — running enrichment against the 215 missing repos.
- No `data-quality.yml` workflow change.
- No deploy was triggered manually.

## Exact next action (if the issue remains outside this repo after this PR merges)

1. Wait for the PR to merge to `main` and Cloud Run auto-deploy to land (typically <15 min).
2. Trigger one `Data Quality Check` workflow run via `workflow_dispatch`.
3. Read the failing gate output — it will now include up to 10 specific `owner/name` repos to chase.
4. In `reporium-ingestion`, run the enrichment Cloud Run Job restricted to those repo IDs (or the most recent N ingested-but-unenriched batch). Per memory `project_ask_sprint1_apr22.md`, the scheduler-triggered nightly job ran cleanly at 07:16 UTC on 2026-04-22 with 181 repos / 0 errors — the job itself is healthy.
5. Re-run the data-quality workflow; expect the gate to clear once 123+ of the missing repos receive a category.

## PR

PR link: https://github.com/perditioinc/reporium-api/pull/442
