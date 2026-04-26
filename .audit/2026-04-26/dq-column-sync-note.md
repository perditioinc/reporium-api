# DQ primary_category column-vs-junction sync fix

**Date:** 2026-04-26
**Branch:** `claude/feature/KAN-DRAFT-dq-primary-category-sync`
**PR:** https://github.com/perditioinc/reporium-api/pull/444

## Exact root cause

`/metrics/data-quality` (in [`app/routers/platform.py`](../../app/routers/platform.py))
computes `primary_category_coverage` from `repos.primary_category IS NOT NULL`:

```sql
SELECT COUNT(*) FROM repos
  WHERE is_private = false AND primary_category IS NOT NULL
```

But the canonical ingest path in [`app/routers/ingest.py`](../../app/routers/ingest.py)
(`_upsert_repo`) only writes the `repo_categories` junction table — it never
writes the `repos.primary_category` column. The schema has both:

- `repos.primary_category : Text NULL` (model: [`app/models/repo.py:97`](../../app/models/repo.py))
- `repo_categories(repo_id, category_id, category_name, is_primary)` (junction)

The column was originally populated by a one-off migration
(`backfill_primary_category.py`, KAN-41) and has been silently drifting since
then because nothing in the live ingest pipeline keeps it in sync.

This matches the observed live state on `main` (run `24947192489`):
`primary_category_coverage` reads 1641/1856 = 88.4% even though the
`/metrics/data-quality` `missing_primary_category_sample` (added in PR #442)
contains repos that already have a correct `is_primary=true` row in the
junction.

## Exact files changed

- `app/routers/ingest.py` — in `_upsert_repo`, after writing `RepoCategory`
  rows, derive the primary category name from any `is_primary=true` entry in
  the incoming payload and assign it to `repo.primary_category`. Lives inside
  the existing `if item.categories:` block, so the skip-empty guard
  (KAN-123) still applies — empty payloads do not clobber existing data.

- `tests/test_ingest.py` — new
  `test_ingest_keeps_primary_category_column_in_sync_with_junction` regression
  test. Asserts the invariant:
  > After ingest, `repos.primary_category` equals the `category_name` of the
  > `repo_categories` row with `is_primary=true`.

  Covers both the create path (first ingest) and the update path
  (re-ingest swapping which category is primary).

## Expected effect on the DQ gate

- **Forward fix.** Going forward, every repo that flows through
  `POST /ingest/repos` with a non-empty `categories` array will keep the
  column and junction synchronized.
- **Existing drift.** ~215 repos already have `is_primary=true` in the
  junction but `NULL` in the column. Those rows do not self-heal until they
  are re-ingested. There are two paths to flip the gate green sooner:

  1. **Re-ingestion sweep** (zero new code): wait for the next nightly
     enrichment / weekly full ingest to pass each repo through the patched
     `_upsert_repo`. Slow but no extra moving part.

  2. **One-off SQL backfill** against Cloud SQL (small, safe, idempotent):

     ```sql
     UPDATE repos r
        SET primary_category = rc.category_name
       FROM repo_categories rc
      WHERE rc.repo_id = r.id
        AND rc.is_primary = true
        AND r.primary_category IS NULL;
     ```

     Read-only side-effect for rows where the column is already populated;
     only writes when the column is currently `NULL` and a primary exists in
     the junction. Recommended path to flip the gate green inside one
     deployment cycle rather than waiting for full re-ingestion.

## Existing drifted rows: backfill required?

**Yes**, if the goal is to flip `Data Quality Check` green this week. The
forward fix alone is necessary but not sufficient against the current
denominator. The backfill SQL above is the cleanest one-off path. It does
not need a migration script — a `gcloud sql connect` session or admin
endpoint will do.

The remaining gap after the backfill (rows with no `is_primary=true` in the
junction at all) is the genuine empty-fork / enriched-no-primary population
covered by `project_dq_gate_denominator_honesty.md` and
`reporium-ingestion#67`.

## Exact next action after merge

1. **Merge this PR** to `reporium-api` `main`.
2. **Re-run `Data Quality Check`** (`gh workflow run "Data Quality Check"`
   or wait for the scheduled cron). On its own this should not yet flip the
   gate green — see step 3.
3. **Run the backfill SQL** above against Cloud SQL once. Expected outcome:
   `public_with_primary_category` jumps from 1641 toward ~1856, putting
   `primary_category_coverage` ≥ 95% and turning the gate green.
4. **Re-run `Data Quality Check`** to confirm the gate is green.
5. **Decide whether `reporium-ingestion#67` is still required** for any
   residual empty-fork population. The denominator-honesty note still
   applies for that secondary gap, but it is a smaller and qualitatively
   different problem than the column-vs-junction sync bug.

## Anti-duplication boundary

This lane only touches `reporium-api` ingest sync. It does not change the
DQ threshold, does not modify the gate query, does not work on
`reporium-ingestion#67`, does not reopen frontend/security/graph/perf/ADR
work.
