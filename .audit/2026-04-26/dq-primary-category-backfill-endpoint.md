# DQ primary_category column backfill — admin endpoint

**Date:** 2026-04-26 (PDT evening / 2026-04-27 UTC)
**Branch:** `claude/feature/KAN-DRAFT-dq-primary-category-backfill-endpoint`
**Predecessor:** PR #444 (forward-fix in `_upsert_repo`, merged 2026-04-27T01:19:48Z, commit `619325e`)

## Why this lane exists

PR #444 keeps `repos.primary_category` in sync with the `repo_categories`
junction **going forward** — every new ingest now writes both. But ~220
public repos already have `is_primary=true` in the junction with `NULL` in
the column (drift accumulated before the forward-fix). The forward-fix
won't heal them until they are re-ingested, which the nightly enrichment
will only do organically over time.

The DQ Check rerun on `main` after #444 (run `24972119928`) confirmed the
gate is still red:

- `primary_category_coverage`: **1641 / 1861 = 88.18 %** (threshold 95 %)
- Need **127 more rows with non-NULL column** to flip green.

A live sample of the most-recent named-10 missing-primary repos showed a
mixed gap shape:

| Bucket | Count (of 10) | Examples |
| --- | --- | --- |
| Drift (junction has primary, column NULL) | 4 | hackingtool, Pixelle-Video, claude-context, skills___ |
| Genuine empty (no junction entries at all) | 6 | build-your-own-x, DeepEP, ml-intern, osv-scanner, Open-Generative-AI, design.md |

Backfill alone won't flip the gate green if the named-10 ratio holds —
projected post-backfill coverage from sampled ratios is roughly 92–94 %
depending on whether older drift is denser than recent. But it heals the
**drift portion** of the gap idempotently, leaving only the genuine-empty
population for `reporium-ingestion#67` to address.

## Why an admin endpoint, not direct SQL

The audit note in PR #444 said `gcloud sql connect` would do — but the
Cloud SQL instance is **PRIVATE-IP only** (10.14.0.3) on the
`projects/perditio-platform/global/networks/default` VPC. `gcloud sql
connect` and `cloud-sql-proxy` from a developer machine cannot reach it
without VPN or VPC peering, neither of which is configured for this
environment.

The cleanest in-VPC path is an admin endpoint on `reporium-api` itself,
which already runs in the VPC with Cloud SQL Auth Proxy sidecar and a
configured admin-key gate. Pattern matches the existing
`/admin/backfill/categories` endpoint.

## What this PR adds

1. **`POST /admin/backfill/primary_category_column`**
   in `app/routers/admin.py`
   - Auth: `verify_api_key` + `require_admin_key` (same pattern as
     `/admin/backfill/categories`).
   - Rate limit: `5/minute` (one-shot operation, doesn't need higher).
   - `?dry_run=true` returns the same before/after stats without writing.
   - Idempotent UPDATE: writes only where `primary_category IS NULL` and a
     `repo_categories` row with `is_primary=true` exists. Re-running is a
     no-op once the drift is healed.
   - Uses `DISTINCT ON (repo_id) ... ORDER BY repo_id, category_name` so
     the chosen primary is deterministic when a junction has duplicate
     `is_primary=true` rows (a separate latent bug — see "Side-finding"
     below).
   - Invalidates `library:full*` and `repos:list:*` caches after a write.
   - Returns `{dry_run, updated, before: {...}, after: {...}}` with
     `public_total / public_with_primary_category / drift_rows / coverage_pct`
     in each before/after block so the caller can verify the effect on the
     gate denominator without a follow-up workflow run.

2. **`tests/test_backfill_primary_category_column.py`** — three tests:
   - Auth gating (no API key → 403; API key without admin key → 401/403).
   - Happy path: drift row gets healed, genuine-empty row stays NULL,
     already-set row is unchanged. Second call is a no-op (idempotency).
   - Dry-run: count returned, no writes performed.

## Anti-duplication boundary

This lane only adds a one-shot remediation endpoint. It does **not**:

- modify the DQ gate query or threshold
- change the gate denominator semantics (separately tracked in
  `project_dq_gate_denominator_honesty.md`)
- touch ingestion (`reporium-ingestion#67` covers genuine-empty forks)
- reopen frontend / security / graph / perf / rendering ADR work
- fix the duplicate `is_primary=true` junction bug (see below)

## Side-finding: duplicate is_primary entries

During the named-10 sampling, `perditioinc/hackingtool` was found to have
**two** `repo_categories` rows with `is_primary=true`
(`MLOps & Infrastructure` and `Edge & Mobile AI`). The junction lacks a
uniqueness constraint enforcing a single primary per repo, and at least
one ingest path is producing duplicate flags. This is independent of the
column-vs-junction sync bug and is **out of scope** for this PR.

The backfill UPDATE handles this safely (DISTINCT ON picks one
deterministically) but the underlying junction integrity issue should be
filed as a separate follow-up bug.

## Operational plan after merge

1. Wait for Deploy to Cloud Run to publish the new revision.
2. Hit the endpoint with `?dry_run=true` first to verify the projected
   numbers match expectations (drift_to_heal count, projected coverage).
3. Hit it without `dry_run` to execute the backfill.
4. Re-run `Data Quality Check` workflow against `main`.
5. Decide based on residual gap:
   - If gate green → DQ lane closes; `#67` drops priority.
   - If gate still red but residual is mostly genuine-empty (no junction
     entries) → promote `reporium-ingestion#67`.
   - If gate still red from a different shape → open the smallest
     ingestion follow-up for that residual.

## Exact files changed

- `app/routers/admin.py` — new endpoint after `backfill_categories`
- `tests/test_backfill_primary_category_column.py` — new
- `.audit/2026-04-26/dq-primary-category-backfill-endpoint.md` — this file
