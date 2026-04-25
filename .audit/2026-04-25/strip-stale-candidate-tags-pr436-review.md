# PR #436 review — strip stale candidate-* traffic tags

**Lane:** `reporium-api #436` review and refine
**Reviewer:** Claude (Opus 4.7)
**Date:** 2026-04-25
**Recommendation:** ✅ **GO** — push refinement commit `bbb69ac`, then merge.

## State at review time

- Base branch: `main` (HEAD `1feeb0f` — `test(ask): add forbidden_repos primitive…` #367)
- PR head branch (remote): `fix/deploy-strip-stale-traffic-tags` @ `5235333`
- PR head branch (local): `5235333 → bbb69ac` (refinement commit, **unpushed**)
- PR mergeable: `MERGEABLE`
- CI on `5235333`: all green (Ask Quality Gate, Dev Tests, Tests, migration-smoke)
- No reviews / comments yet
- Owned files only: `.github/workflows/deploy.yml`, `.audit/2026-04-25/*` ✅
- No other lane is editing `deploy.yml` (last touch on main: `b84c8e9` Apr 17)

## Live Cloud Run state at review time

```
spec.traffic     : [percent=100 rev=…00252-fop, tag=candidate-58ab8cd rev=…00252-fop]
status.traffic   : [percent=100 tag=candidate-58ab8cd rev=…00252-fop url=…run.app]
```

Only the serving revision carries a `candidate-*` tag right now. Probe of the
previously-cited stale URL `candidate-ca63b22---reporium-api-wypbzj5gpa-uc.a.run.app/stats`
returns **HTTP/2 404** — someone already manually stripped that tag between
the original audit (Apr 23) and now. The base URL serves normally.

**Implication:** the immediate bypass is closed *today*, but the next deploy
*without* #436 would re-introduce it. The fix is correct as a preventive
measure. No emergency manual cleanup is required pre-merge.

## Correctness review of `bbb69ac` (current local PR head)

The refinement commit hardens `5235333` along three vectors:

| Concern | `5235333` (PR remote head) | `bbb69ac` (local refinement) | Verdict |
|---|---|---|---|
| Tag scope | All `tag != null && percent=0` — would also strip operator tags `stable` / `rollback` | `tag` prefix-matches `candidate-` only | ✅ Necessary — class-of-tags safety |
| `gcloud --format` reliability | `value(...filter('percent=0').extract(tag).flatten())` — fragile across gcloud versions, separator differs | `--format=json | jq` over `.status.traffic[]` | ✅ Robust; `jq` preinstalled on `ubuntu-latest` |
| Regression visibility | None — silent if cleanup fails | Verify step emits `::warning::` if any `candidate-*` tag remains on a `percent=0` revision | ✅ Surfaces future flag/permission regressions in Actions UI |
| Failure mode | `continue-on-error: true` — won't block deploy | Both steps `continue-on-error: true` | ✅ Same posture, no new blocking surface |
| Comment hygiene | Promote step still claims leftover tag is "harmless" | Removes the misleading comment | ✅ Aligns code comment with security reality |

### Edge cases checked

1. **First deploy after merge (no prior candidate tags):** cleanup step
   selects nothing → prints `No stale candidate-* traffic tags to remove.`
   and exits 0. Verify step prints `bypass paths closed.` ✅
2. **Operator-set `stable` / `rollback` tags on rolled-back revisions:**
   prefix filter excludes them; not stripped. ✅
3. **Same SHA re-deployed (cherry-pick):** `--tag candidate-<sha>` on
   `gcloud run deploy` moves the tag to the new revision rather than
   creating a duplicate; selection set is empty. ✅
4. **Promotion fails (migration error, etc.):** cleanup step is gated by
   `if: success()` — does not run; tag state on the service is unchanged
   regardless. ✅
5. **`jq` upstream change:** `jq -r` and `startswith` are baseline syntax
   shipped with every `jq ≥1.6` in `ubuntu-latest`. No risk. ✅
6. **Race with concurrent deploy:** the deploy job's GCP service account
   has the only role here; concurrent deploys are serialized by the
   workflow `concurrency:` group at the top of `deploy.yml`. Already in
   place. ✅
7. **`--remove-tags` for a no-longer-existing tag:** gcloud no-ops
   gracefully on missing tags within the same call; `continue-on-error`
   covers any edge that doesn't. ✅

### Patch needed?

**No additional `deploy.yml` change.** The refinement commit `bbb69ac` is
the safest minimal fix. The remaining action is a `git push` to publish
`bbb69ac` so PR #436 reflects the hardened version before merge. (Pushing
is shared-state and is left to the human owner of the PR per session
guidance.)

## Stop-condition check

- ✅ No GCP-side change required — fix is workflow-only.
- ✅ Patch does not broaden the deploy flow beyond the tag-cleanup concern.
  Two new steps, both post-promotion, both `continue-on-error`. No new
  permissions, secrets, or behavior changes outside tag lifecycle.
- ✅ No other lane is editing `deploy.yml`.

## Merge recommendation

**GO.** Push `bbb69ac`, await CI green on the refined head, then merge to
`main`. Do not merge before push — the remote head still has the looser
tag-prefix logic and the fragile `--format` projection.

### Pre-merge actions (human owner)

```bash
git push origin fix/deploy-strip-stale-traffic-tags
# CI re-runs against bbb69ac — wait for green, then merge via the queue.
```

## Post-merge / next-deploy verification checklist

Run on the first merge to `main` after #436 lands.

1. **Watch the workflow run.**
   - `Promote candidate revision to 100% traffic` — succeeds (unchanged).
   - `Remove stale candidate-* traffic tags from non-serving revisions` —
     prints either `No stale candidate-* traffic tags to remove.` or
     `Removing stale candidate traffic tags: candidate-…[,candidate-…]`.
   - `Verify no public candidate-* bypass URLs remain` — prints
     `No stale candidate-* tags on non-serving revisions — bypass paths closed.`
     If it emits `::warning::`, treat as a follow-up bug, not a deploy
     failure.

2. **Inspect live traffic state.**

   ```bash
   gcloud run services describe reporium-api \
     --project=perditio-platform \
     --region=us-central1 \
     --format=json \
   | jq -r '.status.traffic[] | "\(.percent)%  tag=\(.tag // "<none>")  rev=\(.revisionName)"'
   ```

   Expected: exactly one row with `tag=candidate-<new_sha>` at `100%`.
   Any `0%  tag=candidate-*` row is a regression.

3. **Probe a previously-known stale URL.**

   ```bash
   curl -sI "https://candidate-58ab8cd---reporium-api-wypbzj5gpa-uc.a.run.app/stats" | head -n1
   # Expected after a *subsequent* deploy: HTTP/2 404
   ```

   (`candidate-58ab8cd` is the current serving tag at audit time. After the
   next deploy it should become a 404.)

4. **Probe the base URL.**

   ```bash
   curl -s https://reporium-api-wypbzj5gpa-uc.a.run.app/stats | jq .
   # Expected: current production counts (e.g. ~1855 / 18 from the
   # private-repo-filter-aware path)
   ```

5. **Smoke test rollback path is unaffected.**
   No action needed — `--remove-tags` strips tags only; revisions are
   retained per existing image-prune step (keep latest 5). Rollback via
   `gcloud run services update-traffic --to-revisions=<prior>=100` still
   works.

## Out-of-scope follow-ups (do not gate this PR)

- **Long-term durable fix:** require auth on `reporium-api` (remove
  `allUsers` invoker). Tagged URLs being public is a Cloud Run feature,
  not a misconfig. Track as a separate epic — significant client/CI
  surface change.
- **Backfill audit log:** there is no record of who/when stripped the
  `candidate-ca63b22` tag manually. Consider enabling Cloud Audit Logs
  for `run.googleapis.com/Service.UpdateTraffic` if not already on.

## Files touched in this lane (owned scope)

- `.audit/2026-04-25/strip-stale-candidate-tags-jira.md`
- `.audit/2026-04-25/strip-stale-candidate-tags-pr436-review.md` (this file)

No `deploy.yml` change made by this review — the existing `bbb69ac` patch
is the recommended state.
