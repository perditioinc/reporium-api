# JIRA draft — strip stale candidate-* Cloud Run traffic tags

> Created in lieu of JIRA (no Atlassian MCP wired into this session).
> Promote to a real ticket when the Workato Atlassian recipe is back online.

## Summary

After every promoted deploy, Cloud Run leaves `candidate-<short_sha>` traffic
tags on prior `reporium-api` revisions. Because the service is invoker
`allUsers`, every leftover tagged URL
(`https://candidate-<sha>---reporium-api-wypbzj5gpa-uc.a.run.app`) is a public
bypass that serves pre-promotion code against the live database.

## Type / priority

- Type: **Bug** (security / data-integrity)
- Priority: **P1** (public exposure of pre-hotfix behavior)
- Component: `reporium-api` deploy pipeline / Cloud Run posture

## Reproduction (pre-fix)

1. Merge any PR that hotfixes user-visible behavior (e.g. PR #433
   private-repo filter).
2. After deploy completes, find a previous merge's short SHA (every commit
   on `main` is public on GitHub).
3. `curl -s
   "https://candidate-<old_sha>---reporium-api-wypbzj5gpa-uc.a.run.app/stats"
   | jq .` returns aggregate counts from before the hotfix while the base URL
   returns the post-fix counts.

## Root cause

`gcloud run services update-traffic --to-tags=candidate-<sha>=100` shifts
traffic but **does not remove the tag** from previously-tagged revisions.
Cloud Run keeps tag → revision URL bindings independent of traffic split, so
the public bypass URL remains addressable indefinitely until manually
removed. Comment in `deploy.yml` previously claimed the leftover tag was
"harmless" — it is not.

## Fix shipped — PR #436 + refinement commit `bbb69ac`

`.github/workflows/deploy.yml`, two new post-promotion steps:

1. **Remove stale candidate-* traffic tags from non-serving revisions.**
   `gcloud run services describe ... --format=json | jq` enumerates every
   traffic entry with `percent == 0` and `tag` prefix `candidate-`, then
   strips them with one `update-traffic --remove-tags=…` call. Operator
   tags (`stable`, `rollback`) are not matched and are preserved.
2. **Verify no public candidate-* bypass URLs remain.** Re-describes the
   service and emits `::warning::` if any `candidate-*` tag still sits on a
   `percent=0` revision. Non-blocking (deploy already succeeded).

Both steps are `if: success()` + `continue-on-error: true`. The
no-traffic-candidate → migrate → promote invariant (Option B1, PR #397) is
untouched.

## Acceptance criteria

- [ ] Next deploy after merge logs either
      `No stale candidate-* traffic tags to remove.` or
      `Removing stale candidate traffic tags: candidate-…,candidate-…`.
- [ ] Verify step logs
      `No stale candidate-* tags on non-serving revisions — bypass paths closed.`
- [ ] `gcloud run services describe reporium-api --format=json | jq
      '.status.traffic[] | select(.tag != null)'` shows exactly one entry
      with `percent: 100`.
- [ ] At least one previously-known stale URL
      (e.g. `candidate-ca63b22---…/stats`) returns HTTP 404.
- [ ] Base URL `reporium-api-wypbzj5gpa-uc.a.run.app/stats` continues to
      return current production counts (currently 405 for HEAD; GET returns
      JSON).

## Out of scope (track separately)

- Service invoker is `allUsers`. Tagged URLs being public is a Cloud Run
  feature, not a misconfiguration. Long-term durable fix is to require
  authentication. Open as a follow-up; this ticket only closes the
  short-term bypass.
- No revision GC. We keep revisions for rollback; only tags are stripped.

## Today's manual cleanup (one-time)

`gcloud` shows live state at audit time has only one tagged revision
(`candidate-58ab8cd` at 100%, the serving one). Probe of the
previously-cited stale URL `candidate-ca63b22---…` returns HTTP 404,
confirming someone already stripped it manually. **No further manual
cleanup required pre-merge.** The fix is preventive against the next
deploy re-introducing the bypass.

If on a future audit a stale tag reappears before #436 is merged, run:

```bash
gcloud run services describe reporium-api \
  --project=perditio-platform \
  --region=us-central1 \
  --format=json \
  | jq -r '.status.traffic[]
           | select(.percent == 0 and .tag != null and (.tag | startswith("candidate-")))
           | .tag' \
  | sort -u \
  | paste -sd',' - \
  | xargs -I{} gcloud run services update-traffic reporium-api \
      --remove-tags={} \
      --project=perditio-platform \
      --region=us-central1
```

## Links

- PR: https://github.com/perditioinc/reporium-api/pull/436
- Underlying invariant: PR #397 (no-traffic candidate Option B1)
- Hotfix that surfaced the leak: PR #433 (newest/oldest smart route)
- Audit detail: `.audit/2026-04-23/candidate-tag-bypass.md`
