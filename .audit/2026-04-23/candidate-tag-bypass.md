# Public candidate-revision bypass in reporium-api Cloud Run

**Finding.** After a promoted deploy, Cloud Run leaves the
`candidate-<short_sha>` traffic tag on the previous revision. Because
`reporium-api` is invoker=`allUsers`, each tagged revision exposes a permanent
public URL of the form:

```
https://candidate-<short_sha>---reporium-api-573778300586.us-central1.run.app
```

These URLs keep serving pre-promotion code against the **current** production
database. Observed: an old tagged revision returned `/stats = 1899 repos / 62 …`
while `main` returned `1855 / 18` (aggregate counts from before the private-repo
filter hotfix). Anyone who knows the old short SHAs — every merged commit is
public on GitHub — can bypass every subsequent hotfix.

## Why it exists

`.github/workflows/deploy.yml` intentionally deploys each new revision at
`no_traffic: true` with a `candidate-<sha>` tag (Option B1: no-traffic
candidate → migrate → promote). The tag is required for the promote step
(`gcloud run services update-traffic --to-tags=<tag>=100`). The prior comment
in the workflow claimed leaving the tag behind was "harmless" — it is not;
Cloud Run makes tagged revisions publicly addressable independent of traffic
split.

## Fix (this PR)

Two new steps in `deploy.yml`, both `continue-on-error: true` (cleanup must
never block a successful promotion):

1. **Remove stale `candidate-*` traffic tags from non-serving revisions.**
   Uses `gcloud run services describe --format=json | jq` to enumerate every
   traffic entry with `percent == 0` and a `candidate-*` tag, then issues a
   single `gcloud run services update-traffic --remove-tags=tag1,tag2,…`. The
   just-promoted revision has `percent == 100` so its tag is preserved.
   Operator-added tags (`stable`, `rollback`, etc.) do not match the prefix
   and are left alone.
2. **Verify no public candidate-\* bypass URLs remain.** Re-describes the
   service and emits `::warning::` annotations if any `candidate-*` tag still
   sits on a `percent=0` revision. Surfaces regressions (e.g. future gcloud
   flag changes) without failing the deploy.

The no-traffic candidate → migrate → promote invariant is untouched; the new
steps run only *after* the successful promotion step.

## Validation

### Prove old candidate URLs are no longer reachable after promotion

Run (from a workstation with `gcloud` auth to `perditio-platform`):

```bash
# List every currently-tagged revision URL on the service.
gcloud run services describe reporium-api \
  --project=perditio-platform \
  --region=us-central1 \
  --format=json \
  | jq -r '.status.traffic[]
           | select(.tag != null)
           | "\(.percent)%  tag=\(.tag)  rev=\(.revisionName)"'
```

After a post-fix deploy, exactly one `candidate-*` entry should be present and
it must show `100%` (the live revision). Any `0%  tag=candidate-*` row is a
regression.

Independently verify each stale URL returns `404` (Cloud Run's response when a
tag is absent):

```bash
# Replace <sha> with a previously-tagged short SHA.
curl -sI \
  "https://candidate-<sha>---reporium-api-573778300586.us-central1.run.app/stats" \
  | head -n1
# Expected: HTTP/2 404
```

And the base URL continues to serve the new code:

```bash
curl -s https://reporium-api-573778300586.us-central1.run.app/stats | jq .
# Expected: current production counts (e.g. 1855 / 18)
```

### Confirm main deploy flow still works

No change to the promotion semantics. On the next merge to `main`:

1. Watch the `Deploy to Cloud Run` workflow in GitHub Actions.
2. `Promote candidate revision to 100% traffic` must succeed (unchanged).
3. `Remove stale candidate-* traffic tags from non-serving revisions` should
   print `Removing stale candidate traffic tags: candidate-…,candidate-…` (or
   `No stale candidate-* traffic tags to remove.` on a clean service).
4. `Verify no public candidate-* bypass URLs remain` must print
   `No stale candidate-* tags on non-serving revisions — bypass paths closed.`
5. Post-deploy `Smoke test graph endpoint` step must still pass as before.

Rollback path is unchanged: `gcloud run services update-traffic
reporium-api --to-revisions=<prior-revision>=100` (the prior revision still
exists; only its *tag* was removed).

## Manual cleanup required for already-existing candidate tags

The new workflow steps only run on *future* deploys. To close the bypass
*today*, run this one-time cleanup against the current service:

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

Then re-run the validation query above — every `candidate-*` entry should be
`100%`. Confirm at least one previously-known stale URL (e.g.
`candidate-ca63b22---…`) returns HTTP 404.

No revision is deleted; only the tags are stripped. The base URL
(`reporium-api-…run.app`) and the live tag are untouched.

## Risk assessment

- **Blast radius of the PR:** post-promotion workflow steps only; no runtime
  code change; no change to traffic-split behavior during promotion.
- **Failure mode of the cleanup step:** `continue-on-error: true`. If gcloud
  flakes, the deploy still succeeds; the next deploy will retry cleanup.
- **Failure mode of the verify step:** warning-only. Does not gate traffic.
- **Not mitigated here:** the underlying invoker=`allUsers` posture. Tagged
  URLs are a *Cloud Run feature*, not a misconfiguration — the only durable
  fix is to remove the tag after promotion (this PR) or make the service
  authenticated (a larger change tracked separately).
