# API Security Scan Lane — execution note (2026-04-26)

## Exact current failure shape (verified live, then locally)

`Security Scan` workflow on `main` exited 1 with:

```
Security Grade: F
  Secrets:      No secrets detected in source files.
  Dependencies: Dependency check skipped: pip-audit not installed.
                (No known CVEs found in dependencies. when run locally
                 with pip-audit installed.)
  Workflows:    Found 18 warning(s) in GitHub Actions workflows.
  Files:        No sensitive files detected in the repository.
  History:      Found 1 potential secret(s) in git history.
```

Most recent failing runs at lane start: 24944058954, 24944029033,
24944026326. First failure landed with PR #440 (commit `0d295b7`,
2026-04-26 00:14 UTC). The previous run on PR #436 (`24943938837`) was
green.

## Workflow-local vs history-local breakdown

### Workflow check — 18 warnings, all "uses tag instead of SHA"

After classification, **0 are real "tag instead of SHA" violations and
18 are scanner false-positives** caused by a single regex bug in
`reporium-security`. Every action in `reporium-api` is already pinned to
a 40-char SHA; the scanner only fails to recognize it because the line
also has the standard inline `# vX.Y.Z` documentation comment.

The four lines that **were** real violations at lane start:

| File | Line | Reference |
|---|---|---|
| `nightly-invariants.yml` | 22 | `actions/checkout@v4` |
| `nightly-invariants.yml` | 25 | `actions/setup-python@v5` |
| `nightly-invariants.yml` | 46 | `actions/upload-artifact@v4` |
| `test.yml` | 94 | `perditioinc/perditio-devkit/.../on-test-failure.yml@main` |

These are real (tag/branch refs, not SHAs) and are fixed by this
lane's PR.

### History check — 1 finding, placeholder false-positive

Matched line:

```
+   ADMIN_API_KEY="<value>" \
```

Origin: an audit doc fenced shell snippet
(`.audit/2026-04-24/kan-data-quality-verification-jira.md`) showing
how to dry-run the data-quality gate script. `<value>` is a literal
angle-bracket placeholder — not a leaked secret. The regex
`(?i)api_key\s*=\s*["\'][^"\']{4,}["\']` matches because `<value>` is
7 characters; the existing skip list (`placeholder`, `example`,
`changeme`, `xxx`, etc.) does not include `<...>`-style placeholders.

**No real secret is leaked.** Git history rewrite is NOT warranted and
is also out of lane scope.

## Repo-local fix made (this lane)

Branch: `claude/feature/KAN-DRAFT-security-scan-fix`
Base: `origin/main` @ `502af14` (post PR #440 merge).
PR: https://github.com/perditioinc/reporium-api/pull/443

Changes:

```
.github/workflows/nightly-invariants.yml | 6 +++---
.github/workflows/test.yml               | 2 +-
2 files changed, 4 insertions(+), 4 deletions(-)
```

Replacements (SHAs verified against `gh api .../git/refs/tags`):

- `actions/checkout@v4` → `@11bd71901bbe5b1630ceea73d27597364c9af683 # v4.2.2`
  (matches the SHA already used by 8 other workflows in this repo)
- `actions/setup-python@v5` → `@a26af69be951a213d495a4c3e4e4022e16d87065 # v5.6.0`
  (matches the SHA used by every other setup-python step in this repo)
- `actions/upload-artifact@v4` → `@ea165f8d65b6e75b540449e92b4886f43607fa02 # v4.6.2`
  (current `v4` tag target as of 2026-04-26)
- `perditioinc/perditio-devkit/.../on-test-failure.yml@main` →
  `@aa588479da71164c0bd2ee493a76ee46830d10dc # main @ 2026-04-26`
  (current `main` HEAD; reusable workflows accept SHAs)

YAML parsed clean with `yaml.safe_load` for both files.

## Why CI stays red after the PR merges

The 4 real violations are eliminated, but the scanner still reports 18
warnings because pinning the previously-floating refs introduces 4 new
inline `# vX.Y.Z` comments — each of which trips the same regex
false-positive that the original 14 already trip. **Net workflow
warning count: 18 → 18.**

Going green requires a one-line fix in `reporium-security` (out of
this lane's scope). See
[security-scan-jira.md](security-scan-jira.md) Issue 2 for the precise
patch:

```python
# reporium-security/reporium_security/checks/workflows.py
action_ref = stripped.split("uses:")[-1].split("#", 1)[0].strip()
```

After that lands, scanning `reporium-api` post-PR will report
`0 warning(s)` for workflows. The history-check false positive needs
a separate one-line skip-list update in
`reporium-security/reporium_security/checks/history.py` (Issue 3 in
the JIRA fallback file).

## Validation performed

- `python -m reporium_security scan .` — full report captured before
  and after pinning. Count stayed at 18 warnings; the *content* of
  the warnings shifted from "real floating refs" to "false-positive
  on the new inline comments", confirming the underlying state is
  better even though the reported number is the same.
- `python -c "import yaml; yaml.safe_load(open(<file>).read())"` for
  both edited workflow files — clean parse.
- No `git filter-repo`, no force-push, no history rewrite of any kind.

## Exact next step if the upstream regex bug remains

Either:

1. **Preferred**: open a PR in `reporium-security` with the two
   one-line fixes in `checks/workflows.py` and `checks/history.py`
   described in the JIRA fallback. After that ships, this repo's
   `Security Scan` will go green on the very next push to `main`
   without any further reporium-api change.

2. **Alternative if upstream is blocked**: as a workaround, strip
   inline `# vX.Y.Z` comments from every `uses:` line in this repo's
   workflows. Sacrifices version readability for a green badge.
   Not recommended; the upstream regex fix is one line.

## Lane scope adherence

- Touched only `.github/workflows/*.yml` and lane-local audit notes.
- Did NOT touch `scripts/quality_gates.py`, `app/routers/platform.py`,
  or `tests/test_platform_metrics.py` (owned by the active
  data-quality recovery lane on a sibling branch).
- Did NOT modify `reporium-security` despite the regex bug being the
  load-bearing blocker — that repo is sibling to `reporium-api` and
  outside this lane's stated workspace.
