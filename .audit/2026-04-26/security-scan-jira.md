# JIRA fallback — Security Scan red on `main` (2026-04-26)

> JIRA was not reachable from this lane session. This file is the
> structured backup; copy fields verbatim into JIRA when available.

## Issue 1 — Pin floating action refs in `nightly-invariants.yml` and `test.yml`

- **Project**: KAN (Reporium API)
- **Type**: Task / Security
- **Component**: CI / GitHub Actions
- **Priority**: Medium
- **Status**: In review (PR opened from this lane)

**Summary**

Two workflow files referenced GitHub Actions by floating tag/branch
instead of by SHA. SHA pinning is required by the platform's own
`reporium-security` scanner and by GitHub's hardening guidance:

| File | Line | Was | Pin |
|---|---|---|---|
| `.github/workflows/nightly-invariants.yml` | 22 | `actions/checkout@v4` | `@11bd71901bbe5b1630ceea73d27597364c9af683 # v4.2.2` |
| `.github/workflows/nightly-invariants.yml` | 25 | `actions/setup-python@v5` | `@a26af69be951a213d495a4c3e4e4022e16d87065 # v5.6.0` |
| `.github/workflows/nightly-invariants.yml` | 46 | `actions/upload-artifact@v4` | `@ea165f8d65b6e75b540449e92b4886f43607fa02 # v4.6.2` |
| `.github/workflows/test.yml` | 94 | `perditioinc/perditio-devkit/.github/workflows/on-test-failure.yml@main` | `@aa588479da71164c0bd2ee493a76ee46830d10dc # main @ 2026-04-26` |

SHAs verified against `gh api repos/<owner>/<repo>/git/refs/tags`. The
`actions/checkout` and `actions/setup-python` SHAs match the values
already in use elsewhere in this repo, so this is not introducing a new
version surface.

**Acceptance**

- [x] Workflow YAML parses cleanly (verified: `yaml.safe_load`).
- [x] PR opened from `claude/feature/KAN-DRAFT-security-scan-fix`.
- [ ] After merge, the next `Tests` and `Nightly Data Invariants` runs
  on `main` succeed without floating-ref drift risk.

## Issue 2 — `reporium-security` scanner false-positive on inline version comments

- **Project**: KAN (or wherever `reporium-security` is owned)
- **Type**: Bug
- **Component**: reporium-security / scanner
- **Priority**: High (blocks `Security Scan` going green across every repo
  that follows the standard "SHA + `# vX.Y.Z`" pinning pattern)
- **Status**: Open — fix is upstream, NOT in `reporium-api` lane scope

**Summary**

`reporium-security/reporium_security/checks/workflows.py` rejects every
correctly-SHA-pinned action that has an inline version comment. After
this lane's fix is merged, **all 18 remaining workflow warnings** in
`reporium-api` are false positives of this single regex bug.

**Root cause**

```python
# reporium-security/reporium_security/checks/workflows.py
SHA_PATTERN = re.compile(r"@[0-9a-f]{40}$")

def _check_uses_sha(workflow_content, filename):
    for line_num, line in enumerate(workflow_content.splitlines(), start=1):
        stripped = line.strip()
        if "uses:" in stripped:
            action_ref = stripped.split("uses:")[-1].strip()
            ...
            if "@" in action_ref and not SHA_PATTERN.search(action_ref):
                findings.append(... "uses tag instead of SHA" ...)
```

For a line like:

```yaml
uses: actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065 # v5.6.0
```

`action_ref` becomes
`actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065 # v5.6.0`.
The regex anchors with `$`, so the trailing ` # v5.6.0` defeats the
match — the action is reported as "uses tag instead of SHA" even though
it is correctly pinned.

**Proposed fix (one line)**

Strip an inline comment from `action_ref` before the regex check:

```python
action_ref = stripped.split("uses:")[-1].split("#", 1)[0].strip()
```

Equivalent loosenings of the regex (e.g. drop `$`, or allow trailing
whitespace + `#`) are also acceptable. This is a one-line change in
`reporium-security`; this lane did not apply it because
`reporium-security` is outside the `reporium-api` lane scope.

**Repro**

Already reproducing on every push to `main` since 2026-04-26 00:14 UTC.
Most recent failing runs at prompt time:
- 24944058954
- 24944029033
- 24944026326

Local repro:
```
cd reporium-api && python -m reporium_security scan .
# → Workflows: Found 18 warning(s) in GitHub Actions workflows.
```

**Acceptance**

- [ ] After upstream patch + redeploy of `reporium-security`, scanning
  `reporium-api` (post this lane's PR) reports `0 warning(s)` from the
  Workflows check.
- [ ] Same scan re-run against `reporium`, `reporium-db`,
  `reporium-ingestion` etc. confirms the regex fix did not regress
  their already-green scans.

## Issue 3 — `History: Found 1 potential secret(s) in git history` is a placeholder false-positive

- **Project**: KAN (or `reporium-security`)
- **Type**: Bug — scanner false-positive
- **Component**: reporium-security / scanner / history check
- **Priority**: Medium

**Summary**

`reporium_security/checks/history.py` flagged commit
`0d295b75ae2af5360e2d6699a0534094da727711` (PR #440 — "fix(data-quality):
pass X-Admin-Key to /metrics/data-quality") as containing a "Hardcoded
API key".

Investigated: the matched line is from
`.audit/2026-04-24/kan-data-quality-verification-jira.md`, inside a
fenced shell-snippet showing how to dry-run the gate script:

```
ADMIN_API_KEY="<value>" \
```

`<value>` is a literal placeholder, not a leaked secret.  The
`Hardcoded API key` regex
(`(?i)api_key\s*=\s*["\'][^"\']{4,}["\']`) requires only 4+ chars
inside the quotes, and `<value>` is 7 chars.

**Why the existing skip list missed it**

`history.py` already filters lines containing `"placeholder"`,
`"example"`, `"changeme"`, `"xxx"`, etc., but does not skip the
`<...>` shell/man-page placeholder convention.

**Proposed fix (one of two)**

(a) In `history.py` skip-keyword list, add the angle-bracket
placeholder:
```python
if any(p in lower_line for p in (
    ..., 'placeholder', 'changeme', 'xxx',
    '<value>', '<your_', '<sha>', '<token>',  # NEW
)):
```

OR (preferred) (b) detect the placeholder shape with a tiny regex on
the matched substring rather than a fixed keyword list:
```python
if re.search(r'<[a-z_][a-z0-9_]*>', lower_line):
    continue
```

This is again a `reporium-security` change — out of `reporium-api` lane
scope.

**Acceptance**

- [ ] After upstream patch, the scan against `reporium-api` reports
  `History: No secrets detected in recent git history.`
- [ ] No real secrets get accidentally suppressed (regex requires a
  word inside angle brackets, not a 32+ char hex blob).

## Out of scope for this lane

- Rewriting git history. The "history" finding is a placeholder, not a
  real leak, so no `git filter-repo` is needed even if the lane
  permitted it.
- Touching `scripts/quality_gates.py`, `app/routers/platform.py`, or
  any data-quality file. Those are owned by the data-quality lane.
