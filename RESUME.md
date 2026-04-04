# Agent Resume — reporium-api

Context for future Claude Code / Codex sessions.

## Infrastructure Decisions

### min-instances=1 (ADR-006, 2026-03-31)

- `min-instances=1` is **intentional** — eliminates cold start latency.
- Monthly cost: **$51.80** (1 vCPU, 2 GiB, us-central1, after free tier).
- Must **not** be reverted without a documented team decision referencing ADR-006.
- `deploy.yml` has `--min-instances=1` to preserve the setting across deploys.
- Future optimization: 3-year CUD commitment would reduce to ~$28/month.
