# Repo Card Data Contract

Every repo returned by `/library/full` must satisfy this contract.
Fallbacks are applied by `sanitize_repo()` in `library_full.py` before the response is sent.
No field may ever be `null` or `undefined` in the API response.

## Required Fields (card will not render without these)

| Field | Type | Fallback | Source |
|-------|------|----------|--------|
| name | string | — (must exist) | repos.name |
| fullName | string | `{owner}/{name}` | repos.owner + repos.name |
| description | string | First 150 chars of readme_summary, or repo name | repos.description → repos.readme_summary |
| url | string | `https://github.com/{owner}/{name}` | repos.github_url |
| isFork | boolean | — (must be false for /library/full) | repos.is_fork |
| isPrivate | boolean | — (must be false for /library/full) | repos.is_private |
| language | string or null | null is acceptable (some repos have no language) | repos.primary_language |
| stars | number | 0 | repos.parent_stars |
| forks | number | 0 | repos.parent_forks |

## Enriched Fields (card shows degraded UI if missing, never null)

| Field | Type | Fallback | Source |
|-------|------|----------|--------|
| readmeSummary | string | description value | repos.readme_summary → description |
| primaryCategory | string | "Uncategorized" | repo_categories (is_primary=true) |
| allCategories | string[] | ["Uncategorized"] | repo_categories |
| enrichedTags | string[] | [] (empty array, never null) | repo_tags + repo_ai_dev_skills |
| builders | object[] | [{"login": owner, "name": null, ...}] | repo_builders → derive from owner |
| pmSkills | string[] | [] | repo_pm_skills |
| industries | string[] | [] | repo_industries |
| aiDevSkills | string[] | [] | repo_ai_dev_skills |
| programmingLanguages | string[] | [] | repo_languages |
| commitStats | object | {today:0, last7Days:0, last30Days:0, last90Days:0, recentCommits:[]} | repos columns |
| languageBreakdown | object | {} | repo_languages |
| languagePercentages | object | {} | repo_languages |

## Computed Fields (always derived, never stored)

| Field | Type | Derivation |
|-------|------|------------|
| id | number | hash of repo UUID |
| lastUpdated | string | github_updated_at or updated_at |
| isArchived | boolean | parent_is_archived or false |
| createdAt | string | ingested_at |
| weeklyCommitCount | number | commits_last_7_days or 0 |
| totalCommitsFetched | number | sum of commit counts or 0 |

## Validation Rules

1. `/library/full` MUST only return repos where `is_private = false`. Public forks (`is_fork=true, is_private=false`) ARE included — the frontend has a built/forked toggle. Private repos (`is_private=true`) are NEVER included regardless of fork status.
2. Every repo MUST have a non-empty `description` (apply fallback before response)
3. Every repo MUST have at least one category (fallback: "Uncategorized")
4. No field listed above may be `null` in the response (arrays default to [], objects to {})
5. `sanitize_repo()` runs on every repo before it enters the response
6. If a fallback is applied, a warning is logged so enrichment gaps are visible

## Quality Gate

The `/admin/data-quality` endpoint reports a `quality_score` (0-100).
The nightly audit workflow fails if `quality_score < 90`.
Any repo missing a required field after fallbacks indicates a data pipeline issue.

---

# Additional Endpoint Contract Notes (March 2026)

The sections above describe the `/library/full` repo-card contract.
The endpoints below were added later and are called by ingestion, operator tooling, or the frontend.

| Method | Path | Auth Required | Description | Key Request Fields | Key Response Fields |
|-------|------|---------------|-------------|--------------------|---------------------|
| `POST` | `/ingest/events/repo-ingested` | `Bearer` API key + `X-Ingest-Key`; optional Pub/Sub JWT when `PUBSUB_AUDIENCE` is set | Receives a repo-ingested event and refreshes taxonomy, gap analysis, and cached portfolio insights. | Pub/Sub push body or direct JSON body | `status`, `received`, `taxonomy_rebuild`, `taxonomy_embed`, `taxonomy_assign`, `gap_rebuild`, `portfolio_insights` |
| `GET` | `/gaps/taxonomy` | none | Returns underrepresented values across taxonomy dimensions. | Query: `min_repos`, `max_repos` | `dimension`, `name`, `repo_count`, `gap_score`, `severity` |
| `GET` | `/trends/report` | none | Returns the aggregated intelligence-sidebar trend report. | none | `generatedAt`, `period`, `trending`, `emerging`, `cooling`, `stable`, `newReleases`, `insights` |
| `POST` | `/admin/quality/compute` | `Bearer` API key + `X-Admin-Key` | Computes `quality_signals` for all repos from stored metadata without external API calls. | none | `computed`, `skipped` |
| `POST` | `/admin/embeddings/backfill` | `Bearer` API key + `X-Admin-Key` | Generates missing repo embeddings from stored repo text and tags. | none | `backfilled`, `errors` |
| `POST` | `/admin/taxonomy/bootstrap` | `Bearer` API key + `X-Admin-Key` | Assigns existing taxonomy values to repos that are missing taxonomy coverage. | Query: `limit`, optional `dimension` | `processed`, `assigned`, `errors` |
| `POST` | `/admin/taxonomy/deduplicate` | `Bearer` API key + `X-Admin-Key` | Groups similar taxonomy values inside a dimension and optionally merges them. | Query: `dimension`, optional `threshold`, optional `dry_run` | `dimension`, `dry_run`, `groups`, `merged` |
| `POST` | `/webhooks/github` | none in development; HMAC signature required when `GITHUB_WEBHOOK_SECRET` is configured | Receives GitHub `ping` and `push` webhooks and invalidates affected cache keys. | Headers: `X-Hub-Signature-256`, `X-GitHub-Event`; raw webhook payload | `status`, `event` |
| `GET` | `/admin/runs` | `X-Admin-Key` | Lists recent ingestion runs for dashboards and operator inspection. | Query: `limit` | Array of run objects with `id`, `run_mode`, `status`, `repos_upserted`, `repos_processed`, `errors`, `started_at`, `finished_at`, `duration_seconds` |
| `POST` | `/admin/runs` | `X-Admin-Key` | Records a completed ingestion run from the ingestion pipeline. | `run_mode`, `status`, `repos_upserted`, `repos_processed`, `errors`, `started_at`, `finished_at` | `id`, `status` |
| `GET` | `/health` | none | Runs a lightweight database-backed health check for Cloud Run and operators. | none | `status`, `db`, optional `detail` on degraded responses |
