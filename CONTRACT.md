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
