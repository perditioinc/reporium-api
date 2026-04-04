# Reporium Naming & Field Conventions

This document is the authoritative reference for naming conventions across the entire Reporium stack. Violations cause the exact regressions documented in `CHANGELOG.md`. **Follow these rules — no exceptions.**

---

## 1. Language-Specific Naming Rules

### Python (reporium-api, reporium-ingestion)
- Variables, functions, method names: `snake_case`
- Class names: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Module/file names: `snake_case.py`
- SQLAlchemy model attributes: `snake_case` (match DB column names exactly)
- Pydantic schema fields: `snake_case` in the model; use `alias` only when required by an external contract

### TypeScript/JavaScript (reporium-frontend)
- Variables, functions, methods: `camelCase`
- Components: `PascalCase`
- Files: `camelCase.ts` / `PascalCase.tsx` for components
- Constants: `UPPER_SNAKE_CASE`
- API response field mapping: TypeScript receives `snake_case` from the API and maps to `camelCase` at the fetch boundary (in `lib/api.ts` or equivalent). **Never** rely on implicit camelCase coercion.

---

## 2. Database → API → Frontend Field Mapping Contract

The golden rule: **each field has one canonical name per layer**. The mapping must be explicit and documented here.

| DB Column (snake_case) | API JSON (snake_case) | Frontend (camelCase) | Notes |
|------------------------|-----------------------|----------------------|-------|
| `name` | `name` | `name` | |
| `owner` | `owner` | `owner` | |
| `description` | `description` | `description` | |
| `is_fork` | `is_fork` | `isFork` | |
| `is_private` | `is_private` | `isPrivate` | |
| `primary_language` | `primary_language` | `language` | aliased at frontend |
| `github_url` | `github_url` | `url` | aliased at frontend |
| `fork_sync_state` | `fork_sync_state` | `forkSyncState` | |
| `behind_by` | `behind_by` | `behindBy` | |
| `ahead_by` | `ahead_by` | `aheadBy` | |
| `parent_stars` | `parent_stars` | `stars` | aliased at frontend |
| `parent_forks` | `parent_forks` | `forks` | aliased at frontend |
| `stargazers_count` | `stargazers_count` | `stargazersCount` | own-repo stars (built repos) |
| `parent_is_archived` | `parent_is_archived` | `isArchived` | |
| `open_issues_count` | `open_issues_count` | `openIssuesCount` | |
| `license_spdx` | `license_spdx` | `licenseSpdx` | |
| `commits_last_7_days` | `commits_last_7_days` | `weeklyCommitCount` | aliased at frontend |
| `commits_last_30_days` | `commits_last_30_days` | `commitsLast30Days` | |
| `commits_last_90_days` | `commits_last_90_days` | `commitsLast90Days` | |
| `readme_summary` | `readme_summary` | `readmeSummary` | |
| `activity_score` | `activity_score` | `activityScore` | |
| `quality_signals` | `quality_signals` | `qualitySignals` | JSON object |
| `problem_solved` | `problem_solved` | `problemSolved` | |
| `ingested_at` | `ingested_at` | `createdAt` | aliased at frontend |
| `updated_at` | `updated_at` | `lastUpdated` | aliased at frontend |
| `github_updated_at` | `github_updated_at` | `githubUpdatedAt` | |
| `upstream_created_at` | `upstream_created_at` | `upstreamCreatedAt` | |
| `forked_at` | `forked_at` | `forkedAt` | |
| `your_last_push_at` | `your_last_push_at` | `yourLastPushAt` | |
| `upstream_last_push_at` | `upstream_last_push_at` | `upstreamLastPushAt` | |

### Junction Tables → API Arrays

| Junction Table | DB Columns | API Field | Frontend Field |
|----------------|-----------|-----------|----------------|
| `repo_tags` | `tag` | `tags: string[]` | `enrichedTags: string[]` |
| `repo_categories` | `category_id`, `category_name`, `is_primary` | `categories: [{category_id, category_name, is_primary}]` | `primaryCategory`, `allCategories` |
| `repo_builders` | `login`, `display_name`, `org_category`, `is_known_org` | `builders: [{login, display_name, org_category, is_known_org}]` | `builders` |
| `repo_ai_dev_skills` | `skill` | `ai_dev_skills: string[]` | `aiDevSkills: string[]` |
| `repo_pm_skills` | `skill` | `pm_skills: string[]` | `pmSkills: string[]` |
| `repo_languages` | `language`, `bytes`, `percentage` | `languages: [{language, bytes, percentage}]` | `programmingLanguages`, `languageBreakdown` |
| `repo_taxonomy` | `dimension`, `raw_value`, `similarity_score`, `assigned_by` | `taxonomy: [{dimension, value, similarityScore, assignedBy}]` | `taxonomy` |

---

## 3. Taxonomy System

Reporium has **two taxonomy tables** — never confuse them:

| Table | Purpose | Populated by |
|-------|---------|--------------|
| `repo_taxonomy` | Per-repo dimension assignments (e.g., `{repo_id, dimension="skill_area", raw_value="prompt-engineering"}`) | `POST /admin/taxonomy/assign` |
| `taxonomy_values` | Aggregate counts per dimension+value (used for filter chips, coverage badges) | `POST /admin/taxonomy/rebuild` |

**The old tables `skill_areas` and `repo_ai_dev_skills` are DEPRECATED.** Do not write new code that reads from or writes to them. All taxonomy reads must go through `repo_taxonomy` / `taxonomy_values`.

### Dimension Names (canonical)

| Dimension | Values example |
|-----------|---------------|
| `skill_area` | `prompt-engineering`, `rag`, `fine-tuning`, `agents`, `vision` |
| `use_case` | `code-generation`, `data-analysis`, `content-creation` |
| `maturity` | `experimental`, `production-ready`, `deprecated` |
| `integration` | `langchain`, `openai`, `huggingface`, `anthropic` |

---

## 4. Ingest Payload Convention

The ingest endpoint (`POST /ingest/repo`) uses the schema defined in `app/schemas/repo.py` (`RepoIngest`). Key rules:

1. **Skip-empty guard**: The `_upsert_repo()` function in `ingest.py` MUST NOT delete junction table rows when the ingest payload sends an empty array. Guard:
   ```python
   if item.tags:  # Only clear+replace when payload has data
       await db.execute(RepoTag.__table__.delete().where(...))
       for tag in item.tags:
           db.add(RepoTag(...))
   ```
   This guard is applied to: `tags`, `categories`, `builders`, `ai_dev_skills`, `pm_skills`, `languages`.

2. **Quick mode vs full mode**: Quick-mode ingestion (no README fetch) produces sparse tags. Junction table data must never be wiped by a quick-mode run — the skip-empty guard above ensures this.

3. **Field name in payload**: The ingest payload field `stargazers_count` maps to DB column `stargazers_count`. The ingestion pipeline `_to_api_payload()` must include `'stargazers_count': repo.stars` for built repos.

---

## 5. SQLAlchemy Relationship Loading

Any endpoint that accesses `repo.taxonomy`, `repo.tags`, `repo.categories`, etc. **must** include the corresponding `selectinload()` in its query options. Accessing a relationship without `selectinload` in an async context triggers `sqlalchemy.exc.MissingGreenlet`.

Required selectinloads for any repo query:
```python
.options(
    selectinload(Repo.tags),
    selectinload(Repo.categories),
    selectinload(Repo.builders),
    selectinload(Repo.ai_dev_skills),
    selectinload(Repo.pm_skills),
    selectinload(Repo.languages),
    selectinload(Repo.taxonomy),  # REQUIRED — omitting this crashes async endpoints
)
```

When serializing, use `getattr(repo, "taxonomy", [])` as a defensive guard in shared helpers (`_repo_to_summary()`) so test mocks (SimpleNamespace) don't crash:
```python
taxonomy=[
    {"dimension": t.dimension, "value": t.raw_value, ...}
    for t in getattr(repo, "taxonomy", [])
],
```

---

## 6. Cache Key Conventions

Cache keys follow the pattern `{resource}:{scope}:{discriminator}`.

| Key pattern | TTL | Invalidated by |
|-------------|-----|----------------|
| `library:full:{page}:{limit}` | 5 min | Full ingest, cache bust endpoint |
| `repos:list:{md5_hash}` | default | Ingest upsert |
| `repos:detail:{name}` | 10 min | Ingest upsert of that repo |
| `trends:latest` | 1 hr | `POST /ingest/trends/snapshot` |
| `trends:report` | 1 hr | `POST /ingest/trends/snapshot` (**both** keys must be invalidated) |

**Rule**: When a write operation can affect multiple cache keys, invalidate **all affected keys** in the same transaction. Missing `trends:report` while only invalidating `trends:latest` caused KAN-56.

---

## 7. API Authentication

| Header | Used for |
|--------|---------|
| `Authorization: Bearer <INGESTION_API_KEY>` | Ingest endpoints (`/ingest/*`) |
| `Authorization: Bearer <INGESTION_API_KEY>` + `X-Admin-Key: <ADMIN_API_KEY>` | Admin endpoints (`/admin/*`) |

Environment variables:
- `INGESTION_API_KEY` — required for ingest + admin routes
- `ADMIN_API_KEY` — required for admin-only routes (double-key protection)

Both must be set as GitHub Actions secrets (`INGEST_API_KEY`, `ADMIN_API_KEY`) for the post-deploy taxonomy rebuild to fire automatically.

---

## 8. Error Budget Rules

- **Never regress junction table data** — the skip-empty guard is non-negotiable
- **Never add a new relationship access** without a corresponding `selectinload` — async context requires eager loading
- **Never add a new cache write** without auditing all keys that should be invalidated on the corresponding write path
- **Never read from deprecated tables** (`skill_areas`, `repo_ai_dev_skills` for enrichment output) — use `repo_taxonomy`/`taxonomy_values`
- **Never rely on in-memory cache surviving a Cloud Run cold start** — treat every request as cache-cold

---

*Last updated: 2026-03-25. Update this file whenever a new field, table, or cache key is added.*
