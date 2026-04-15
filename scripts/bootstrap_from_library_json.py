"""
Phase 3a: Seed repos table (and child tables) from library.json snapshot.

Usage (from Cloud Run Job):
    python scripts/bootstrap_from_library_json.py /tmp/library.json

The script is idempotent: all inserts use ON CONFLICT DO NOTHING.
"""
import asyncio
import json
import os
import sys
import uuid
from datetime import datetime, timezone


def parse_dt(s):
    if not s:
        return None
    if isinstance(s, datetime):
        return s
    try:
        # Handle both Z-suffix and +00:00
        s = s.replace("Z", "+00:00")
        return datetime.fromisoformat(s)
    except Exception:
        return None


async def main(library_path: str):
    import asyncpg

    INSTANCE = "perditio-platform:us-central1:reporium-db"
    PGPASS = os.environ.get("PGPASS", "")

    print(f"Loading {library_path}...")
    with open(library_path, encoding="utf-8") as f:
        data = json.load(f)

    repos = data["repos"]
    print(f"Loaded {len(repos)} repos")

    print("Connecting to Cloud SQL...")
    conn = await asyncpg.connect(
        host=f"/cloudsql/{INSTANCE}",
        port=5432,
        user="postgres",
        password=PGPASS,
        database="reporium",
    )
    print("Connected!")

    inserted_repos = 0
    skipped_repos = 0

    for repo in repos:
        full_name = repo.get("fullName", "")
        parts = full_name.split("/", 1)
        owner = parts[0] if len(parts) == 2 else "perditioinc"
        name = parts[1] if len(parts) == 2 else full_name

        repo_id_str = repo.get("id")
        try:
            repo_id = uuid.UUID(repo_id_str) if repo_id_str else uuid.uuid4()
        except ValueError:
            repo_id = uuid.uuid4()

        fork_sync = repo.get("forkSync") or {}
        parent_stats = repo.get("parentStats") or {}
        commit_stats = repo.get("commitStats") or {}
        is_fork = repo.get("isFork", False)

        # stars: for forks use parentStats.stars, for own repos use stars
        stargazers = repo.get("stars") if not is_fork else None
        parent_stars = parent_stats.get("stars") if is_fork else None
        parent_forks = parent_stats.get("forks") if is_fork else None
        parent_is_archived = parent_stats.get("isArchived", False) if is_fork else False

        db_secondary = repo.get("dbSecondaryCategories") or []
        if isinstance(db_secondary, str):
            db_secondary = [db_secondary]

        try:
            result = await conn.execute(
                """
                INSERT INTO repos (
                    id, name, owner, description,
                    is_fork, forked_from, primary_language,
                    github_url, fork_sync_state, behind_by, ahead_by,
                    github_created_at, upstream_created_at, forked_at,
                    your_last_push_at, upstream_last_push_at,
                    parent_stars, parent_forks, parent_is_archived,
                    stargazers_count, forks_count, open_issues_count,
                    commits_last_7_days, commits_last_30_days, commits_last_90_days,
                    readme_summary, license_spdx, quality_signals, security_signals,
                    primary_category, secondary_categories,
                    github_updated_at, ingested_at, updated_at
                ) VALUES (
                    $1, $2, $3, $4,
                    $5, $6, $7,
                    $8, $9, $10, $11,
                    $12, $13, $14,
                    $15, $16,
                    $17, $18, $19,
                    $20, $21, $22,
                    $23, $24, $25,
                    $26, $27, $28::jsonb, $29::jsonb,
                    $30, $31,
                    $32, NOW(), NOW()
                )
                ON CONFLICT (name) DO NOTHING
                """,
                repo_id,
                name,
                owner,
                repo.get("description"),
                is_fork,
                repo.get("forkedFrom"),
                repo.get("language"),
                repo.get("url", f"https://github.com/{full_name}"),
                fork_sync.get("state"),
                fork_sync.get("behindBy", 0),
                fork_sync.get("aheadBy", 0),
                parse_dt(repo.get("createdAt")),
                parse_dt(repo.get("upstreamCreatedAt")),
                parse_dt(repo.get("forkedAt")),
                parse_dt(repo.get("yourLastPushAt")),
                parse_dt(repo.get("upstreamLastPushAt")),
                parent_stars,
                parent_forks,
                parent_is_archived,
                stargazers,
                repo.get("forks", 0) or 0,
                repo.get("openIssuesCount", 0) or 0,
                commit_stats.get("last7Days", 0) or 0,
                commit_stats.get("last30Days", 0) or 0,
                commit_stats.get("last90Days", 0) or 0,
                repo.get("readmeSummary"),
                repo.get("licenseSpdx"),
                json.dumps(repo.get("qualitySignals")) if repo.get("qualitySignals") else None,
                json.dumps(repo.get("securitySignals")) if repo.get("securitySignals") else None,
                repo.get("primaryCategory") or repo.get("dbCategory"),
                db_secondary if db_secondary else None,
                parse_dt(repo.get("lastUpdated")),
            )
            if result == "INSERT 0 1":
                inserted_repos += 1
            else:
                skipped_repos += 1
        except Exception as e:
            print(f"ERROR inserting {name}: {e}")
            continue

        # Child tables — use resolved repo_id from DB to handle conflicts
        actual_id = await conn.fetchval("SELECT id FROM repos WHERE name = $1", name)
        if not actual_id:
            continue
        actual_id_str = str(actual_id)

        # repo_tags
        tags = repo.get("enrichedTags") or repo.get("topics") or []
        for tag in tags:
            if tag:
                await conn.execute(
                    "INSERT INTO repo_tags (repo_id, tag) VALUES ($1, $2) ON CONFLICT DO NOTHING",
                    actual_id, tag,
                )

        # repo_categories
        all_cats = repo.get("allCategories") or []
        primary_cat = repo.get("primaryCategory") or repo.get("dbCategory")
        for cat in all_cats:
            if cat:
                await conn.execute(
                    """INSERT INTO repo_categories (repo_id, category_id, category_name, is_primary)
                       VALUES ($1, $2, $3, $4) ON CONFLICT DO NOTHING""",
                    actual_id, cat.lower().replace(" ", "_"), cat, (cat == primary_cat),
                )

        # repo_ai_dev_skills (may be list of str or list of {skill, lifecycleGroup})
        for skill_entry in (repo.get("aiDevSkills") or []):
            skill = skill_entry.get("skill") if isinstance(skill_entry, dict) else skill_entry
            if skill:
                await conn.execute(
                    "INSERT INTO repo_ai_dev_skills (repo_id, skill) VALUES ($1, $2) ON CONFLICT DO NOTHING",
                    actual_id, skill,
                )

        # repo_pm_skills (same pattern)
        for skill_entry in (repo.get("pmSkills") or []):
            skill = skill_entry.get("skill") if isinstance(skill_entry, dict) else skill_entry
            if skill:
                await conn.execute(
                    "INSERT INTO repo_pm_skills (repo_id, skill) VALUES ($1, $2) ON CONFLICT DO NOTHING",
                    actual_id, skill,
                )

        # repo_industries
        for ind in (repo.get("industries") or []):
            if ind:
                await conn.execute(
                    "INSERT INTO repo_industries (repo_id, industry) VALUES ($1, $2) ON CONFLICT DO NOTHING",
                    actual_id, ind,
                )

        # repo_languages
        lang_breakdown = repo.get("languageBreakdown") or {}
        lang_pct = repo.get("languagePercentages") or {}
        for lang, bytes_val in lang_breakdown.items():
            pct = lang_pct.get(lang, 0.0) or 0.0
            await conn.execute(
                """INSERT INTO repo_languages (repo_id, language, bytes, percentage)
                   VALUES ($1, $2, $3, $4) ON CONFLICT DO NOTHING""",
                actual_id, lang, int(bytes_val or 0), float(pct),
            )

        # repo_builders
        for builder in (repo.get("builders") or []):
            login = builder.get("login")
            if login:
                await conn.execute(
                    """INSERT INTO repo_builders (repo_id, login, display_name, org_category, is_known_org)
                       VALUES ($1, $2, $3, $4, $5) ON CONFLICT DO NOTHING""",
                    actual_id,
                    login,
                    builder.get("name"),
                    builder.get("orgCategory"),
                    builder.get("isKnownOrg", False),
                )

        # repo_taxonomy
        for tax in (repo.get("taxonomy") or []):
            dim = tax.get("dimension")
            val = tax.get("value")
            if dim and val:
                await conn.execute(
                    """INSERT INTO repo_taxonomy (repo_id, dimension, raw_value, assigned_by)
                       VALUES ($1, $2, $3, $4)
                       ON CONFLICT (repo_id, dimension, raw_value) DO NOTHING""",
                    actual_id, dim, val, tax.get("assignedBy", "enrichment"),
                )

    await conn.close()
    print(f"\nBOOTSTRAP_COMPLETE: inserted={inserted_repos} skipped_conflicts={skipped_repos} total={len(repos)}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/library.json"
    asyncio.run(main(path))
