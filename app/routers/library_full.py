"""
GET /library/full — Returns the complete dataset in the exact shape
the reporium.com frontend expects (LibraryData TypeScript interface).

All fields camelCase, nested objects, all repos in one response.
Cached for 5 minutes to avoid repeated expensive queries.
"""

import asyncio
import logging
import time
from collections import defaultdict
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Query, Request, Response
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.cache_redis import redis_cache
from app.database import get_db
# KAN-188: aggregate builders moved into library_aggregates_helpers so the new
# /library/aggregates endpoint can reuse them without paying the per-repo array
# cost. We re-export the constants + helper functions below for back-compat
# with tests/external code that imports the old names from this module.
from app.routers.library_aggregates_helpers import (
    KNOWN_ORG_CATEGORIES,
    LIFECYCLE_GROUPS,
    SYSTEM_TAGS,
    _AI_DEV_SKILL_SET,
    _AI_DEV_SKILLS_ORDERED,
    _LIFECYCLE_GROUPS_FALLBACK,
    _SKILL_TAG_TO_GROUP,
    _TAXONOMY_RAW_TO_CANONICAL,
    build_ai_dev_skill_stats,
    build_builder_stats,
    build_categories,
    build_gap_analysis,
    build_skill_stats,
    build_stats,
    build_tag_metrics,
)

# Back-compat aliases — historical underscore-prefixed names that tests and
# downstream callers import directly from app.routers.library_full. Aliasing
# keeps those imports working after the helpers moved out.
_build_ai_dev_skill_stats = build_ai_dev_skill_stats
_build_builder_stats = build_builder_stats
_build_categories = build_categories
_build_skill_stats = build_skill_stats
_build_stats = build_stats
_build_tag_metrics = build_tag_metrics

logger = logging.getLogger(__name__)

# Map ingestion category names → frontend canonical category names
# Frontend CATEGORIES constant uses these exact names with colors/icons
CATEGORY_MAP = {
    "Agents": "AI Agents",
    "Tooling": "Dev Tools & Automation",
    "Security": "Security & Safety",
    "Observability": "Observability & Monitoring",
    "Research": "Learning Resources",
    "Ocr": "Computer Vision",
    "Vision": "Computer Vision",
    "Llm Serving": "Inference & Serving",
    "Orchestration": "AI Agents",
    "Rag": "RAG & Retrieval",
    "Vector Databases": "MLOps & Infrastructure",
    "Other": "Dev Tools & Automation",
    "Data Processing": "MLOps & Infrastructure",
    "Embeddings": "RAG & Retrieval",
    "Audio": "Industry: Audio & Music",
    "Fine Tuning": "Model Training",
    "Deployment": "MLOps & Infrastructure",
    "Evaluation": "Evals & Benchmarking",
    "Datasets": "Datasets",
}


def _normalize_category(name: str) -> str:
    """Map raw DB category name to the frontend's canonical name."""
    return CATEGORY_MAP.get(name, name)


# NOTE: KAN-188 moved SYSTEM_TAGS, KNOWN_ORG_CATEGORIES, _AI_DEV_SKILLS_ORDERED,
# _LIFECYCLE_GROUPS_FALLBACK, _AI_DEV_SKILL_SET, _SKILL_TAG_TO_GROUP, and
# _TAXONOMY_RAW_TO_CANONICAL into app.routers.library_aggregates_helpers.
# They're imported at the top of this file and re-exported here for back-compat.

_lifecycle_groups_cache: dict = {}
_LIFECYCLE_GROUPS_TTL = 300  # 5 minutes


async def _get_lifecycle_groups(db: AsyncSession) -> dict:
    """Return {skill_area_name: lifecycle_group}.

    taxonomy_values does not carry a lifecycle_group column, so this function
    returns the compile-time fallback dict which encodes the 28-skill taxonomy.
    The async signature is kept so call sites do not need to change.
    """
    return _LIFECYCLE_GROUPS_FALLBACK

from app.rate_limit import rate_limit_storage

router = APIRouter(tags=["Library"])
_limiter = Limiter(key_func=get_remote_address, storage_uri=rate_limit_storage)

# In-memory cache: two tiers
#   _cache["page_{page}_{page_size}"] → per-page enriched repos (5 min TTL)
#   _cache["aggregates"]              → stats/categories/tagMetrics across all repos (5 min TTL)
_cache: dict = {}
CACHE_TTL = 300  # 5 minutes


def invalidate_library_cache() -> None:
    """Bust the in-memory /library/full cache and shared Redis library cache.

    Called by the ingest router after writes. The Redis ``library:`` prefix
    sweep covers BOTH /library/full's keys (``library:page:...``) and
    /library/preview's keys (``library:preview:...``) — keep it broad so any
    future ``library:*`` cache surface is invalidated automatically. This
    satisfies feedback_backfill_must_invalidate_cache.md for /library/preview.

    KAN-175: also sweeps the ``taxonomy:`` prefix so /taxonomy/categories and
    /taxonomy/tags (which aggregate from the same ``repos``/``repo_tags`` rows
    that ingest writes) cannot serve stale aggregates after a backfill.
    """
    _cache.clear()
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(redis_cache.clear_prefix("library:"))
            asyncio.ensure_future(redis_cache.clear_prefix("taxonomy:"))
        else:
            loop.run_until_complete(redis_cache.clear_prefix("library:"))
            loop.run_until_complete(redis_cache.clear_prefix("taxonomy:"))
    except Exception:
        logger.warning("invalidate_library_cache: could not clear Redis prefix", exc_info=True)


def sanitize_repo(repo: dict) -> dict:
    """
    Enforce CONTRACT.md — every field gets a valid value, never null.
    Logs a warning for each fallback applied so enrichment gaps are visible.
    """
    name = repo.get("name", "unknown")

    # Required fields — apply fallbacks
    if not repo.get("description"):
        summary = repo.get("readmeSummary") or ""
        repo["description"] = summary[:150] if summary else name
        logger.warning("Contract fallback: %s missing description", name)

    if not repo.get("readmeSummary"):
        repo["readmeSummary"] = repo["description"]
        logger.warning("Contract fallback: %s missing readmeSummary", name)

    if repo.get("openIssuesCount") is None:
        repo["openIssuesCount"] = 0

    if not repo.get("primaryCategory") or repo["primaryCategory"] == "Other":
        repo["primaryCategory"] = "Uncategorized"

    if not repo.get("allCategories"):
        repo["allCategories"] = [repo["primaryCategory"]]
        logger.warning("Contract fallback: %s missing categories", name)

    if not repo.get("enrichedTags"):
        repo["enrichedTags"] = []

    if not repo.get("builders"):
        # For forks, use upstream owner as builder; for owned repos, use the repo owner
        forked_from = repo.get("forkedFrom") or ""
        if forked_from and "/" in forked_from:
            owner = forked_from.split("/")[0]
        else:
            owner = repo.get("fullName", "").split("/")[0] if repo.get("fullName") else "perditioinc"
        repo["builders"] = [{"login": owner, "name": None, "type": "user",
                             "avatarUrl": f"https://avatars.githubusercontent.com/{owner}",
                             "isKnownOrg": False, "orgCategory": "individual"}]

    if not repo.get("pmSkills"):
        repo["pmSkills"] = []
    if not repo.get("industries"):
        repo["industries"] = []
    if not repo.get("aiDevSkills"):
        repo["aiDevSkills"] = []
    if not repo.get("programmingLanguages"):
        repo["programmingLanguages"] = []
    if not repo.get("topics"):
        repo["topics"] = []

    # Date fields — conservative fallbacks only for fields that have safe proxies.
    # Never substitute ingested_at for upstream_created_at — that shows the wrong date.
    last_updated = repo.get("lastUpdated") or ""
    if last_updated:
        ps = repo.get("parentStats")
        if ps and not ps.get("lastCommitDate"):
            ps["lastCommitDate"] = last_updated
        if not repo.get("upstreamLastPushAt"):
            repo["upstreamLastPushAt"] = last_updated if repo.get("isFork") else ""
        # Do NOT fall back upstreamCreatedAt to createdAt/ingested_at — that produces a
        # misleading "Project created: Mar 2026" for repos that were created years ago.
        # Leave it empty until a proper GitHub API backfill populates upstream_created_at.

    # Commit stats — never null
    if not repo.get("commitStats"):
        repo["commitStats"] = {"today": 0, "last7Days": 0, "last30Days": 0,
                               "last90Days": 0, "recentCommits": []}

    # Arrays that must never be null
    for arr_field in ("recentCommits", "commitsLast7Days", "commitsLast30Days", "commitsLast90Days"):
        if not repo.get(arr_field):
            repo[arr_field] = []

    # Objects that must never be null
    if not repo.get("languageBreakdown"):
        repo["languageBreakdown"] = {}
    if not repo.get("languagePercentages"):
        repo["languagePercentages"] = {}

    # Scalars with safe defaults
    if repo.get("stars") is None:
        repo["stars"] = 0
    if repo.get("forks") is None:
        repo["forks"] = 0
    if repo.get("weeklyCommitCount") is None:
        repo["weeklyCommitCount"] = 0
    if repo.get("totalCommitsFetched") is None:
        repo["totalCommitsFetched"] = 0

    return repo


def _iso(val) -> str:
    """Convert a datetime or string to ISO format string."""
    if val is None:
        return ""
    if isinstance(val, datetime):
        return val.isoformat()
    return str(val)


def _build_enriched_repo(repo: dict, languages: list, categories: list,
                         ai_skills: list, tags: list, pm_skills: list,
                         builders: list = None, industries: list = None,
                         lifecycle_groups: dict = None,
                         taxonomy: list = None,
                         commits: list = None) -> dict:
    """Transform a DB repo row + junction data into the frontend EnrichedRepo shape."""
    forked_from = repo.get("forked_from")
    owner = repo.get("owner", "perditioinc")
    name = repo.get("name", "")

    # Build language breakdown
    lang_breakdown = {}
    lang_percentages = {}
    for lang in languages:
        lang_breakdown[lang["language"]] = lang.get("bytes", 0)
        lang_percentages[lang["language"]] = lang.get("percentage", 0)

    # Build parent stats if forked
    parent_stats = None
    if forked_from:
        parts = forked_from.split("/", 1)
        parent_owner = parts[0] if len(parts) == 2 else ""
        parent_repo = parts[1] if len(parts) == 2 else forked_from
        parent_stats = {
            "owner": parent_owner,
            "repo": parent_repo,
            "stars": repo.get("parent_stars") or 0,
            "forks": repo.get("parent_forks") or 0,
            "openIssues": 0,
            "lastCommitDate": _iso(repo.get("upstream_last_push_at")),
            "isArchived": repo.get("parent_is_archived") or False,
            "description": repo.get("description"),
            "url": f"https://github.com/{forked_from}",
        }

    # Fork sync status — behind_by/ahead_by are often stale (0) in the DB,
    # so cross-check with dates: if upstream pushed after our last sync,
    # the fork is behind regardless of what the commit counts say.
    fork_sync = None
    if repo.get("is_fork"):
        behind = repo.get("behind_by") or 0
        ahead = repo.get("ahead_by") or 0

        # Date-based override: compare your_last_push_at vs upstream_last_push_at
        your_push = repo.get("your_last_push_at")
        upstream_push = repo.get("upstream_last_push_at")
        date_says_behind = False
        if your_push and upstream_push:
            import datetime as _dt_mod
            def _to_naive_utc(v):
                """Convert to naive UTC datetime for safe comparison."""
                if isinstance(v, _dt_mod.datetime):
                    if v.tzinfo is not None:
                        return v.replace(tzinfo=None)
                    return v
                if isinstance(v, str):
                    try:
                        dt = _dt_mod.datetime.fromisoformat(v.replace("Z", "+00:00"))
                        return dt.replace(tzinfo=None)
                    except Exception:
                        return None
                return None
            yp = _to_naive_utc(your_push)
            up = _to_naive_utc(upstream_push)
            if yp and up and up > yp:
                date_says_behind = True

        if behind == 0 and ahead == 0 and not date_says_behind:
            state = "up-to-date"
        elif date_says_behind or behind > 0:
            if behind > 0 and ahead > 0:
                state = "diverged"
            else:
                state = "behind"
                # If commit count is 0 but dates say behind, estimate ~1
                if behind == 0:
                    behind = 1
        elif ahead > 0:
            state = "ahead"
        else:
            state = "unknown"
        fork_sync = {
            "state": state,
            "behindBy": behind,
            "aheadBy": ahead,
            "upstreamBranch": "main",
        }

    c7 = repo.get("commits_last_7_days") or 0
    c30 = repo.get("commits_last_30_days") or 0
    c90 = repo.get("commits_last_90_days") or 0

    # Bin commit history from repo_commits table into time buckets
    all_commit_data = commits or []
    now = datetime.now(tz=None)  # naive UTC-ish for comparison
    commits_7d = []
    commits_30d = []
    commits_90d = []
    for cmt in all_commit_data:
        date_str = cmt.get("date", "")
        if not date_str:
            continue
        try:
            cdate = datetime.fromisoformat(date_str.replace("Z", "+00:00")).replace(tzinfo=None)
        except (ValueError, TypeError):
            continue
        days_ago = (now - cdate).days
        if days_ago <= 7:
            commits_7d.append(cmt)
        if days_ago <= 30:
            commits_30d.append(cmt)
        if days_ago <= 90:
            commits_90d.append(cmt)

    # Use actual commit counts when DB scalars are 0 but we have commit rows
    effective_c7 = max(c7, len(commits_7d))
    effective_c30 = max(c30, len(commits_30d))
    effective_c90 = max(c90, len(commits_90d))

    all_cats = list(dict.fromkeys(_normalize_category(c["category_name"]) for c in categories))
    primary_cat = all_cats[0] if all_cats else "Dev Tools & Automation"

    # Use the DB-computed full_name (owner/name) as canonical identity.
    # Falls back to constructing it if the column is somehow NULL (shouldn't happen post-006).
    full_name = repo.get("full_name") or f"{owner}/{name}"

    return {
        "id": str(repo.get("id")),  # Stable DB UUID — never use hash() which changes per restart
        "name": name,
        "fullName": full_name,
        "description": repo.get("description"),
        "isFork": repo.get("is_fork", False),
        # SECURITY: `isPrivate` is emitted on every repo so the frontend's
        # `validate:privacy` gate (reporium#278) can fail the build closed if any
        # row is missing the field or carries `true`. The DB filter above already
        # excludes `is_private = true`, so this is defense-in-depth at the wire.
        "isPrivate": bool(repo.get("is_private", False)),
        "forkedFrom": forked_from,
        "language": repo.get("primary_language"),
        "topics": [t["tag"] for t in tags],
        "enrichedTags": list(dict.fromkeys([s["skill"] for s in ai_skills] + [t["tag"] for t in tags])),
        "stars": repo.get("parent_stars") if repo.get("is_fork") else (repo.get("stargazers_count") or 0),
        "forks": repo.get("parent_forks") if repo.get("is_fork") else (repo.get("fork_count") or 0),
        "openIssuesCount": repo.get("open_issues_count") or 0,
        "lastUpdated": _iso(repo.get("github_updated_at") or repo.get("updated_at")),
        "url": repo.get("github_url") or f"https://github.com/{owner}/{name}",
        "isArchived": repo.get("parent_is_archived") or False,
        "readmeSummary": repo.get("readme_summary"),
        "parentStats": parent_stats,
        "recentCommits": all_commit_data[:10],
        "createdAt": _iso(
            repo.get("upstream_created_at")
            if repo.get("forked_from")
            else (repo.get("ingested_at") or repo.get("github_updated_at"))
        ),
        "forkedAt": _iso(repo.get("forked_at")),
        "yourLastPushAt": _iso(repo.get("your_last_push_at")),
        "upstreamLastPushAt": _iso(repo.get("upstream_last_push_at")),
        "upstreamCreatedAt": _iso(repo.get("upstream_created_at")),
        "forkSync": fork_sync,
        "weeklyCommitCount": effective_c7,
        "languageBreakdown": lang_breakdown,
        "languagePercentages": lang_percentages,
        "commitsLast7Days": commits_7d,
        "commitsLast30Days": commits_30d,
        "commitsLast90Days": commits_90d,
        "totalCommitsFetched": len(all_commit_data),
        "primaryCategory": primary_cat,
        "allCategories": all_cats,
        "commitStats": {
            "today": len([c for c in commits_7d if c.get("date") and (now - datetime.fromisoformat(c["date"].replace("Z", "+00:00")).replace(tzinfo=None)).days == 0]),
            "last7Days": effective_c7,
            "last30Days": effective_c30,
            "last90Days": effective_c90,
            "recentCommits": all_commit_data[:5],
        },
        "latestRelease": None,
        "aiDevSkills": [
            {"skill": s["skill"], "lifecycleGroup": (lifecycle_groups or _LIFECYCLE_GROUPS_FALLBACK).get(s["skill"], "")}
            for s in ai_skills
        ],
        "pmSkills": [s["skill"] for s in pm_skills],
        "industries": [ind["industry"] for ind in (industries or [])],
        "programmingLanguages": list(lang_breakdown.keys()),
        "taxonomy": [
            {
                "dimension": t["dimension"],
                "value": t["raw_value"],
                "similarityScore": t["similarity_score"],
                "assignedBy": t["assigned_by"],
            }
            for t in (taxonomy or [])
        ],
        "problemSolved": repo.get("problem_solved"),
        "licenseSpdx": repo.get("license_spdx"),
        "qualitySignals": repo.get("quality_signals"),
        "securitySignals": repo.get("security_signals"),
        "dbCategory": repo.get("primary_category"),
        "dbSecondaryCategories": repo.get("secondary_categories") or [],
        "builders": [
            {
                "login": b["login"],
                "name": b.get("display_name") or b["login"],
                "type": "organization" if b.get("is_known_org") else "user",
                "avatarUrl": f"https://avatars.githubusercontent.com/{b['login']}",
                "isKnownOrg": b.get("is_known_org", False),
                "orgCategory": b.get("org_category"),
            }
            for b in (builders or [])
        ],
    }



async def _fetch_page_repos(
    db: AsyncSession, page: int, page_size: int
) -> tuple[list[dict], int]:
    """
    Fetch one page of enriched repos. Junction data is fetched only for the
    current page's IDs — never the full table — so memory is O(page_size), not O(N).
    Returns (enriched_repos, total_count).
    """
    offset = (page - 1) * page_size

    # Main repos query — paginated
    # SECURITY: `is_private` is selected (and the WHERE clause filters it to false)
    # so the response payload can carry the field forward. Downstream consumers
    # (reporium frontend's `validate:privacy` gate, audit harness) require every
    # row to expose `isPrivate` so a future regression that drops the WHERE clause
    # is detected at the artifact boundary instead of silently leaking.
    result = await db.execute(text("""
        SELECT id, name, owner, (owner || '/' || name) AS full_name, description, is_fork, forked_from, primary_language,
               is_private,
               github_url, fork_sync_state, behind_by, ahead_by,
               github_created_at, upstream_created_at, forked_at, your_last_push_at, upstream_last_push_at,
               parent_stars, parent_forks, parent_is_archived, stargazers_count, open_issues_count,
               commits_last_7_days, commits_last_30_days, commits_last_90_days,
               readme_summary, activity_score, ingested_at, updated_at, github_updated_at,
               problem_solved, license_spdx, quality_signals, has_tests, has_ci, security_signals,
               primary_category, secondary_categories
        FROM repos
        WHERE is_private = false
        -- KAN-190: id ASC tiebreaker is REQUIRED for deterministic pagination.
        -- Without it, rows with equal sort-key (e.g. all test fixtures share
        -- parent_stars=1000) get implementation-defined order across LIMIT/OFFSET
        -- pages, so a row at a page boundary can appear twice or be skipped
        -- on different connections — surfaced as
        -- test_library_full_excludes_private_repos_across_all_pages flake when
        -- a sibling test changes corpus size.
        ORDER BY COALESCE(parent_stars, stargazers_count, 0) DESC, id ASC
        LIMIT :lim OFFSET :off
    """), {"lim": page_size, "off": offset})
    rows = result.fetchall()
    columns = list(result.keys())

    count_result = await db.execute(text(
        "SELECT COUNT(*) FROM repos WHERE is_private = false"
    ))
    total = count_result.scalar() or 0

    if not rows:
        return [], total

    # Extract just this page's IDs for targeted junction fetches
    repo_dicts = [dict(zip(columns, row)) for row in rows]
    page_ids = [str(r["id"]) for r in repo_dicts]

    # Fetch junction data only for this page.
    # CAST(:ids AS uuid[]) rather than :ids::uuid[] — asyncpg's parameter parser
    # chokes on the :: suffix immediately after a named bind parameter.
    async def _fetch_junction(q: str) -> list:
        r = await db.execute(text(q), {"ids": page_ids})
        return r.fetchall()

    # Run junction fetches sequentially — asyncpg connections are strictly serial;
    # concurrent await calls on the same session cause InterfaceError.
    # Migration 036 added indexes so each query is a fast index scan.
    lang_rows = await _fetch_junction("SELECT repo_id, language, bytes, percentage FROM repo_languages WHERE repo_id = ANY(CAST(:ids AS uuid[]))")
    cat_rows = await _fetch_junction("SELECT repo_id, category_name, is_primary FROM repo_categories WHERE repo_id = ANY(CAST(:ids AS uuid[]))")
    skill_rows = await _fetch_junction("SELECT repo_id, raw_value AS skill FROM repo_taxonomy WHERE dimension = 'skill_area' AND repo_id = ANY(CAST(:ids AS uuid[]))")
    tag_rows = await _fetch_junction("SELECT repo_id, tag FROM repo_tags WHERE repo_id = ANY(CAST(:ids AS uuid[]))")
    pm_rows = await _fetch_junction("SELECT repo_id, skill FROM repo_pm_skills WHERE repo_id = ANY(CAST(:ids AS uuid[]))")
    builder_rows = await _fetch_junction("SELECT repo_id, login, display_name, org_category, is_known_org FROM repo_builders WHERE repo_id = ANY(CAST(:ids AS uuid[]))")
    taxonomy_rows = await _fetch_junction("SELECT repo_id, dimension, raw_value, similarity_score, assigned_by FROM repo_taxonomy WHERE repo_id = ANY(CAST(:ids AS uuid[]))")
    commit_rows = await _fetch_junction(
        "SELECT repo_id, sha, message, author, committed_at, url FROM repo_commits "
        "WHERE repo_id = ANY(CAST(:ids AS uuid[])) ORDER BY committed_at DESC"
    )
    industry_rows = await _fetch_junction("SELECT repo_id, industry FROM repo_industries WHERE repo_id = ANY(CAST(:ids AS uuid[]))")

    all_languages: dict = defaultdict(list)
    for r in lang_rows:
        all_languages[str(r.repo_id)].append({"language": r.language, "bytes": r.bytes, "percentage": r.percentage})

    all_categories: dict = defaultdict(list)
    for r in cat_rows:
        all_categories[str(r.repo_id)].append({"category_name": r.category_name, "is_primary": r.is_primary})

    all_ai_skills: dict = defaultdict(list)
    for r in skill_rows:
        all_ai_skills[str(r.repo_id)].append({"skill": r.skill})

    all_tags: dict = defaultdict(list)
    for r in tag_rows:
        all_tags[str(r.repo_id)].append({"tag": r.tag})

    all_pm_skills: dict = defaultdict(list)
    for r in pm_rows:
        all_pm_skills[str(r.repo_id)].append({"skill": r.skill})

    all_builders: dict = defaultdict(list)
    for r in builder_rows:
        all_builders[str(r.repo_id)].append({
            "login": r.login, "display_name": r.display_name,
            "org_category": r.org_category, "is_known_org": r.is_known_org,
        })

    all_industries: dict = defaultdict(list)
    for r in industry_rows:
        all_industries[str(r.repo_id)].append({"industry": r.industry})

    all_taxonomy: dict = defaultdict(list)
    for r in taxonomy_rows:
        all_taxonomy[str(r.repo_id)].append({
            "dimension": r.dimension,
            "raw_value": r.raw_value,
            "similarity_score": r.similarity_score,
            "assigned_by": r.assigned_by,
        })

    all_commits: dict = defaultdict(list)
    for r in commit_rows:
        all_commits[str(r.repo_id)].append({
            "sha": r.sha,
            "message": r.message,
            "author": r.author,
            "date": r.committed_at.isoformat() if r.committed_at else "",
            "url": r.url or "",
        })

    lifecycle_groups = await _get_lifecycle_groups(db)

    enriched = []
    for repo in repo_dicts:
        rid = str(repo["id"])
        enriched.append(sanitize_repo(_build_enriched_repo(
            repo,
            languages=all_languages.get(rid, []),
            categories=all_categories.get(rid, []),
            ai_skills=all_ai_skills.get(rid, []),
            tags=all_tags.get(rid, []),
            pm_skills=all_pm_skills.get(rid, []),
            builders=all_builders.get(rid, []),
            industries=all_industries.get(rid, []),
            lifecycle_groups=lifecycle_groups,
            taxonomy=all_taxonomy.get(rid, []),
            commits=all_commits.get(rid, []),
        )))

    return enriched, total


async def _fetch_aggregates(db: AsyncSession) -> dict:
    """
    Compute library-wide aggregates (stats, categories, tagMetrics, etc.) by loading
    all repos in pages to avoid a single OOM-inducing fetch.

    This runs at most once per CACHE_TTL window — cached under _cache['aggregates'].
    """
    now = time.time()
    cached = _cache.get("aggregates")
    if cached and cached.get("expires_at", 0) > now:
        return cached["data"]

    t0 = time.monotonic()
    all_repos: list[dict] = []
    page = 1
    while True:
        page_repos, total = await _fetch_page_repos(db, page=page, page_size=500)
        all_repos.extend(page_repos)
        if len(all_repos) >= total or not page_repos:
            break
        page += 1

    aggregates = {
        "stats": _build_stats(all_repos),
        "tagMetrics": _build_tag_metrics(all_repos),
        "categories": _build_categories(all_repos),
        "builderStats": _build_builder_stats(all_repos),
        "aiDevSkillStats": _build_ai_dev_skill_stats(all_repos, lifecycle_groups=await _get_lifecycle_groups(db)),
        "pmSkillStats": _build_skill_stats(all_repos, "pmSkills"),
    }
    logger.info(f"Aggregates built in {time.monotonic() - t0:.1f}s across {len(all_repos)} repos")

    _cache["aggregates"] = {"data": aggregates, "expires_at": now + CACHE_TTL}
    return aggregates


@router.get("/library/full", response_model=dict)
# 60/minute: the frontend paginates (page_size=500, ~4 pages) so a single
# homepage load consumes ~4 requests. 5/minute caused 429s on any refresh.
# Responses are served from Redis + in-memory cache so cost is negligible.
@_limiter.limit("60/minute")
async def library_full(
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
    page: int = Query(default=1, ge=1, description="1-based page number"),
    page_size: int = Query(default=200, ge=1, le=500, description="Repos per page (max 500)"),
):
    """
    Returns a paginated page of LibraryData. Aggregates (stats, categories, tagMetrics)
    are included on every page from a separate cache — they reflect the full corpus.

    ?page=1&page_size=200  → first 200 repos
    ?page=2&page_size=200  → next 200 repos

    Junction data (tags, categories, languages, etc.) is fetched only for the current
    page — memory is O(page_size), not O(total). Safe at 10K+ repos.
    """
    cache_key = f"page_{page}_{page_size}"
    redis_key = f"library:page:{page}:size:{page_size}"
    now = time.time()

    response.headers["Cache-Control"] = "public, max-age=300, stale-while-revalidate=3600"

    # 1. Check Redis cache first (shared, survives restarts)
    redis_hit = await redis_cache.get(redis_key)
    if redis_hit is not None:
        logger.info(f"Redis hit /library/full page={page} page_size={page_size}")
        # Warm in-memory cache too so subsequent requests on this instance are instant
        _cache[cache_key] = {"data": redis_hit, "expires_at": now + CACHE_TTL}
        return redis_hit

    # 2. Fall back to in-memory cache (per-instance, zero latency)
    mem_cached = _cache.get(cache_key)
    if mem_cached and mem_cached.get("expires_at", 0) > now:
        logger.info(f"Memory hit /library/full page={page} page_size={page_size}")
        return mem_cached["data"]

    t0 = time.monotonic()
    logger.info(f"Building /library/full page={page} page_size={page_size}...")

    # SECURITY: Only return public repos — is_private=false enforced inside _fetch_page_repos
    # KAN-190: Sentry's auto-instrumentation (FastApiIntegration transaction +
    # SqlalchemyIntegration per-execute spans) covers /library/full's hot
    # path without manual wrapping. Manual sentry_sdk.start_span / set_tag
    # / set_transaction_name calls in this handler triggered a deterministic
    # CI regression on Python 3.12
    # (test_library_full_excludes_private_repos_across_all_pages was missing
    # one repo from the 12-page walk). Auto spans + transaction provide
    # equivalent observability in Sentry; the structured exception handler
    # in app/main.py still covers the silent-failure gap.
    enriched_repos, total = await _fetch_page_repos(db, page=page, page_size=page_size)
    aggregates = await _fetch_aggregates(db)

    response = {
        "username": "perditioinc",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "page": page,
        "pageSize": page_size,
        "totalRepos": total,
        "totalPages": (total + page_size - 1) // page_size,
        "repos": enriched_repos,
        "gapAnalysis": build_gap_analysis(enriched_repos),
        **aggregates,
    }

    elapsed = time.monotonic() - t0
    logger.info(f"/library/full page={page} built in {elapsed:.1f}s — {len(enriched_repos)}/{total} repos")

    # Store in both caches
    _cache[cache_key] = {"data": response, "expires_at": now + CACHE_TTL}
    await redis_cache.set(redis_key, response, ttl=CACHE_TTL)
    return response


@router.get("/forks", response_model=dict)
async def list_forks(
    db: AsyncSession = Depends(get_db),
    limit: int = 100,
    offset: int = 0,
):
    """Returns fork repos for internal/intelligence use. Not displayed on reporium.com.

    SECURITY: never expose private repos. The `is_private = false` predicate is the
    same constant published in app.db_filters.PUBLIC_REPO_SQL_PREDICATE — kept inline
    here so the SQL grep audit ("rg 'FROM repos'") catches stragglers immediately.
    """
    # SECURITY: select `is_private` so the response carries the field forward
    # for the same defense-in-depth contract as /library/full. Frontend privacy
    # validators can now verify the invariant at the wire.
    result = await db.execute(text("""
        SELECT id, name, owner, forked_from, primary_language, parent_stars, parent_forks,
               readme_summary, problem_solved, behind_by, ahead_by, is_private
        FROM repos
        WHERE is_fork = true
          AND is_private = false
        ORDER BY parent_stars DESC NULLS LAST
        LIMIT :limit OFFSET :offset;
    """), {"limit": limit, "offset": offset})
    rows = result.fetchall()
    columns = result.keys()

    count_result = await db.execute(text(
        "SELECT COUNT(*) FROM repos WHERE is_fork = true AND is_private = false;"
    ))
    total = count_result.scalar()

    # Re-emit `is_private` as `isPrivate` so the response uses the same camelCase
    # contract as /library/full. The DB filter above guarantees every value is
    # `False`; emitting the field is the leak-detection signal, not the filter.
    def _shape(row_dict: dict) -> dict:
        out = dict(row_dict)
        out["isPrivate"] = bool(out.pop("is_private", False))
        return out

    return {
        "forks": [_shape(dict(zip(columns, row))) for row in rows],
        "total": total,
        "limit": limit,
        "offset": offset,
    }
