"""
GET /library/full — Returns the complete dataset in the exact shape
the reporium.com frontend expects (LibraryData TypeScript interface).

All fields camelCase, nested objects, all repos in one response.
Cached for 5 minutes to avoid repeated expensive queries.
"""

import logging
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db

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
}


def _normalize_category(name: str) -> str:
    """Map raw DB category name to the frontend's canonical name."""
    return CATEGORY_MAP.get(name, name)

router = APIRouter(tags=["Library"])

# In-memory cache
_cache: dict = {"data": None, "expires_at": 0}
CACHE_TTL = 300  # 5 minutes


def _iso(val) -> str:
    """Convert a datetime or string to ISO format string."""
    if val is None:
        return ""
    if isinstance(val, datetime):
        return val.isoformat()
    return str(val)


def _build_enriched_repo(repo: dict, languages: list, categories: list,
                         ai_skills: list, tags: list, pm_skills: list,
                         builders: list = None, industries: list = None) -> dict:
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

    # Fork sync status
    fork_sync = None
    if repo.get("is_fork"):
        behind = repo.get("behind_by") or 0
        ahead = repo.get("ahead_by") or 0
        if behind == 0 and ahead == 0:
            state = "up-to-date"
        elif behind > 0 and ahead > 0:
            state = "diverged"
        elif behind > 0:
            state = "behind"
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

    all_cats = list(dict.fromkeys(_normalize_category(c["category_name"]) for c in categories))
    primary_cat = all_cats[0] if all_cats else "Dev Tools & Automation"

    return {
        "id": hash(str(repo.get("id"))) & 0x7FFFFFFF,  # Convert UUID to positive int
        "name": name,
        "fullName": f"{owner}/{name}",
        "description": repo.get("description"),
        "isFork": repo.get("is_fork", False),
        "forkedFrom": forked_from,
        "language": repo.get("primary_language"),
        "topics": [t["tag"] for t in tags],
        "enrichedTags": [s["skill"] for s in ai_skills] + [t["tag"] for t in tags],
        "stars": repo.get("parent_stars") or 0,
        "forks": repo.get("parent_forks") or 0,
        "lastUpdated": _iso(repo.get("github_updated_at") or repo.get("updated_at")),
        "url": repo.get("github_url") or f"https://github.com/{owner}/{name}",
        "isArchived": repo.get("parent_is_archived") or False,
        "readmeSummary": repo.get("readme_summary"),
        "parentStats": parent_stats,
        "recentCommits": [],
        "createdAt": _iso(repo.get("upstream_created_at") or repo.get("ingested_at")),
        "forkedAt": _iso(repo.get("forked_at")),
        "yourLastPushAt": _iso(repo.get("your_last_push_at")),
        "upstreamLastPushAt": _iso(repo.get("upstream_last_push_at")),
        "upstreamCreatedAt": _iso(repo.get("upstream_created_at")),
        "forkSync": fork_sync,
        "weeklyCommitCount": c7,
        "languageBreakdown": lang_breakdown,
        "languagePercentages": lang_percentages,
        "commitsLast7Days": [],
        "commitsLast30Days": [],
        "commitsLast90Days": [],
        "totalCommitsFetched": 0,
        "primaryCategory": primary_cat,
        "allCategories": all_cats,
        "commitStats": {
            "today": 0,
            "last7Days": c7,
            "last30Days": c30,
            "last90Days": c90,
            "recentCommits": [],
        },
        "latestRelease": None,
        "aiDevSkills": [s["skill"] for s in ai_skills],
        "pmSkills": [s["skill"] for s in pm_skills],
        "industries": [ind["industry"] for ind in (industries or [])],
        "programmingLanguages": list(lang_breakdown.keys()),
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


def _build_stats(repos: list) -> dict:
    """Build LibraryStats from enriched repos."""
    languages = set()
    tag_counter = Counter()
    built = 0
    forked = 0

    for r in repos:
        if r["isFork"]:
            forked += 1
        else:
            built += 1
        if r["language"]:
            languages.add(r["language"])
        for t in r["enrichedTags"]:
            tag_counter[t] += 1

    return {
        "total": len(repos),
        "built": built,
        "forked": forked,
        "languages": sorted(languages),
        "topTags": [t for t, _ in tag_counter.most_common(20)],
    }


def _build_tag_metrics(repos: list) -> list:
    """Build TagMetrics[] from enriched repos."""
    tag_repos = defaultdict(list)
    for r in repos:
        for t in r["enrichedTags"]:
            tag_repos[t].append(r)

    metrics = []
    total = len(repos) if repos else 1
    for tag, tag_repo_list in sorted(tag_repos.items()):
        lang_counter = Counter()
        for r in tag_repo_list:
            if r["language"]:
                lang_counter[r["language"]] += 1

        metrics.append({
            "tag": tag,
            "repoCount": len(tag_repo_list),
            "percentage": round(len(tag_repo_list) / total * 100, 1),
            "topLanguage": lang_counter.most_common(1)[0][0] if lang_counter else None,
            "languageBreakdown": dict(lang_counter),
            "updatedLast30Days": 0,
            "updatedLast90Days": 0,
            "olderThan90Days": 0,
            "activityScore": 0,
            "relatedTags": [],
            "mostRecentRepo": tag_repo_list[0]["name"] if tag_repo_list else "",
            "mostRecentDate": tag_repo_list[0]["lastUpdated"] if tag_repo_list else "",
            "repos": [r["name"] for r in tag_repo_list[:20]],
            "avgUpstreamAge": 0,
            "avgTimeSinceForked": 0,
            "mostOutdatedRepo": "",
            "avgBehindBy": 0,
        })

    return metrics


def _build_categories(repos: list) -> list:
    """Build Category[] from enriched repos."""
    cat_repos = defaultdict(list)
    for r in repos:
        for c in r["allCategories"]:
            cat_repos[c].append(r)

    COLORS = {
        "Agents": "#7c3aed", "Audio": "#ec4899", "Code Generation": "#3b82f6",
        "Datasets": "#10b981", "Deployment": "#f59e0b", "Evaluation": "#6366f1",
        "Fine Tuning": "#ef4444", "Llm Serving": "#8b5cf6", "Observability": "#14b8a6",
        "Ocr": "#f97316", "Research": "#06b6d4", "Security": "#64748b",
        "Tooling": "#84cc16", "Vector Databases": "#a855f7", "Other": "#94a3b8",
    }
    ICONS = {
        "Agents": "🤖", "Audio": "🎵", "Code Generation": "💻",
        "Datasets": "📊", "Deployment": "🚀", "Evaluation": "📏",
        "Fine Tuning": "🔧", "Llm Serving": "⚡", "Observability": "👁",
        "Ocr": "📝", "Research": "🔬", "Security": "🔒",
        "Tooling": "🛠", "Vector Databases": "🗄", "Other": "📦",
    }

    categories = []
    for cat, cat_repo_list in sorted(cat_repos.items()):
        # Collect tags in this category
        cat_tags = set()
        for r in cat_repo_list:
            cat_tags.update(r["enrichedTags"])

        categories.append({
            "id": cat.lower().replace(" ", "-"),
            "name": cat,
            "description": f"Repos related to {cat.lower()}",
            "tags": sorted(cat_tags),
            "repoCount": len(cat_repo_list),
            "color": COLORS.get(cat, "#94a3b8"),
            "icon": ICONS.get(cat, "📦"),
        })

    return categories


def _build_skill_stats(repos: list, skill_field: str) -> list:
    """Build SkillStats[] from enriched repos."""
    skill_repos = defaultdict(list)
    for r in repos:
        for s in r.get(skill_field, []):
            skill_repos[s].append(r)

    total = len(repos) if repos else 1
    stats = []
    for skill, skill_repo_list in sorted(skill_repos.items()):
        count = len(skill_repo_list)
        pct = count / total
        if pct >= 0.1:
            coverage = "strong"
        elif pct >= 0.05:
            coverage = "moderate"
        elif pct >= 0.01:
            coverage = "weak"
        else:
            coverage = "none"

        stats.append({
            "skill": skill,
            "repoCount": count,
            "coverage": coverage,
            "topRepos": [r["name"] for r in sorted(
                skill_repo_list, key=lambda x: x["stars"], reverse=True
            )[:5]],
        })

    return stats


def _build_builder_stats(repos: list) -> list:
    """Build BuilderStats from enriched repos."""
    builder_data = defaultdict(lambda: {
        "repoCount": 0, "totalParentStars": 0, "topRepos": [],
        "category": "individual", "avatarUrl": ""
    })
    for r in repos:
        for b in r.get("builders", []):
            login = b["login"]
            bd = builder_data[login]
            bd["repoCount"] += 1
            bd["totalParentStars"] += r.get("stars", 0)
            bd["topRepos"].append(r["name"])
            bd["category"] = b.get("orgCategory") or "individual"
            bd["avatarUrl"] = b.get("avatarUrl", "")
    stats = []
    for login, bd in sorted(builder_data.items(), key=lambda x: x[1]["totalParentStars"], reverse=True):
        stats.append({
            "login": login,
            "displayName": login,
            "category": bd["category"],
            "repoCount": bd["repoCount"],
            "totalParentStars": bd["totalParentStars"],
            "topRepos": bd["topRepos"][:5],
            "avatarUrl": bd["avatarUrl"],
        })
    return stats[:50]  # Top 50 builders


@router.get("/library/full")
async def library_full(db: AsyncSession = Depends(get_db)):
    """
    Returns the complete LibraryData response matching the frontend TypeScript interface.
    Cached for 5 minutes.
    """
    now = time.time()
    if _cache["data"] and _cache["expires_at"] > now:
        logger.info("Returning cached /library/full response")
        return _cache["data"]

    t0 = time.monotonic()
    logger.info("Building /library/full response...")

    # Fetch all repos
    result = await db.execute(text("""
        SELECT id, name, owner, description, is_fork, forked_from, primary_language,
               github_url, fork_sync_state, behind_by, ahead_by,
               upstream_created_at, forked_at, your_last_push_at, upstream_last_push_at,
               parent_stars, parent_forks, parent_is_archived,
               commits_last_7_days, commits_last_30_days, commits_last_90_days,
               readme_summary, activity_score, ingested_at, updated_at, github_updated_at,
               problem_solved, integration_tags, dependencies
        FROM repos
        ORDER BY parent_stars DESC NULLS LAST;
    """))
    rows = result.fetchall()
    columns = result.keys()

    # Fetch all junction data in bulk
    lang_result = await db.execute(text(
        "SELECT repo_id, language, bytes, percentage FROM repo_languages;"
    ))
    all_languages = defaultdict(list)
    for r in lang_result.fetchall():
        all_languages[str(r.repo_id)].append({
            "language": r.language, "bytes": r.bytes, "percentage": r.percentage
        })

    cat_result = await db.execute(text(
        "SELECT repo_id, category_name, is_primary FROM repo_categories;"
    ))
    all_categories = defaultdict(list)
    for r in cat_result.fetchall():
        all_categories[str(r.repo_id)].append({
            "category_name": r.category_name, "is_primary": r.is_primary
        })

    skill_result = await db.execute(text(
        "SELECT repo_id, skill FROM repo_ai_dev_skills;"
    ))
    all_ai_skills = defaultdict(list)
    for r in skill_result.fetchall():
        all_ai_skills[str(r.repo_id)].append({"skill": r.skill})

    tag_result = await db.execute(text(
        "SELECT repo_id, tag FROM repo_tags;"
    ))
    all_tags = defaultdict(list)
    for r in tag_result.fetchall():
        all_tags[str(r.repo_id)].append({"tag": r.tag})

    pm_result = await db.execute(text(
        "SELECT repo_id, skill FROM repo_pm_skills;"
    ))
    all_pm_skills = defaultdict(list)
    for r in pm_result.fetchall():
        all_pm_skills[str(r.repo_id)].append({"skill": r.skill})

    builder_result = await db.execute(text(
        "SELECT repo_id, login, display_name, org_category, is_known_org FROM repo_builders;"
    ))
    all_builders = defaultdict(list)
    for r in builder_result.fetchall():
        all_builders[str(r.repo_id)].append({
            "login": r.login, "display_name": r.display_name,
            "org_category": r.org_category, "is_known_org": r.is_known_org
        })

    industry_result = await db.execute(text(
        "SELECT repo_id, industry FROM repo_industries;"
    ))
    all_industries = defaultdict(list)
    for r in industry_result.fetchall():
        all_industries[str(r.repo_id)].append({"industry": r.industry})

    # Build enriched repos
    enriched_repos = []
    for row in rows:
        repo = dict(zip(columns, row))
        rid = str(repo["id"])
        enriched = _build_enriched_repo(
            repo,
            languages=all_languages.get(rid, []),
            categories=all_categories.get(rid, []),
            ai_skills=all_ai_skills.get(rid, []),
            tags=all_tags.get(rid, []),
            pm_skills=all_pm_skills.get(rid, []),
            builders=all_builders.get(rid, []),
            industries=all_industries.get(rid, []),
        )
        enriched_repos.append(enriched)

    # Build aggregated data
    stats = _build_stats(enriched_repos)
    tag_metrics = _build_tag_metrics(enriched_repos)
    categories = _build_categories(enriched_repos)
    ai_skill_stats = _build_skill_stats(enriched_repos, "aiDevSkills")
    pm_skill_stats = _build_skill_stats(enriched_repos, "pmSkills")

    response = {
        "username": "perditioinc",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "stats": stats,
        "repos": enriched_repos,
        "tagMetrics": tag_metrics,
        "categories": categories,
        "gapAnalysis": {"generatedAt": datetime.now(timezone.utc).isoformat(), "gaps": []},
        "builderStats": _build_builder_stats(enriched_repos),
        "aiDevSkillStats": ai_skill_stats,
        "pmSkillStats": pm_skill_stats,
    }

    elapsed = time.monotonic() - t0
    logger.info(f"/library/full built in {elapsed:.1f}s — {len(enriched_repos)} repos")

    # Cache
    _cache["data"] = response
    _cache["expires_at"] = now + CACHE_TTL

    return response
