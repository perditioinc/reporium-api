"""
GET /taxonomy/skill-areas — Returns all skill areas from DB, grouped by lifecycle_group.
GET /taxonomy/skill-areas/{name}/repos — Returns repos tagged with a given skill area.

Skill areas are stored in the skill_areas table; adding a new area requires only a DB insert.
"""

import logging
import time
from collections import defaultdict

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.repo import SkillArea

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Taxonomy"])

# In-memory cache for skill-areas list (5 min TTL)
_taxonomy_cache: dict = {}
_TAXONOMY_CACHE_TTL = 300  # 5 minutes


@router.get("/skill-areas")
async def list_skill_areas(db: AsyncSession = Depends(get_db)) -> dict:
    """
    Return all skill areas from the DB, grouped by lifecycle_group.
    Each skill area includes a repoCount (repos in repo_ai_dev_skills).
    Skill areas with repoCount < min_repos_to_display are excluded.
    """
    now = time.monotonic()
    cached = _taxonomy_cache.get("skill_areas")
    if cached and cached.get("expires_at", 0) > now:
        return cached["data"]

    # Fetch all skill areas ordered by sort_order
    stmt = select(SkillArea).order_by(SkillArea.sort_order, SkillArea.name)
    result = await db.execute(stmt)
    skill_areas = result.scalars().all()

    # Count repos per skill area from repo_ai_dev_skills
    count_stmt = text(
        "SELECT skill, COUNT(DISTINCT repo_id) AS repo_count "
        "FROM repo_ai_dev_skills "
        "GROUP BY skill"
    )
    count_result = await db.execute(count_stmt)
    repo_counts: dict[str, int] = {row.skill: row.repo_count for row in count_result}

    grouped: dict[str, list] = defaultdict(list)
    for sa in skill_areas:
        count = repo_counts.get(sa.name, 0)
        if count < sa.min_repos_to_display:
            continue
        grouped[sa.lifecycle_group].append({
            "id": sa.id,
            "name": sa.name,
            "lifecycleGroup": sa.lifecycle_group,
            "description": sa.description,
            "icon": sa.icon,
            "color": sa.color,
            "sortOrder": sa.sort_order,
            "repoCount": count,
        })

    response = {
        "groups": [
            {
                "lifecycleGroup": group,
                "skillAreas": areas,
            }
            for group, areas in grouped.items()
        ],
        "total": sum(len(areas) for areas in grouped.values()),
    }

    _taxonomy_cache["skill_areas"] = {"data": response, "expires_at": now + _TAXONOMY_CACHE_TTL}
    return response


@router.get("/skill-areas/{name}/repos")
async def get_repos_for_skill_area(name: str, db: AsyncSession = Depends(get_db)) -> dict:
    """
    Return repos that have this skill area (via repo_ai_dev_skills join).
    Validates that the skill area exists in the skill_areas table.
    """
    # Verify skill area exists
    sa_stmt = select(SkillArea).where(SkillArea.name == name)
    sa_result = await db.execute(sa_stmt)
    skill_area = sa_result.scalar_one_or_none()
    if skill_area is None:
        raise HTTPException(status_code=404, detail=f"Skill area '{name}' not found")

    # Fetch repos with this skill
    repos_stmt = text(
        "SELECT r.name, r.owner, r.description, r.github_url, r.primary_language, "
        "       r.stargazers_count, r.parent_stars, r.activity_score, r.readme_summary "
        "FROM repos r "
        "JOIN repo_ai_dev_skills s ON s.repo_id = r.id "
        "WHERE s.skill = :skill "
        "ORDER BY COALESCE(r.stargazers_count, r.parent_stars, 0) DESC"
    )
    repos_result = await db.execute(repos_stmt, {"skill": name})
    rows = repos_result.mappings().all()

    repos = [
        {
            "name": row["name"],
            "owner": row["owner"],
            "description": row["description"],
            "githubUrl": row["github_url"],
            "primaryLanguage": row["primary_language"],
            "stars": row["stargazers_count"] or row["parent_stars"] or 0,
            "activityScore": row["activity_score"],
            "readmeSummary": row["readme_summary"],
        }
        for row in rows
    ]

    return {
        "skillArea": {
            "id": skill_area.id,
            "name": skill_area.name,
            "lifecycleGroup": skill_area.lifecycle_group,
            "description": skill_area.description,
            "icon": skill_area.icon,
            "color": skill_area.color,
        },
        "repos": repos,
        "repoCount": len(repos),
    }
