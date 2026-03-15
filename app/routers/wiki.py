from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.models.repo import Repo, RepoAIDevSkill, RepoCategory, RepoPMSkill
from app.routers.library import _repo_to_summary
from app.schemas.repo import RepoSummary

router = APIRouter()


class SkillWikiResponse(BaseModel):
    skill: str
    repo_count: int
    repos: list[RepoSummary]
    related_skills: list[str]


class CategoryWikiResponse(BaseModel):
    category_id: str
    category_name: str
    repo_count: int
    repos: list[RepoSummary]
    top_tags: list[str]
    top_languages: list[str]


@router.get("/wiki/skills/{skill}", response_model=SkillWikiResponse)
async def get_skill_wiki(skill: str, db: AsyncSession = Depends(get_db)) -> SkillWikiResponse:
    stmt = (
        select(Repo)
        .join(RepoAIDevSkill, RepoAIDevSkill.repo_id == Repo.id)
        .where(RepoAIDevSkill.skill == skill)
        .options(
            selectinload(Repo.tags),
            selectinload(Repo.categories),
            selectinload(Repo.builders),
            selectinload(Repo.ai_dev_skills),
            selectinload(Repo.pm_skills),
            selectinload(Repo.languages),
        )
        .order_by(Repo.activity_score.desc())
        .limit(20)
    )
    result = await db.execute(stmt)
    repos = result.scalars().all()

    if not repos:
        # Also check PM skills
        stmt2 = (
            select(Repo)
            .join(RepoPMSkill, RepoPMSkill.repo_id == Repo.id)
            .where(RepoPMSkill.skill == skill)
            .options(
                selectinload(Repo.tags),
                selectinload(Repo.categories),
                selectinload(Repo.builders),
                selectinload(Repo.ai_dev_skills),
                selectinload(Repo.pm_skills),
                selectinload(Repo.languages),
            )
            .order_by(Repo.activity_score.desc())
            .limit(20)
        )
        result2 = await db.execute(stmt2)
        repos = result2.scalars().all()

    if not repos:
        raise HTTPException(status_code=404, detail=f"No repos found for skill '{skill}'")

    # Related skills (from same repos)
    related: dict[str, int] = {}
    for repo in repos:
        for s in repo.ai_dev_skills:
            if s.skill != skill:
                related[s.skill] = related.get(s.skill, 0) + 1
        for s in repo.pm_skills:
            if s.skill != skill:
                related[s.skill] = related.get(s.skill, 0) + 1

    related_skills = sorted(related, key=lambda x: related[x], reverse=True)[:10]

    return SkillWikiResponse(
        skill=skill,
        repo_count=len(repos),
        repos=[_repo_to_summary(r) for r in repos],
        related_skills=related_skills,
    )


@router.get("/wiki/categories/{category}", response_model=CategoryWikiResponse)
async def get_category_wiki(
    category: str, db: AsyncSession = Depends(get_db)
) -> CategoryWikiResponse:
    stmt = (
        select(Repo)
        .join(RepoCategory, RepoCategory.repo_id == Repo.id)
        .where(RepoCategory.category_id == category)
        .options(
            selectinload(Repo.tags),
            selectinload(Repo.categories),
            selectinload(Repo.builders),
            selectinload(Repo.ai_dev_skills),
            selectinload(Repo.pm_skills),
            selectinload(Repo.languages),
        )
        .order_by(Repo.activity_score.desc())
        .limit(50)
    )
    result = await db.execute(stmt)
    repos = result.scalars().all()

    if not repos:
        raise HTTPException(status_code=404, detail=f"No repos found for category '{category}'")

    category_name = repos[0].categories[0].category_name if repos[0].categories else category

    # Top tags and languages
    tag_counts: dict[str, int] = {}
    lang_counts: dict[str, int] = {}
    for repo in repos:
        for t in repo.tags:
            tag_counts[t.tag] = tag_counts.get(t.tag, 0) + 1
        if repo.primary_language:
            lang_counts[repo.primary_language] = lang_counts.get(repo.primary_language, 0) + 1

    top_tags = sorted(tag_counts, key=lambda x: tag_counts[x], reverse=True)[:10]
    top_languages = sorted(lang_counts, key=lambda x: lang_counts[x], reverse=True)[:5]

    return CategoryWikiResponse(
        category_id=category,
        category_name=category_name,
        repo_count=len(repos),
        repos=[_repo_to_summary(r) for r in repos],
        top_tags=top_tags,
        top_languages=top_languages,
    )
