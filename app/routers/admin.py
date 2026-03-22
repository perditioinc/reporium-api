import os

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db

router = APIRouter()


@router.get("/admin/data-quality")
async def data_quality(request: Request, db: AsyncSession = Depends(get_db)):
    # Require API key
    api_key = request.headers.get("X-API-Key")
    if api_key != os.getenv("INGESTION_API_KEY"):
        raise HTTPException(status_code=403, detail="Invalid API key")

    # Query counts
    total = (await db.execute(text("SELECT COUNT(*) FROM repos;"))).scalar()
    owned = (await db.execute(text("SELECT COUNT(*) FROM repos WHERE is_fork = false;"))).scalar()
    forks = (await db.execute(text("SELECT COUNT(*) FROM repos WHERE is_fork = true;"))).scalar()
    missing_summary = (await db.execute(text("SELECT COUNT(*) FROM repos WHERE readme_summary IS NULL OR readme_summary = '';"))).scalar()
    missing_desc = (await db.execute(text("SELECT COUNT(*) FROM repos WHERE description IS NULL OR description = '';"))).scalar()
    missing_cats = (await db.execute(text("SELECT COUNT(*) FROM repos r WHERE r.id NOT IN (SELECT DISTINCT repo_id FROM repo_categories);"))).scalar()
    missing_builders = (await db.execute(text("SELECT COUNT(*) FROM repos r WHERE r.id NOT IN (SELECT DISTINCT repo_id FROM repo_builders);"))).scalar()
    missing_embeddings = (await db.execute(text("SELECT COUNT(*) FROM repos r WHERE r.id NOT IN (SELECT DISTINCT repo_id FROM repo_embeddings);"))).scalar()

    # Category distribution
    cat_result = await db.execute(text("""
        SELECT category_name, COUNT(DISTINCT repo_id) as cnt
        FROM repo_categories WHERE is_primary = true
        GROUP BY category_name ORDER BY cnt DESC;
    """))
    cat_dist = {r[0]: r[1] for r in cat_result.fetchall()}
    max_cat_pct = (max(cat_dist.values()) / total * 100) if cat_dist and total > 0 else 0

    # Quality score: 100 minus penalties
    score = 100
    if missing_summary > 0:
        score -= min(20, missing_summary)
    if missing_desc > 0:
        score -= min(10, missing_desc)
    if missing_cats > 0:
        score -= min(10, missing_cats)
    if missing_builders > 0:
        score -= min(10, missing_builders)
    if missing_embeddings > 0:
        score -= min(15, missing_embeddings)
    if max_cat_pct > 25:
        score -= min(10, int(max_cat_pct - 25))

    return {
        "total_repos": total,
        "owned_repos": owned,
        "fork_repos": forks,
        "missing_summary": missing_summary,
        "missing_description": missing_desc,
        "missing_categories": missing_cats,
        "missing_builders": missing_builders,
        "missing_embeddings": missing_embeddings,
        "category_distribution": cat_dist,
        "max_category_percent": round(max_cat_pct, 1),
        "quality_score": max(0, score),
    }
