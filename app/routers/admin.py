from fastapi import APIRouter, Depends, Query
from sqlalchemy import delete, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import require_admin_key, verify_api_key
from app.cache import cache
from app.database import get_db
from app.models.repo import Repo, RepoTag
from app.routers.library_full import invalidate_library_cache

router = APIRouter()

# Canonical noise-tag list from Reporium taxonomy Phase 4 cleanup rules.
NOISE_TAGS = frozenset({
    "python", "javascript", "typescript", "rust", "go", "java", "c++",
    "react", "nextjs", "nodejs", "express", "fastapi", "flask", "django",
    "postgresql", "mysql", "mongodb", "redis", "docker", "kubernetes",
    "aws", "gcp", "azure", "terraform", "nginx", "linux", "macos", "windows",
    "git", "github", "api", "rest", "graphql", "grpc", "websocket", "cli",
    "sdk", "library", "framework", "tutorial", "example", "demo", "template",
    "boilerplate", "starter", "awesome", "list", "collection", "open-source",
    "free", "fast", "simple", "easy", "lightweight", "minimal",
})


async def _prune_noise_tags(db: AsyncSession, *, dry_run: bool) -> dict:
    tag_counts_result = await db.execute(
        select(
            func.lower(RepoTag.tag).label("tag"),
            func.count().label("count"),
        )
        .where(func.lower(RepoTag.tag).in_(NOISE_TAGS))
        .group_by(func.lower(RepoTag.tag))
        .order_by(func.count().desc(), func.lower(RepoTag.tag))
    )
    matched_tags = {row.tag: row.count for row in tag_counts_result.fetchall()}
    matched_rows = sum(matched_tags.values())

    deleted_rows = 0
    if not dry_run and matched_rows:
        delete_result = await db.execute(
            delete(RepoTag).where(func.lower(RepoTag.tag).in_(NOISE_TAGS))
        )
        deleted_rows = delete_result.rowcount or 0
        await db.commit()

        await cache.invalidate("library:full*")
        await cache.invalidate("repos:list:*")
        invalidate_library_cache()

    return {
        "dry_run": dry_run,
        "matched_rows": matched_rows,
        "matched_tag_count": len(matched_tags),
        "deleted_rows": 0 if dry_run else deleted_rows,
        "matched_tags": matched_tags,
    }


@router.get("/admin/data-quality")
async def data_quality(
    db: AsyncSession = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
    _admin_key: None = Depends(require_admin_key),
):

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


@router.post("/admin/tags/prune")
async def prune_tags(
    dry_run: bool = Query(default=False),
    db: AsyncSession = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
    _admin_key: None = Depends(require_admin_key),
):
    return await _prune_noise_tags(db, dry_run=dry_run)


@router.post("/admin/quality/compute")
async def compute_quality_signals(
    db: AsyncSession = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
    _admin_key: None = Depends(require_admin_key),
):
    """Compute quality_signals for all repos from existing data (no GitHub API calls)."""
    BATCH_SIZE = 100
    offset = 0
    computed = 0
    skipped = 0

    while True:
        stmt = select(Repo).offset(offset).limit(BATCH_SIZE)
        result = await db.execute(stmt)
        repos = result.scalars().all()

        if not repos:
            break

        for repo in repos:
            try:
                commit_velocity_30d = repo.commits_last_30_days / 30.0
                commit_velocity_7d = repo.commits_last_7_days / 7.0
                is_active = repo.commits_last_30_days > 0
                has_open_issues = repo.open_issues_count > 0

                activity = repo.activity_score  # 0-100

                weekly_score = (min(repo.commits_last_7_days, 10) / 10.0) * 100
                if repo.open_issues_count < 10:
                    issues_score = 100
                else:
                    issues_score = max(0, 100 - repo.open_issues_count * 2)

                overall_raw = (
                    activity * 0.5
                    + weekly_score * 0.3
                    + issues_score * 0.2
                )
                overall_score = max(0, min(100, round(overall_raw)))

                repo.quality_signals = {
                    "commit_velocity_30d": commit_velocity_30d,
                    "commit_velocity_7d": commit_velocity_7d,
                    "is_active": is_active,
                    "has_open_issues": has_open_issues,
                    "activity_score": activity,
                    "overall_score": overall_score,
                }
                computed += 1
            except Exception:
                skipped += 1

        await db.commit()
        offset += BATCH_SIZE

    return {"computed": computed, "skipped": skipped}
