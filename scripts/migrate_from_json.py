"""
Migrates library.json from reporium lite mode into the production database.
Run once when setting up production mode. Safe to run multiple times (idempotent).

Usage:
    python scripts/migrate_from_json.py --json-path ../reporium/public/data/library.json

Options:
    --json-path     Path to library.json (required)
    --dry-run       Print what would be inserted without writing to DB
    --batch-size    Number of repos per batch (default: 50)
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import settings
from app.database import Base
from app.models.repo import (
    Repo,
    RepoAIDevSkill,
    RepoBuilder,
    RepoCategory,
    RepoCommit,
    RepoLanguage,
    RepoPMSkill,
    RepoTag,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_repo(raw: dict) -> dict:
    """
    Normalise a raw library.json repo entry into ingest-compatible shape.
    Handles both camelCase and snake_case fields.
    """
    def get(obj, *keys, default=None):
        for key in keys:
            if key in obj:
                return obj[key]
        return default

    def to_int(val, default=0):
        if isinstance(val, int):
            return val
        if isinstance(val, list):
            return len(val)
        try:
            return int(val)
        except (TypeError, ValueError):
            return default

    tags = get(raw, "tags", default=[])
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]

    categories = get(raw, "categories", default=[])
    normalized_cats = []
    for cat in categories:
        if isinstance(cat, str):
            normalized_cats.append({
                "category_id": cat.lower().replace(" ", "-"),
                "category_name": cat,
                "is_primary": False,
            })
        elif isinstance(cat, dict):
            normalized_cats.append({
                "category_id": get(cat, "id", "category_id", default=""),
                "category_name": get(cat, "name", "category_name", default=""),
                "is_primary": get(cat, "is_primary", "isPrimary", default=False),
            })

    builders = get(raw, "builders", default=[])
    normalized_builders = []
    for b in builders:
        if isinstance(b, dict):
            normalized_builders.append({
                "login": get(b, "login", default="unknown"),
                "display_name": get(b, "displayName", "display_name"),
                "org_category": get(b, "orgCategory", "org_category"),
                "is_known_org": get(b, "isKnownOrg", "is_known_org", default=False),
            })

    languages = get(raw, "languages", default=[])
    normalized_langs = []
    for lang in languages:
        if isinstance(lang, dict):
            normalized_langs.append({
                "language": get(lang, "language", "name", default=""),
                "bytes": get(lang, "bytes", default=0),
                "percentage": get(lang, "percentage", default=0.0),
            })

    return {
        "name": get(raw, "name", default=""),
        "owner": get(raw, "owner", default=""),
        "description": get(raw, "description"),
        "is_fork": get(raw, "is_fork", "isFork", default=False),
        "forked_from": get(raw, "forked_from", "forkedFrom"),
        "primary_language": get(raw, "primary_language", "primaryLanguage"),
        "github_url": get(raw, "github_url", "githubUrl", "url", default=""),
        "fork_sync_state": get(raw, "fork_sync_state", "forkSyncState"),
        "behind_by": get(raw, "behind_by", "behindBy", default=0),
        "ahead_by": get(raw, "ahead_by", "aheadBy", default=0),
        "parent_stars": get(raw, "parent_stars", "parentStars"),
        "parent_forks": get(raw, "parent_forks", "parentForks"),
        "parent_is_archived": get(raw, "parent_is_archived", "parentIsArchived", default=False),
        "commits_last_7_days": to_int(get(raw, "commits_last_7_days", "commitsLast7Days")),
        "commits_last_30_days": to_int(get(raw, "commits_last_30_days", "commitsLast30Days")),
        "commits_last_90_days": to_int(get(raw, "commits_last_90_days", "commitsLast90Days")),
        "readme_summary": get(raw, "readme_summary", "readmeSummary"),
        "activity_score": get(raw, "activity_score", "activityScore", default=0),
        "tags": tags,
        "categories": normalized_cats,
        "builders": normalized_builders,
        "ai_dev_skills": get(raw, "ai_dev_skills", "aiDevSkills", default=[]),
        "pm_skills": get(raw, "pm_skills", "pmSkills", default=[]),
        "languages": normalized_langs,
        "commits": [],
    }


async def migrate(json_path: Path, dry_run: bool, batch_size: int) -> None:
    logger.info(f"Loading {json_path}")
    data = json.loads(json_path.read_text(encoding="utf-8"))

    # library.json may be { repos: [...] } or a plain array
    if isinstance(data, dict):
        raw_repos = data.get("repos", [])
    elif isinstance(data, list):
        raw_repos = data
    else:
        logger.error("Unexpected library.json format")
        sys.exit(1)

    logger.info(f"Found {len(raw_repos)} repos")

    if dry_run:
        for raw in raw_repos[:3]:
            repo = parse_repo(raw)
            logger.info(f"  [DRY RUN] Would upsert: {repo['name']}")
        logger.info("Dry run complete — no changes written.")
        return

    engine = create_async_engine(settings.database_url, echo=False)
    SessionFactory = async_sessionmaker(engine, expire_on_commit=False)

    upserted = 0
    errors = []

    from sqlalchemy import select

    for i in range(0, len(raw_repos), batch_size):
        batch = raw_repos[i: i + batch_size]
        batch_upserted = 0
        for raw in batch:
            async with SessionFactory() as session:
                try:
                    repo_data = parse_repo(raw)
                    if not repo_data["name"]:
                        logger.warning(f"Skipping repo with no name: {raw}")
                        continue

                    stmt = select(Repo).where(Repo.name == repo_data["name"])
                    result = await session.execute(stmt)
                    repo = result.scalar_one_or_none()

                    fields = {k: v for k, v in repo_data.items()
                              if k not in {"tags", "categories", "builders", "ai_dev_skills",
                                           "pm_skills", "languages", "commits"}}

                    if repo is None:
                        repo = Repo(**fields)
                        session.add(repo)
                    else:
                        for key, val in fields.items():
                            setattr(repo, key, val)

                    await session.flush()

                    # Replace child rows
                    for model in (RepoTag, RepoCategory, RepoBuilder, RepoAIDevSkill,
                                  RepoPMSkill, RepoLanguage):
                        await session.execute(model.__table__.delete().where(model.repo_id == repo.id))

                    for tag in repo_data["tags"]:
                        session.add(RepoTag(repo_id=repo.id, tag=tag))
                    for cat in repo_data["categories"]:
                        session.add(RepoCategory(repo_id=repo.id, **cat))
                    for builder in repo_data["builders"]:
                        session.add(RepoBuilder(repo_id=repo.id, **builder))
                    for skill in repo_data["ai_dev_skills"]:
                        session.add(RepoAIDevSkill(repo_id=repo.id, skill=skill))
                    for skill in repo_data["pm_skills"]:
                        session.add(RepoPMSkill(repo_id=repo.id, skill=skill))
                    for lang in repo_data["languages"]:
                        session.add(RepoLanguage(repo_id=repo.id, **lang))

                    await session.commit()
                    upserted += 1
                    batch_upserted += 1

                except Exception as e:
                    await session.rollback()
                    errors.append(f"{raw.get('name', '?')}: {e}")
                    logger.error(f"Failed to migrate repo '{raw.get('name', '?')}': {e}")

        logger.info(f"Batch {i // batch_size + 1}: {batch_upserted}/{len(batch)} upserted")

    await engine.dispose()

    logger.info(f"\nMigration complete: {upserted} upserted, {len(errors)} errors")
    if errors:
        logger.warning("Errors:")
        for err in errors:
            logger.warning(f"  {err}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate library.json to reporium-api database")
    parser.add_argument("--json-path", required=True, help="Path to library.json")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument("--batch-size", type=int, default=50, help="Repos per DB batch")
    args = parser.parse_args()

    json_path = Path(args.json_path)
    if not json_path.exists():
        logger.error(f"File not found: {json_path}")
        sys.exit(1)

    asyncio.run(migrate(json_path, args.dry_run, args.batch_size))


if __name__ == "__main__":
    main()
