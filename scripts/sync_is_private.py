#!/usr/bin/env python3
"""
Sync is_private flag from GitHub for all repos in the database.

Run this after any ingestion batch to ensure is_private is accurate.
The ingestion pipeline may not always pass the correct is_private value,
so this script is the authoritative source of truth sync.

Usage (local):
  DATABASE_URL=postgresql+asyncpg://... GITHUB_TOKEN=ghp_... python scripts/sync_is_private.py

Usage (production — secrets fetched from GCP Secret Manager automatically):
  ENVIRONMENT=production GCP_PROJECT=perditio-platform python scripts/sync_is_private.py

The script:
  1. Fetches all repos from the database.
  2. Calls GitHub API to get isPrivate for each (batched via GraphQL).
  3. Updates is_private in the DB for any mismatches.
  4. Prints a summary of changes made.
"""

import asyncio
import os
import sys
from typing import Optional

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

OWNER = "perditioinc"
GH_GRAPHQL_URL = "https://api.github.com/graphql"
BATCH_SIZE = 50  # GitHub GraphQL allows ~100 aliases per query safely


def _get_gcp_secret(secret_id: str, project_id: str) -> str:
    from google.cloud import secretmanager
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
    return client.access_secret_version(request={"name": name}).payload.data.decode("UTF-8")


def resolve_config() -> tuple[str, str]:
    """Return (database_url, github_token)."""
    db_url = os.environ.get("DATABASE_URL", "")
    gh_token = os.environ.get("GITHUB_TOKEN", "")

    if os.environ.get("ENVIRONMENT") == "production" or not (db_url and gh_token):
        project = os.environ.get("GCP_PROJECT", "perditio-platform")
        if not db_url:
            print("Loading DATABASE_URL from Secret Manager...")
            db_url = _get_gcp_secret("reporium-db-url-async", project)
        if not gh_token:
            print("Loading GITHUB_TOKEN from Secret Manager...")
            gh_token = _get_gcp_secret("github-token", project)

    if not db_url:
        print("ERROR: DATABASE_URL not set and not available in Secret Manager")
        sys.exit(1)
    if not gh_token:
        print("ERROR: GITHUB_TOKEN not set and not available in Secret Manager")
        sys.exit(1)

    return db_url, gh_token


def build_graphql_query(names: list[str]) -> str:
    """Build a GraphQL query that fetches isPrivate for a batch of repo names."""
    aliases = "\n".join(
        f'  r{i}: repository(owner: "{OWNER}", name: "{name}") {{ isPrivate }}'
        for i, name in enumerate(names)
    )
    return f"{{ {aliases} }}"


async def fetch_github_privacy(names: list[str], token: str) -> dict[str, Optional[bool]]:
    """
    Returns {name: is_private} for each name.
    None means the repo wasn't found on GitHub (deleted or renamed).
    """
    result: dict[str, Optional[bool]] = {}
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=30) as client:
        for i in range(0, len(names), BATCH_SIZE):
            batch = names[i : i + BATCH_SIZE]
            query = build_graphql_query(batch)
            resp = await client.post(GH_GRAPHQL_URL, json={"query": query}, headers=headers)
            resp.raise_for_status()
            data = resp.json().get("data", {}) or {}
            for j, name in enumerate(batch):
                alias = f"r{j}"
                repo_data = data.get(alias)
                result[name] = repo_data["isPrivate"] if repo_data else None

    return result


async def sync(db_url: str, gh_token: str, dry_run: bool = False) -> None:
    engine = create_async_engine(db_url, echo=False)
    Session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with Session() as session:
        rows = await session.execute(text("SELECT id, name, is_private FROM repos ORDER BY name"))
        repos = rows.fetchall()

    print(f"Found {len(repos)} repos in DB")

    names = [r.name for r in repos]
    print(f"Fetching privacy status from GitHub for {len(names)} repos...")
    github_privacy = await fetch_github_privacy(names, gh_token)

    updates_needed: list[tuple[int, str, bool, bool]] = []  # (id, name, db_val, gh_val)
    not_found_on_github: list[str] = []

    for row in repos:
        gh_val = github_privacy.get(row.name)
        if gh_val is None:
            not_found_on_github.append(row.name)
            continue
        if bool(row.is_private) != bool(gh_val):
            updates_needed.append((row.id, row.name, bool(row.is_private), bool(gh_val)))

    if not_found_on_github:
        print(f"\nNot found on GitHub ({len(not_found_on_github)} repos — may be deleted/renamed):")
        for name in not_found_on_github:
            print(f"  {name}")

    if not updates_needed:
        print("\nAll is_private values are already in sync. No changes needed.")
        await engine.dispose()
        return

    print(f"\nMismatches found ({len(updates_needed)} repos):")
    for repo_id, name, db_val, gh_val in updates_needed:
        arrow = "false → true" if gh_val else "true → false"
        print(f"  {name}: DB={db_val} GitHub={gh_val}  ({arrow})")

    if dry_run:
        print("\nDry run — no changes applied. Re-run without --dry-run to apply.")
        await engine.dispose()
        return

    async with Session() as session:
        for repo_id, name, db_val, gh_val in updates_needed:
            await session.execute(
                text("UPDATE repos SET is_private = :val WHERE id = :id"),
                {"val": gh_val, "id": repo_id},
            )
        await session.commit()

    print(f"\nUpdated {len(updates_needed)} repo(s) successfully.")
    await engine.dispose()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Sync is_private from GitHub to the reporium DB")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without applying them")
    args = parser.parse_args()

    db_url, gh_token = resolve_config()
    asyncio.run(sync(db_url, gh_token, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
