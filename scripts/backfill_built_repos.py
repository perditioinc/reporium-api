#!/usr/bin/env python3
"""
One-time backfill: fetch stars, forks, languages, and recent commits
from GitHub API for all built (non-forked) repos in the database,
then update the DB directly.

Usage:
  DATABASE_URL=postgresql+asyncpg://... GITHUB_TOKEN=ghp_... python scripts/backfill_built_repos.py
"""

import asyncio
import os
import sys
from datetime import datetime, timezone, timedelta

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.environ.get("DATABASE_URL", "")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
OWNER = "perditioinc"

if not DATABASE_URL:
    print("ERROR: Set DATABASE_URL environment variable")
    sys.exit(1)
if not GITHUB_TOKEN:
    print("ERROR: Set GITHUB_TOKEN environment variable")
    sys.exit(1)

engine = create_async_engine(DATABASE_URL, echo=False)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
}


async def fetch_github_data(client: httpx.AsyncClient, repo_name: str) -> dict | None:
    """Fetch stars, forks, languages, and recent commits from GitHub REST API."""
    base = f"https://api.github.com/repos/{OWNER}/{repo_name}"

    # Repo metadata (stars, forks)
    resp = await client.get(base, headers=HEADERS)
    if resp.status_code != 200:
        print(f"  SKIP {repo_name}: GitHub returned {resp.status_code}")
        return None
    meta = resp.json()

    # Languages
    lang_resp = await client.get(f"{base}/languages", headers=HEADERS)
    languages = lang_resp.json() if lang_resp.status_code == 200 else {}

    # Recent commits (last 90 days)
    since = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
    commits_resp = await client.get(
        f"{base}/commits",
        params={"since": since, "per_page": 100},
        headers=HEADERS,
    )
    commits = commits_resp.json() if commits_resp.status_code == 200 and isinstance(commits_resp.json(), list) else []

    return {
        "stars": meta.get("stargazers_count", 0),
        "forks": meta.get("forks_count", 0),
        "open_issues": meta.get("open_issues_count", 0),
        "languages": languages,  # {lang: bytes}
        "commits": commits,
        "updated_at": meta.get("updated_at"),
        "pushed_at": meta.get("pushed_at"),
    }


async def main():
    async with async_session() as db:
        # Get all built (non-forked) repos
        result = await db.execute(text(
            "SELECT id, name FROM repos WHERE is_fork = false AND is_private = false"
        ))
        built_repos = result.fetchall()
        print(f"Found {len(built_repos)} built repos to backfill\n")

        async with httpx.AsyncClient(timeout=30) as client:
            for repo_id, repo_name in built_repos:
                print(f"Processing {repo_name}...")
                data = await fetch_github_data(client, repo_name)
                if not data:
                    continue

                # Update repo metadata
                def _parse_dt(s):
                    if not s:
                        return None
                    return datetime.fromisoformat(s.replace("Z", "+00:00"))

                await db.execute(text("""
                    UPDATE repos SET
                        stargazers_count = :stars,
                        open_issues_count = :open_issues,
                        github_updated_at = :updated_at,
                        your_last_push_at = :pushed_at,
                        updated_at = NOW()
                    WHERE id = :repo_id
                """), {
                    "stars": data["stars"],
                    "open_issues": data["open_issues"],
                    "updated_at": _parse_dt(data["updated_at"]),
                    "pushed_at": _parse_dt(data["pushed_at"]),
                    "repo_id": str(repo_id),
                })

                # Update languages
                await db.execute(text(
                    "DELETE FROM repo_languages WHERE repo_id = :repo_id"
                ), {"repo_id": str(repo_id)})

                total_bytes = sum(data["languages"].values()) or 1
                for lang, byte_count in data["languages"].items():
                    pct = round(byte_count / total_bytes * 100, 1)
                    await db.execute(text("""
                        INSERT INTO repo_languages (repo_id, language, bytes, percentage)
                        VALUES (:repo_id, :lang, :bytes, :pct)
                    """), {
                        "repo_id": str(repo_id),
                        "lang": lang,
                        "bytes": byte_count,
                        "pct": pct,
                    })

                # Update commit counts
                now = datetime.now(timezone.utc)
                c7 = sum(1 for c in data["commits"]
                         if _days_ago(c, now) <= 7)
                c30 = sum(1 for c in data["commits"]
                          if _days_ago(c, now) <= 30)
                c90 = len(data["commits"])

                await db.execute(text("""
                    UPDATE repos SET
                        commits_last_7_days = :c7,
                        commits_last_30_days = :c30,
                        commits_last_90_days = :c90
                    WHERE id = :repo_id
                """), {"c7": c7, "c30": c30, "c90": c90, "repo_id": str(repo_id)})

                # Insert recent commits
                await db.execute(text(
                    "DELETE FROM repo_commits WHERE repo_id = :repo_id"
                ), {"repo_id": str(repo_id)})

                for c in data["commits"][:50]:  # cap at 50 commits
                    commit_data = c.get("commit", {})
                    sha = c.get("sha", "")
                    message = (commit_data.get("message") or "")[:500]
                    author = commit_data.get("author", {}).get("name", "")
                    date_str = commit_data.get("author", {}).get("date", "")
                    url = c.get("html_url", "")

                    if sha and date_str:
                        await db.execute(text("""
                            INSERT INTO repo_commits (id, repo_id, sha, message, author, committed_at, url)
                            VALUES (gen_random_uuid(), :repo_id, :sha, :message, :author, :date, :url)
                        """), {
                            "repo_id": str(repo_id),
                            "sha": sha,
                            "message": message,
                            "author": author,
                            "date": _parse_dt(date_str),
                            "url": url,
                        })

                print(f"  ✓ stars={data['stars']}, forks={data['forks']}, "
                      f"langs={len(data['languages'])}, commits={c90} (7d={c7}, 30d={c30})")

                # Small delay to avoid rate limiting
                await asyncio.sleep(0.5)

        await db.commit()
        print("\nBackfill complete!")


def _days_ago(commit: dict, now: datetime) -> int:
    """Calculate days between a commit and now."""
    date_str = commit.get("commit", {}).get("author", {}).get("date", "")
    if not date_str:
        return 999
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return (now - dt).days
    except (ValueError, TypeError):
        return 999


if __name__ == "__main__":
    asyncio.run(main())
