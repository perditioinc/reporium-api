"""
Populate reporium-api from reporium-db GitHub data files.

Usage:
    export INGESTION_API_KEY=...
    export REPORIUM_API_URL=https://reporium-api-573778300586.us-central1.run.app
    python scripts/populate_from_reporium_db.py
"""

import asyncio
import json
import os
import sys
import time

import httpx

REPORIUM_DB_RAW = "https://raw.githubusercontent.com/perditioinc/reporium-db/main/data"
API_URL = os.getenv("REPORIUM_API_URL", "https://reporium-api-573778300586.us-central1.run.app")
API_KEY = os.getenv("INGESTION_API_KEY")

if not API_KEY:
    print("ERROR: INGESTION_API_KEY environment variable is required")
    sys.exit(1)

BATCH_SIZE = 50
MAX_RETRIES = 3
CONCURRENCY = 20


def map_repo(raw: dict) -> dict:
    """Map reporium-db fields to reporium-api RepoIngestItem."""
    name_with_owner = raw.get("nameWithOwner", "")
    parts = name_with_owner.split("/", 1)
    owner = parts[0] if len(parts) == 2 else "unknown"
    name = parts[1] if len(parts) == 2 else name_with_owner

    return {
        "name": name,
        "owner": owner,
        "description": raw.get("description"),
        "is_fork": bool(raw.get("isFork", False)),
        "forked_from": raw.get("parentRepo"),
        "primary_language": raw.get("primaryLanguage"),
        "github_url": f"https://github.com/{name_with_owner}",
        "parent_stars": raw.get("parentStars"),
        "parent_forks": raw.get("parentForks"),
        "parent_is_archived": bool(raw.get("isArchived", False)),
        "tags": raw.get("topics", []),
        "github_updated_at": raw.get("pushedAt"),
        "categories": [],
        "builders": [],
        "ai_dev_skills": [],
        "pm_skills": [],
        "languages": (
            [{"language": raw["primaryLanguage"], "bytes": 0, "percentage": 100.0}]
            if raw.get("primaryLanguage")
            else []
        ),
        "commits": [],
    }


async def ingest_batch(
    client: httpx.AsyncClient,
    batch: list[dict],
    semaphore: asyncio.Semaphore,
    stats: dict,
):
    """POST a batch of repos to the ingest endpoint with retry."""
    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                resp = await client.post(
                    f"{API_URL}/ingest/repos",
                    json=batch,
                    headers={"Authorization": f"Bearer {API_KEY}"},
                    timeout=60,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    stats["success"] += data.get("upserted", 0)
                    errors = data.get("errors", [])
                    if errors:
                        stats["failed"] += len(errors)
                        stats["error_details"].extend(errors)
                    return
                elif resp.status_code == 409:
                    stats["existed"] += len(batch)
                    return
                else:
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        stats["failed"] += len(batch)
                        stats["error_details"].append(
                            {"batch_names": [r["name"] for r in batch], "status": resp.status_code, "body": resp.text[:200]}
                        )
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    stats["failed"] += len(batch)
                    stats["error_details"].append(
                        {"batch_names": [r["name"] for r in batch], "error": str(e)}
                    )


async def main():
    start = time.time()

    # Fetch index
    async with httpx.AsyncClient() as client:
        print(f"Fetching index from {REPORIUM_DB_RAW}/index.json ...")
        resp = await client.get(f"{REPORIUM_DB_RAW}/index.json", timeout=30)
        index = resp.json()
        total = index["meta"]["total"]
        print(f"reporium-db reports {total} repos")

        # Fetch the single partition
        print(f"Fetching {REPORIUM_DB_RAW}/full/repos_0000.json ...")
        resp = await client.get(f"{REPORIUM_DB_RAW}/full/repos_0000.json", timeout=60)
        raw_repos = resp.json()
        print(f"Loaded {len(raw_repos)} repos from partition")

    # Map to API schema
    mapped = [map_repo(r) for r in raw_repos]

    # Split into batches
    batches = [mapped[i : i + BATCH_SIZE] for i in range(0, len(mapped), BATCH_SIZE)]
    print(f"Ingesting {len(mapped)} repos in {len(batches)} batches of {BATCH_SIZE}...")

    stats = {"success": 0, "existed": 0, "failed": 0, "error_details": []}
    semaphore = asyncio.Semaphore(CONCURRENCY)

    async with httpx.AsyncClient() as client:
        tasks = [ingest_batch(client, batch, semaphore, stats) for batch in batches]
        done = 0
        for coro in asyncio.as_completed(tasks):
            await coro
            done += 1
            ingested_so_far = stats["success"] + stats["existed"]
            if done % 2 == 0 or done == len(tasks):
                print(f"  Batch {done}/{len(batches)} complete — {ingested_so_far} repos processed")

    duration = time.time() - start

    print()
    print("=" * 50)
    print("Migration complete")
    print(f"  Total repos in reporium-db: {len(raw_repos)}")
    print(f"  Successfully ingested: {stats['success']}")
    print(f"  Already existed: {stats['existed']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Duration: {duration:.1f}s")
    print("=" * 50)

    if stats["error_details"]:
        with open("scripts/failed_repos.json", "w") as f:
            json.dump(stats["error_details"], f, indent=2)
        print(f"\nFailed repos written to scripts/failed_repos.json")

    if stats["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
