#!/usr/bin/env python3
"""
Data quality check — runs after every deploy and nightly.
Creates GitHub issues when quality drops below threshold.

Usage:
    python scripts/check_data_quality.py

Requires: REPORIUM_API_URL, INGESTION_API_KEY, GH_TOKEN env vars
"""
import json
import os
import sys

import httpx

API_URL = os.getenv("REPORIUM_API_URL", "https://reporium-api-573778300586.us-central1.run.app")
API_KEY = os.getenv("INGESTION_API_KEY", "")
GH_TOKEN = os.getenv("GH_TOKEN", "")

PRIVATE_NAMES = {
    'didymo-ai-agent', 'didymo-ai-api', 'didymo-ai-auth', 'didymo-ai-gcp-tts',
    'didymo-ai-ingest', 'didymo-ai-mini', 'didymo-ai-openai-stt', 'didymo-ai-openai-tts',
    'didymo-ai-ptr', 'didymo-ai-services-lab', 'didymo-ai-studio', 'didymo-ai-submissions-website',
    'didymo-ai-usage', 'didymo-ai-vector', 'didymo-ai-webgl', 'didymo-ai-webgl-v2',
    'didymo-ai-website', 'digital-panda-planner', 'event-schedule-generator',
    'figma-make-perditio-website-claude', 'giveaway-generator', 'ideas-2026',
    'mind-guard-app', 'perditio-figma-website', 'perditio-infra', 'perditio-platform-api',
    'perditio-services', 'perditio-web', 'perditio-web-app', 'perditio-website',
    'perditioinc.github.io', 'simon-brain', 'ticket-generator', 'ticket-issuer',
    'v0-edm-demo-submission-website', 'whatsapp-template-generator', 'whatsapp-webhook',
    '18degrees-ecom', 'aa-backend-interview-template-main',
}


def check_library_full() -> list[str]:
    """Check /library/full for data quality issues."""
    issues = []
    resp = httpx.get(f"{API_URL}/library/full", timeout=30)
    if resp.status_code != 200:
        issues.append(f"/library/full returned HTTP {resp.status_code}")
        return issues

    data = resp.json()
    repos = data.get("repos", [])

    # Check: no private repos
    exposed = [r["name"] for r in repos if r["name"] in PRIVATE_NAMES]
    if exposed:
        issues.append(f"CRITICAL: {len(exposed)} private repos exposed: {exposed[:5]}")

    # Check: no null descriptions
    no_desc = [r["name"] for r in repos if not r.get("description")]
    if no_desc:
        issues.append(f"{len(no_desc)} repos missing description")

    # Check: duplicate tags
    dup_tags = sum(1 for r in repos if len(r.get("enrichedTags", [])) != len(set(r.get("enrichedTags", []))))
    if dup_tags:
        issues.append(f"{dup_tags} repos have duplicate tags")

    # Check: forks have parent data
    forks = [r for r in repos if r.get("isFork")]
    no_parent = [r["name"] for r in forks if not r.get("forkedFrom")]
    if no_parent:
        issues.append(f"{len(no_parent)} forks missing forkedFrom")

    print(f"Checked {len(repos)} repos: {len(issues)} issue(s)")
    return issues


def create_issue(title: str, body: str) -> None:
    """Create a GitHub issue on reporium-api."""
    if not GH_TOKEN:
        print(f"SKIP issue creation (no GH_TOKEN): {title}")
        return
    resp = httpx.post(
        "https://api.github.com/repos/perditioinc/reporium-api/issues",
        headers={"Authorization": f"Bearer {GH_TOKEN}", "Accept": "application/vnd.github+json"},
        json={"title": title, "body": body, "labels": ["bug", "automated", "data-quality"]},
        timeout=15,
    )
    if resp.status_code == 201:
        print(f"Created issue: {resp.json()['html_url']}")
    else:
        print(f"Failed to create issue: {resp.status_code}")


def main():
    issues = check_library_full()
    if not issues:
        print("ALL CHECKS PASS")
        return

    print("ISSUES FOUND:")
    for i in issues:
        print(f"  - {i}")

    # Create GitHub issue for critical problems
    critical = [i for i in issues if "CRITICAL" in i or "private" in i.lower()]
    if critical:
        create_issue(
            "SECURITY: Private repos exposed in /library/full",
            "## Data Quality Check Failed\n\n" + "\n".join(f"- {i}" for i in issues) +
            "\n\n*Auto-created by check_data_quality.py*"
        )
        sys.exit(1)

    # Non-critical issues — create issue but don't fail
    if len(issues) > 2:
        create_issue(
            f"Data quality: {len(issues)} issues found",
            "## Data Quality Check\n\n" + "\n".join(f"- {i}" for i in issues) +
            "\n\n*Auto-created by check_data_quality.py*"
        )


if __name__ == "__main__":
    main()
