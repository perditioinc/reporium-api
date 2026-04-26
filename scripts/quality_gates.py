"""KAN-80: Data quality gates for the Reporium platform.

Reads aggregate counts from the Reporium API over HTTPS — no direct Cloud SQL
connection is required.  Exits with code 1 on threshold breach.  Designed to
run unattended in GitHub Actions (hosted runner, no VPC peering).

Usage:
    python scripts/quality_gates.py               # exit 1 on failure
    python scripts/quality_gates.py --report-only # always exit 0 (informational)

Gates:
1. primary_category coverage >= 95% of public repos
2. embeddings coverage >= 95% of public repos
3. No "_"-prefixed repos in /library/full (heuristic private-leak probe)
4. No NULL is_private values in repos table
5. readme_summary coverage >= 80% of public repos
"""

from __future__ import annotations

import json
import logging
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPORT_ONLY = "--report-only" in sys.argv

THRESHOLDS = {
    "primary_category_coverage_pct": 95.0,
    "embeddings_coverage_pct": 95.0,
    "readme_summary_coverage_pct": 80.0,
    "null_is_private_count": 0,
    "private_repos_in_api": 0,
}


def _fetch_json(url: str, headers: dict[str, str] | None = None, timeout: int = 20) -> dict:
    merged = {"Accept": "application/json", **(headers or {})}
    admin_key = os.getenv("ADMIN_API_KEY", "").strip()
    if admin_key and "X-Admin-Key" not in merged:
        merged["X-Admin-Key"] = admin_key
    req = urllib.request.Request(url, headers=merged)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _pct(numerator: int, denominator: int) -> float:
    return (numerator / denominator * 100) if denominator > 0 else 0.0


def _shortfall(numerator: int, denominator: int, threshold_pct: float) -> int:
    """Count of additional rows needed for `numerator/denominator` to clear `threshold_pct`."""
    if denominator <= 0:
        return 0
    needed = -(-int(threshold_pct * denominator) // 100)  # ceil(threshold * denom / 100)
    return max(0, needed - numerator)


def run_counts_checks(api_url: str) -> list[dict]:
    """Gates 1/2/4/5: read aggregate counts from /metrics/data-quality."""
    try:
        data = _fetch_json(f"{api_url}/metrics/data-quality")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as e:
        return [{
            "gate": "metrics_api_reachable",
            "value": None,
            "threshold": None,
            "unit": None,
            "pass": False,
            "detail": f"GET {api_url}/metrics/data-quality failed: {e}",
            "extra": [],
        }]

    total_public = int(data.get("total_public_repos") or 0)
    with_category = int(data.get("public_with_primary_category") or 0)
    with_readme = int(data.get("public_with_readme_summary") or 0)
    with_embeddings = int(data.get("public_with_embeddings") or 0)
    null_is_private = int(data.get("null_is_private_count") or 0)
    missing_primary_sample = data.get("missing_primary_category_sample") or []

    category_pct = _pct(with_category, total_public)
    readme_pct = _pct(with_readme, total_public)
    emb_pct = _pct(with_embeddings, total_public)

    category_gap = _shortfall(with_category, total_public, THRESHOLDS["primary_category_coverage_pct"])
    primary_extra: list[str] = []
    if category_gap > 0:
        primary_extra.append(
            f"need {category_gap} more enriched repos to reach "
            f"{THRESHOLDS['primary_category_coverage_pct']:.0f}% threshold"
        )
        if missing_primary_sample:
            primary_extra.append(
                "most-recent public repos missing primary_category (top "
                f"{len(missing_primary_sample)}):"
            )
            for entry in missing_primary_sample:
                ingested = entry.get("ingested_at") or "unknown"
                primary_extra.append(f"  - {entry.get('name', '?')}  (ingested_at={ingested})")
        primary_extra.append(
            "fix in: reporium-ingestion enrichment Cloud Run Job — re-run the nightly "
            "enrichment workflow against the missing repo IDs."
        )

    return [
        {
            "gate": "primary_category_coverage",
            "value": round(category_pct, 1),
            "threshold": THRESHOLDS["primary_category_coverage_pct"],
            "unit": "%",
            "pass": category_pct >= THRESHOLDS["primary_category_coverage_pct"],
            "detail": f"{with_category}/{total_public} public repos have primary_category",
            "extra": primary_extra,
        },
        {
            "gate": "embeddings_coverage",
            "value": round(emb_pct, 1),
            "threshold": THRESHOLDS["embeddings_coverage_pct"],
            "unit": "%",
            "pass": emb_pct >= THRESHOLDS["embeddings_coverage_pct"],
            "detail": f"{with_embeddings}/{total_public} public repos have embeddings",
            "extra": [],
        },
        {
            "gate": "null_is_private",
            "value": null_is_private,
            "threshold": THRESHOLDS["null_is_private_count"],
            "unit": "rows",
            "pass": null_is_private <= THRESHOLDS["null_is_private_count"],
            "detail": f"{null_is_private} repos have NULL is_private",
            "extra": [],
        },
        {
            "gate": "readme_summary_coverage",
            "value": round(readme_pct, 1),
            "threshold": THRESHOLDS["readme_summary_coverage_pct"],
            "unit": "%",
            "pass": readme_pct >= THRESHOLDS["readme_summary_coverage_pct"],
            "detail": f"{with_readme}/{total_public} public repos have readme_summary",
            "extra": [],
        },
    ]


def run_library_full_check(api_url: str) -> list[dict]:
    """Gate 3: heuristic private-leak probe against /library/full."""
    try:
        data = _fetch_json(f"{api_url}/library/full?page=1&page_size=100")
        repos = data.get("repos", [])
        private_exposed = [
            r["name"]
            for r in repos
            if r.get("isArchived") is None and r.get("name", "").startswith("_")
        ]
        return [{
            "gate": "no_private_repos_in_api",
            "value": len(private_exposed),
            "threshold": THRESHOLDS["private_repos_in_api"],
            "unit": "repos",
            "pass": len(private_exposed) == 0,
            "detail": f"API returned {len(repos)} repos, {len(private_exposed)} potentially private",
            "extra": [],
        }]
    except Exception as e:  # noqa: BLE001 — surface full failure reason
        return [{
            "gate": "no_private_repos_in_api",
            "value": -1,
            "threshold": 0,
            "unit": "repos",
            "pass": False,
            "detail": f"API check failed: {e}",
            "extra": [],
        }]


def main() -> None:
    logger.info("=" * 60)
    logger.info("Reporium Data Quality Gates")
    logger.info("=" * 60)

    api_url = os.getenv(
        "REPORIUM_API_URL",
        "https://reporium-api-573778300586.us-central1.run.app",
    ).rstrip("/")

    all_results = run_counts_checks(api_url) + run_library_full_check(api_url)

    print()
    print("=" * 60)
    print("QUALITY GATE RESULTS")
    print("=" * 60)

    failures: list[str] = []
    for r in all_results:
        status = "PASS" if r["pass"] else "FAIL"
        print(f"[{status}] {r['gate']}: {r['detail']}")
        if not r["pass"]:
            failures.append(r["gate"])
            for line in r.get("extra") or []:
                print(f"        {line}")

    print()
    if failures:
        print(f"FAILED gates ({len(failures)}): {', '.join(failures)}")
    else:
        print(f"All {len(all_results)} gates passed.")

    if failures and not REPORT_ONLY:
        sys.exit(1)


if __name__ == "__main__":
    main()
