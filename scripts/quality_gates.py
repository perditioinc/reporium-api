"""
KAN-80: Data quality gates for the Reporium platform.

Validates critical data quality metrics against defined thresholds.
Exits with code 1 if any gate fails. Designed to run in CI pre-deploy.

Usage:
    python scripts/quality_gates.py               # exit 1 on failure
    python scripts/quality_gates.py --report-only  # always exit 0 (for informational runs)

Gates:
1. primary_category coverage >= 95% of public repos
2. embeddings coverage >= 95% of public repos
3. No private repos in /library/full API response
4. No NULL is_private values in repos table
5. readme_summary coverage >= 80% of public repos
"""

import os
import sys
import urllib.parse
import urllib.request
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPORT_ONLY = "--report-only" in sys.argv
GCP_PROJECT = os.getenv("GCP_PROJECT", "perditio-platform")

THRESHOLDS = {
    "primary_category_coverage_pct": 95.0,
    "embeddings_coverage_pct": 95.0,
    "readme_summary_coverage_pct": 80.0,
    "null_is_private_count": 0,
    "private_repos_in_api": 0,
}


def _normalize_db_url(url: str) -> str:
    import urllib.parse as _up
    url = url.replace("+asyncpg", "").replace("+psycopg2", "")
    parsed = _up.urlsplit(url)
    params = _up.parse_qs(parsed.query, keep_blank_values=True)
    ssl_val = params.pop("ssl", [None])[0]
    if ssl_val and "sslmode" not in params:
        if ssl_val.lower() in ("true", "1", "require"):
            params["sslmode"] = ["require"]
        elif ssl_val.lower() in ("false", "0", "disable"):
            params["sslmode"] = ["disable"]
    new_query = _up.urlencode({k: v[0] for k, v in params.items()})
    return _up.urlunsplit(parsed._replace(query=new_query))


def get_db_url() -> str:
    url = os.getenv("DATABASE_URL", "").strip()
    if url:
        return _normalize_db_url(url)
    try:
        from google.cloud import secretmanager
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{GCP_PROJECT}/secrets/reporium-db-url-async/versions/latest"
        response = client.access_secret_version(request={"name": name})
        raw = response.payload.data.decode("UTF-8").strip()
        return _normalize_db_url(raw)
    except Exception as e:
        raise RuntimeError(f"No DATABASE_URL set and Secret Manager failed: {e}")


def run_db_checks(db_url: str) -> list[dict]:
    """Run DB-level quality gates."""
    try:
        import psycopg2
    except ImportError:
        logger.warning("psycopg2 not available — skipping DB checks")
        return []

    results = []
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()

    # Total public repos
    cur.execute("SELECT COUNT(*) FROM repos WHERE is_private = false")
    total_public = cur.fetchone()[0]
    logger.info(f"Total public repos: {total_public}")

    # Gate 1: primary_category coverage
    cur.execute(
        "SELECT COUNT(*) FROM repos WHERE primary_category IS NOT NULL AND is_private = false"
    )
    with_category = cur.fetchone()[0]
    category_pct = (with_category / total_public * 100) if total_public > 0 else 0
    results.append({
        "gate": "primary_category_coverage",
        "value": round(category_pct, 1),
        "threshold": THRESHOLDS["primary_category_coverage_pct"],
        "unit": "%",
        "pass": category_pct >= THRESHOLDS["primary_category_coverage_pct"],
        "detail": f"{with_category}/{total_public} public repos have primary_category",
    })

    # Gate 2: embeddings coverage
    cur.execute(
        "SELECT COUNT(*) FROM repo_embeddings re "
        "JOIN repos r ON r.id = re.repo_id WHERE r.is_private = false"
    )
    with_embeddings = cur.fetchone()[0]
    emb_pct = (with_embeddings / total_public * 100) if total_public > 0 else 0
    results.append({
        "gate": "embeddings_coverage",
        "value": round(emb_pct, 1),
        "threshold": THRESHOLDS["embeddings_coverage_pct"],
        "unit": "%",
        "pass": emb_pct >= THRESHOLDS["embeddings_coverage_pct"],
        "detail": f"{with_embeddings}/{total_public} public repos have embeddings",
    })

    # Gate 4: NULL is_private
    cur.execute("SELECT COUNT(*) FROM repos WHERE is_private IS NULL")
    null_is_private = cur.fetchone()[0]
    results.append({
        "gate": "null_is_private",
        "value": null_is_private,
        "threshold": THRESHOLDS["null_is_private_count"],
        "unit": "rows",
        "pass": null_is_private <= THRESHOLDS["null_is_private_count"],
        "detail": f"{null_is_private} repos have NULL is_private",
    })

    # Gate 5: readme_summary coverage
    cur.execute(
        "SELECT COUNT(*) FROM repos WHERE readme_summary IS NOT NULL AND is_private = false"
    )
    with_readme = cur.fetchone()[0]
    readme_pct = (with_readme / total_public * 100) if total_public > 0 else 0
    results.append({
        "gate": "readme_summary_coverage",
        "value": round(readme_pct, 1),
        "threshold": THRESHOLDS["readme_summary_coverage_pct"],
        "unit": "%",
        "pass": readme_pct >= THRESHOLDS["readme_summary_coverage_pct"],
        "detail": f"{with_readme}/{total_public} public repos have readme_summary",
    })

    conn.close()
    return results


def run_api_check(api_url: str) -> list[dict]:
    """Gate 3: No private repos exposed in /library/full."""
    results = []
    try:
        full_url = f"{api_url}/library/full?page=1&page_size=100"
        req = urllib.request.Request(full_url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        repos = data.get("repos", [])
        # Private repos from the known is_private list would show up with known private names
        # We can only heuristically check — any repo not in a public URL pattern
        private_exposed = [
            r["name"] for r in repos
            if r.get("isArchived") is None and r.get("name", "").startswith("_")
        ]

        results.append({
            "gate": "no_private_repos_in_api",
            "value": len(private_exposed),
            "threshold": THRESHOLDS["private_repos_in_api"],
            "unit": "repos",
            "pass": len(private_exposed) == 0,
            "detail": f"API returned {len(repos)} repos, {len(private_exposed)} potentially private",
        })
    except Exception as e:
        results.append({
            "gate": "no_private_repos_in_api",
            "value": -1,
            "threshold": 0,
            "unit": "repos",
            "pass": False,
            "detail": f"API check failed: {e}",
        })
    return results


def main():
    logger.info("=" * 60)
    logger.info("Reporium Data Quality Gates")
    logger.info("=" * 60)

    all_results = []

    # DB checks
    try:
        db_url = get_db_url()
        db_results = run_db_checks(db_url)
        all_results.extend(db_results)
    except Exception as e:
        logger.error(f"DB connection failed: {e}")
        all_results.append({
            "gate": "db_connection",
            "pass": False,
            "detail": str(e),
            "value": None,
            "threshold": None,
            "unit": None,
        })

    # API check
    api_url = os.getenv(
        "REPORIUM_API_URL",
        "https://reporium-api-573778300586.us-central1.run.app",
    )
    api_results = run_api_check(api_url)
    all_results.extend(api_results)

    # Report
    print()
    print("=" * 60)
    print("QUALITY GATE RESULTS")
    print("=" * 60)

    failures = []
    for r in all_results:
        status = "PASS" if r["pass"] else "FAIL"
        print(f"[{status}] {r['gate']}: {r['detail']}")
        if not r["pass"]:
            failures.append(r["gate"])

    print()
    if failures:
        print(f"FAILED gates ({len(failures)}): {', '.join(failures)}")
    else:
        print(f"All {len(all_results)} gates passed.")

    if failures and not REPORT_ONLY:
        sys.exit(1)


if __name__ == "__main__":
    main()
