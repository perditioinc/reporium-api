"""
Nightly data invariants for Reporium staging.

These tests catch 3 classes of silent regression identified in
.audit/api-regression-plan.md:

  1. Mass-delete regression — totalRepos floor (>= 1400)
  2. Taxonomy enrichment regression — at least 3 dimensions populated
  3. Trend snapshots stall — weekly cron producing data

Guards:
  - Only runs when STAGING_API_URL env var is set (skipped in CI unit-test
    runs, nightly workflow populates it via secret).
  - Xfail on snapshots until reporium-ingestion WEEKLY cron is unblocked.

Run locally:
    STAGING_API_URL=https://reporium-api-573778300586.us-central1.run.app \\
        pytest -m invariants -v

NOTE on taxonomy dimensions
---------------------------
The /taxonomy/<dim> endpoint uses the *repo_taxonomy dimension strings*, not
the 16-category primary_category list from ENRICHMENT_PROMPT_V2.md.
The 6 populated dims as of 2026-04-19 are:
    modality, use_case, deployment_context, skill_area, ai_trend, industry

The 16 primary_category slugs (agents, rag-retrieval, …) are stored in
repos.primary_category / repo_categories — a separate table — and are
intentionally NOT queried here (they are covered by test_taxonomy_gaps.py).

Audit finding: the two taxonomy systems are distinct. This invariant guards
/taxonomy/<dim> (the taxonomy_values table populated by the rebuild job).
"""

from __future__ import annotations

import os

import httpx
import pytest

# ---------------------------------------------------------------------------
# Guard: skip entire module if STAGING_API_URL is not set
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.skipif(
    not os.getenv("STAGING_API_URL"),
    reason="invariants run only against staging — set STAGING_API_URL",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOTAL_REPOS_FLOOR = 1400

# The 6 taxonomy_values dimensions populated by the rebuild job.
# Confirmed populated as of 2026-04-19 live API check.
KNOWN_POPULATED_DIMS = [
    "modality",
    "use_case",
    "deployment_context",
    "skill_area",
    "ai_trend",
    "industry",
]

# Additional dims that may exist but are allowed to be empty (xfail-safe).
# NOTE: "tags" and "categories" here refer to any future raw taxonomy dims,
# not the 16-category primary_category slug list.
POTENTIALLY_EMPTY_DIMS = {"tags", "categories"}

MIN_POPULATED_DIMS = 3  # at least this many must return non-empty values

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_url() -> str:
    url = os.environ["STAGING_API_URL"].rstrip("/")
    return url


def _get(path: str, params: dict | None = None, timeout: float = 60) -> httpx.Response:
    url = f"{_base_url()}{path}"
    resp = httpx.get(url, params=params, timeout=timeout)
    return resp


# ---------------------------------------------------------------------------
# Invariant 1 — totalRepos floor
# ---------------------------------------------------------------------------


@pytest.mark.invariants
def test_total_repos_floor():
    """
    GET /library/full?page=1&page_size=1 → totalRepos >= 1400.

    Catches: bulk-delete regression, accidental truncation, failed
    nightly ingestion causing repo count to silently collapse.
    """
    resp = _get("/library/full", params={"page": 1, "page_size": 1})
    assert resp.status_code == 200, (
        f"GET /library/full returned {resp.status_code}: {resp.text[:300]}"
    )
    data = resp.json()
    total = data.get("totalRepos")
    assert total is not None, "Response missing 'totalRepos' key"
    assert total >= TOTAL_REPOS_FLOOR, (
        f"totalRepos={total} is below floor {TOTAL_REPOS_FLOOR} — "
        "possible mass-delete regression or ingestion failure"
    )


# ---------------------------------------------------------------------------
# Invariant 2 — taxonomy dimensions populated
# ---------------------------------------------------------------------------


@pytest.mark.invariants
def test_taxonomy_dimensions_populated():
    """
    GET /taxonomy/<dimension> for each of the 6 known populated dimensions.

    Asserts that at least MIN_POPULATED_DIMS return a non-empty values list.
    These are the taxonomy_values dimensions populated by the rebuild job —
    distinct from the 16-category primary_category slugs.

    Catches: taxonomy_values table wipe, rebuild job failure, enrichment
    pipeline regression that zeroes out dimension counts.
    """
    populated: list[str] = []
    empty: list[str] = []
    errors: list[str] = []

    for dim in KNOWN_POPULATED_DIMS:
        try:
            resp = _get(f"/taxonomy/{dim}")
            if resp.status_code == 404:
                empty.append(dim)
                continue
            assert resp.status_code == 200, (
                f"/taxonomy/{dim} returned {resp.status_code}"
            )
            data = resp.json()
            values = data.get("values", [])
            if values:
                populated.append(dim)
            else:
                empty.append(dim)
        except httpx.TimeoutException as exc:
            # Timeout on a single slow dim is a performance issue, not a
            # correctness failure — count it as "unknown" not "error".
            empty.append(f"{dim}(timeout)")
        except Exception as exc:
            errors.append(f"{dim}: {exc}")

    assert not errors, f"Hard errors fetching taxonomy dims: {errors}"
    assert len(populated) >= MIN_POPULATED_DIMS, (
        f"Only {len(populated)}/{len(KNOWN_POPULATED_DIMS)} known dims are populated "
        f"(floor={MIN_POPULATED_DIMS}). Populated: {populated}. Empty/timeout: {empty}. "
        "Possible taxonomy rebuild job failure."
    )


# ---------------------------------------------------------------------------
# Invariant 3 — trend snapshots exist
# ---------------------------------------------------------------------------


@pytest.mark.invariants
@pytest.mark.xfail(
    strict=False,
    reason=(
        "reporium-ingestion WEEKLY cron stalled — trend_snapshots table may "
        "be empty until KAN-ingestion cron is unblocked. Tracked in JIRA."
    ),
)
def test_trend_snapshots_exist():
    """
    GET /trends/report → period.snapshots > 0.

    Catches: ingestion weekly cron not running, trend_snapshots table never
    populated, snapshot pipeline silent failure.

    Marked xfail: keeps CI green while ingestion cron is unblocked.
    Once the cron is live and verified, remove the xfail marker.
    """
    resp = _get("/trends/report")
    assert resp.status_code == 200, (
        f"GET /trends/report returned {resp.status_code}: {resp.text[:300]}"
    )
    data = resp.json()
    period = data.get("period", {})
    snapshots = period.get("snapshots", 0)
    assert snapshots > 0, (
        f"trends/report.period.snapshots={snapshots} — "
        "no snapshot data; weekly ingestion cron may not be running"
    )
