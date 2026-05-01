"""
Nightly data invariants for Reporium staging.

These tests catch 6 classes of silent regression:

  1. Mass-delete regression — totalRepos floor (>= 1400)
  2. Taxonomy enrichment regression — at least 3 dimensions populated
  3. Trend snapshots stall — weekly cron producing any data at all
     (test_trend_snapshots_exist)
  4. Trend snapshots stale — most recent snapshot must be < 48h old
     (test_trend_snapshots_recent)
  5. Commit-stats stall — not every repo can have commits_last_7_days == 0
     (test_commit_stats_not_universally_zero)
  6. Fork-discovery stall — at least one repo must have been forked/added
     in the last 30 days (test_repo_discovery_recent)

Tests 4-6 + the strengthened #3 were added 2026-04-30 after a 10-day silent
staleness incident: the existing nightly workflow stayed GREEN every day
(see workflow runs 2026-04-21 through 2026-04-30) while production data
was visibly stale — period.snapshots=0, every repo's commitStats.last7Days=0,
no new repos in 15+ days. Root cause for the silent green: test #3 was
marked @pytest.mark.xfail(strict=False), which masks a failing assertion
as XFAIL (still green). It now asserts strictly.

These tests will INTENTIONALLY FAIL in nightly runs until the sibling
fixes for the trend-snapshot writer (T1), commit-stats refresh (T2), and
fork-discovery sweep (T3) land and are redeployed. The failure is the
alarm working as designed.

Guards:
  - Only runs when STAGING_API_URL env var is set (skipped in CI unit-test
    runs and PR-time test workflow; nightly workflow populates it via secret).

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
from datetime import datetime, timedelta, timezone

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

# Trend snapshots must be written daily; alarm if the latest is older than this.
SNAPSHOT_FRESHNESS_HOURS = 48

# Fork-discovery sweep should add new repos at least monthly.
DISCOVERY_FRESHNESS_DAYS = 30

# Pagination ceiling for /library/full sweeps. ~1900 repos / 500 = 4 pages,
# 8 pages here is generous headroom in case totalRepos grows.
MAX_PAGES = 8
PAGE_SIZE = 500

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


def _parse_iso(value: str | None) -> datetime | None:
    """Tolerant ISO-8601 parser. Returns None for empty / unparseable strings."""
    if not value:
        return None
    try:
        # fromisoformat handles "+00:00" but not bare "Z" until Python 3.11+.
        # Be defensive for older runners.
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _iter_library_full() -> list[dict]:
    """Page through /library/full and return all repo dicts.

    Stops at the lesser of:
      - totalRepos (from page 1)
      - MAX_PAGES (safety cap)
    """
    all_repos: list[dict] = []
    page = 1
    total: int | None = None

    while page <= MAX_PAGES:
        resp = _get(
            "/library/full",
            params={"page": page, "page_size": PAGE_SIZE},
        )
        assert resp.status_code == 200, (
            f"GET /library/full page={page} returned {resp.status_code}: {resp.text[:300]}"
        )
        data = resp.json()
        if total is None:
            total = data.get("totalRepos") or 0
        repos = data.get("repos", []) or []
        all_repos.extend(repos)
        if total is not None and len(all_repos) >= total:
            break
        if not repos:
            # Empty page — protects against an off-by-one infinite loop.
            break
        page += 1

    return all_repos


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
        except httpx.TimeoutException:
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
# Invariant 3 — trend snapshots exist (table not empty)
# ---------------------------------------------------------------------------


@pytest.mark.invariants
def test_trend_snapshots_exist():
    """
    GET /trends/report → period.snapshots > 0.

    Catches: ingestion weekly cron not running, trend_snapshots table never
    populated, snapshot pipeline silent failure.

    History: this test was previously @pytest.mark.xfail(strict=False),
    which silently masked the empty-table case as XFAIL (still GREEN).
    Removed 2026-04-30 after the empty-snapshots state went undetected
    for 10+ days. The strict assertion below now fails loudly so the
    nightly workflow turns RED and notifies Workato → JIRA.
    """
    resp = _get("/trends/report")
    assert resp.status_code == 200, (
        f"GET /trends/report returned {resp.status_code}: {resp.text[:300]}"
    )
    data = resp.json()
    period = data.get("period", {}) or {}
    snapshots = period.get("snapshots", 0) or 0
    assert snapshots > 0, (
        f"trends/report.period.snapshots={snapshots} — "
        "trend_snapshots table is empty; weekly ingestion writer is not running"
    )


# ---------------------------------------------------------------------------
# Invariant 4 — trend snapshots are recent (not just non-empty)
# ---------------------------------------------------------------------------


@pytest.mark.invariants
def test_trend_snapshots_recent():
    """
    GET /trends/report → period.to (latest snapshotted_at) must be within
    the last SNAPSHOT_FRESHNESS_HOURS hours.

    Catches: snapshot writer ran historically but has stopped recently —
    e.g. a Cloud Run Job whose schedule was disabled, an auth/permission
    regression that silently broke the daily insert, or any other case
    where the table has rows but isn't being refreshed.

    test_trend_snapshots_exist (#3) catches the empty-table case;
    this test catches the stale-table case. Both must pass.
    """
    resp = _get("/trends/report")
    assert resp.status_code == 200, (
        f"GET /trends/report returned {resp.status_code}: {resp.text[:300]}"
    )
    data = resp.json()
    period = data.get("period", {}) or {}
    latest_iso = period.get("to")
    assert latest_iso, (
        f"trends/report.period.to is null (period={period!r}) — "
        "no snapshots written; cannot evaluate freshness"
    )
    latest = _parse_iso(latest_iso)
    assert latest is not None, (
        f"trends/report.period.to={latest_iso!r} is unparseable"
    )
    now = datetime.now(timezone.utc)
    age_hours = (now - latest).total_seconds() / 3600.0
    assert age_hours <= SNAPSHOT_FRESHNESS_HOURS, (
        f"latest trend_snapshot is {age_hours:.1f}h old "
        f"(at {latest_iso}); freshness ceiling is {SNAPSHOT_FRESHNESS_HOURS}h. "
        "Snapshot writer may be stalled."
    )


# ---------------------------------------------------------------------------
# Invariant 5 — commit_stats not universally zero
# ---------------------------------------------------------------------------


@pytest.mark.invariants
def test_commit_stats_not_universally_zero():
    """
    Sweep all repos via /library/full and assert the sum of
    commitStats.last7Days across the entire corpus is > 0.

    Catches: the exact 2026-04 production state where every repo's
    commits_last_7_days column was 0 because the commit-refresh job
    silently stopped. A single active fork in the corpus is enough to
    keep this green; only the universal-zero case fails.

    Rationale for sum > 0 rather than per-row sampling: with ~1900 repos,
    a few may legitimately have no commits in the last 7 days (archived,
    truly idle). The smell we want to catch is "EVERY repo == 0" which
    is what the commit-refresh stall produces.
    """
    repos = _iter_library_full()
    assert repos, (
        "GET /library/full returned 0 repos — corpus is empty; "
        "this is also caught by test_total_repos_floor but worth flagging"
    )

    def _c7(r: dict) -> int:
        # Be defensive: commitStats may be missing on a malformed row.
        stats = r.get("commitStats") or {}
        try:
            return int(stats.get("last7Days") or 0)
        except (TypeError, ValueError):
            return 0

    total_c7 = sum(_c7(r) for r in repos)
    nonzero_count = sum(1 for r in repos if _c7(r) > 0)

    assert total_c7 > 0, (
        f"commitStats.last7Days summed across {len(repos)} repos == 0 "
        f"(0 repos with any 7-day commits). The commit-refresh job has "
        "silently stalled — every row is zero. This is the 2026-04-XX "
        "regression pattern."
    )
    # Soft observability — show the distribution in the failure log even on pass.
    # Not an assertion; just useful in --tb=short output if a *later* test fails.
    print(
        f"[invariant] commits-7d sum={total_c7} nonzero_repos={nonzero_count}/{len(repos)}"
    )


# ---------------------------------------------------------------------------
# Invariant 6 — fork-discovery sweep is recent
# ---------------------------------------------------------------------------


@pytest.mark.invariants
def test_repo_discovery_recent():
    """
    Sweep all repos via /library/full and assert at least one repo was
    forked or added (createdAt fallback for non-fork rows) within the
    last DISCOVERY_FRESHNESS_DAYS days.

    Catches: the fork-discovery sweep / nightly enrichment job has stopped
    adding new repos to the corpus. This is the "no new repos in 15+ days"
    half of the 2026-04 staleness incident.

    Field semantics (from app/routers/library_full.py):
      - forkedAt: when *we* forked the upstream (most reliable freshness signal)
      - createdAt: for forks → upstream_created_at (NOT a freshness signal,
        upstream may be years old); for non-forks → ingested_at OR
        github_updated_at (which IS our DB row creation for non-forks)

    We accept the latest of forkedAt OR (createdAt for non-forks). If
    nothing has appeared in 30 days, the discovery job is stalled.
    """
    repos = _iter_library_full()
    assert repos, "GET /library/full returned 0 repos — cannot evaluate"

    cutoff = datetime.now(timezone.utc) - timedelta(days=DISCOVERY_FRESHNESS_DAYS)
    most_recent: datetime | None = None
    most_recent_name: str | None = None

    for r in repos:
        forked_at = _parse_iso(r.get("forkedAt"))
        # createdAt is only a row-freshness signal for non-fork rows.
        is_fork = bool(r.get("forkedFrom") or r.get("forked_from"))
        created_at = None if is_fork else _parse_iso(r.get("createdAt"))

        for ts in (forked_at, created_at):
            if ts is None:
                continue
            if most_recent is None or ts > most_recent:
                most_recent = ts
                most_recent_name = r.get("name")

    assert most_recent is not None, (
        f"None of {len(repos)} repos had a parseable forkedAt or non-fork "
        "createdAt timestamp — cannot determine discovery freshness"
    )
    assert most_recent >= cutoff, (
        f"Most recent repo discovery was {most_recent.isoformat()} "
        f"(repo={most_recent_name!r}), older than the {DISCOVERY_FRESHNESS_DAYS}-day "
        "freshness ceiling. Fork-discovery sweep is stalled."
    )
