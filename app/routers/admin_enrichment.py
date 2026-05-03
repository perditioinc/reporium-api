"""
Admin endpoint that runs the enrichment quality probe (KAN-191) in-process.

KAN-192: the original probe lives in ``reporium-ingestion/ingestion/enrichment/
quality_probe.py`` and connects to the DB directly via psycopg2. That works
inside the Cloud Run Job (Unix-socket Cloud SQL connector) but fails from
GitHub Actions, where ``DATABASE_URL`` points at a socket path that isn't
mounted.

Mirroring the established ``/admin/graph/rebuild-snapshot`` pattern: the API
runs in-VPC where ``DATABASE_URL`` already works, so we expose the probe as
an admin endpoint and let GitHub Actions call it via curl + ``X-Admin-Key``.

The check logic here is a faithful port of the ground-truth probe in
``reporium-ingestion@b10eae9``: same canonical category vocabulary, same
sample-shape semantics, same floor thresholds. The probe-side Python module
is preserved upstream for local development; this is the production path the
nightly workflow now hits.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import require_admin_key
from app.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Admin"])


# ── Canonical vocabulary ──────────────────────────────────────────────────────
#
# ``primary_category`` in the DB stores the human-readable category NAME (see
# reporium-ingestion/ingestion/main.py — written as ``category_name``). This
# must match what ``taxonomy.assign_primary_category()`` returns on the
# enrichment side. The list below is copied verbatim from
# ``reporium-ingestion/ingestion/enrichment/taxonomy.py`` (commit b10eae9).
#
# Drift risk: if the ingestion side adds/removes a category, this list must
# be updated in the same change. The KAN-191 probe noted that
# ``ENRICHMENT_PROMPT_V2.md`` is already out of sync with the live
# ``taxonomy.py`` — we deliberately mirror the live code, not the doc.
CANONICAL_CATEGORY_NAMES: frozenset[str] = frozenset(
    {
        "Foundation Models",
        "AI Agents",
        "RAG & Retrieval",
        "Model Training",
        "Evals & Benchmarking",
        "Observability & Monitoring",
        "Inference & Serving",
        "Generative Media",
        "Computer Vision",
        "Robotics",
        "Spatial & XR",
        "MLOps & Infrastructure",
        "Dev Tools & Automation",
        "Cloud & Platforms",
        "Learning Resources",
        "Industry: Healthcare",
        "Industry: FinTech",
        "Industry: Audio & Music",
        "Industry: Gaming",
        "Security & Safety",
        "Data Science & Analytics",
    }
)


# Telltale strings that indicate the LLM refused / errored / hallucinated a
# meta-comment instead of producing a clean summary. Match anywhere in the
# summary, case-insensitive. Verbatim from the upstream probe.
LLM_FAILURE_MARKERS: tuple[str, ...] = (
    "I cannot",
    "As an AI",
    "I don't have",
    "I do not have",
    "Sorry, I",
    "I'm unable",
    "I am unable",
    "I'm sorry",
    "I am sorry",
)


# ── Result types ──────────────────────────────────────────────────────────────


@dataclass
class CheckResult:
    name: str
    passed: bool
    floor: str
    observed: str
    failures: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ProbeReport:
    run_at: str
    sample_size_target: int
    sample_size_actual: int
    total_enriched_in_corpus: int
    overall_passed: bool
    checks: list[CheckResult]


# ── Config ────────────────────────────────────────────────────────────────────


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r — using default %d", name, raw, default)
        return default


@dataclass
class ProbeConfig:
    sample_size: int = 20
    tags_total_floor: int = 100
    summary_min_chars: int = 50
    summary_max_chars: int = 2000
    tag_max_chars: int = 50
    total_enriched_floor: int = 1500
    freshness_hours: int = 36

    @classmethod
    def from_env(cls) -> "ProbeConfig":
        return cls(
            sample_size=_env_int("PROBE_SAMPLE_SIZE", 20),
            tags_total_floor=_env_int("PROBE_TAGS_TOTAL_FLOOR", 100),
            summary_min_chars=_env_int("PROBE_SUMMARY_MIN_CHARS", 50),
            summary_max_chars=_env_int("PROBE_SUMMARY_MAX_CHARS", 2000),
            tag_max_chars=_env_int("PROBE_TAG_MAX_CHARS", 50),
            total_enriched_floor=_env_int("PROBE_TOTAL_ENRICHED_FLOOR", 1500),
            freshness_hours=_env_int("PROBE_FRESHNESS_HOURS", 36),
        )


# ── DB helpers ────────────────────────────────────────────────────────────────


async def _fetch_sample(
    db: AsyncSession, sample_size: int, freshness_hours: int
) -> list[dict[str, Any]]:
    """Pull a sample of recently-enriched repos for inspection.

    "Recently enriched" = ``updated_at`` within ``freshness_hours`` AND has at
    least one non-null enrichment field. There is no ``last_enriched_at``
    column today; ``updated_at`` is the closest proxy because the enrichment
    UPDATE bumps it (KAN-191 follow-up could add an explicit column).

    If fewer than ``sample_size`` fresh rows exist, falls back to the most-
    recently-updated enriched rows so the probe can still run on demand
    (matches upstream probe behavior).
    """
    sql = text(
        """
        SELECT
            id::text          AS id,
            owner,
            name,
            primary_category,
            integration_tags,
            readme_summary,
            updated_at
        FROM repos
        WHERE is_private = false
          AND (
              primary_category IS NOT NULL
              OR integration_tags IS NOT NULL
              OR readme_summary IS NOT NULL
          )
          AND updated_at >= NOW() - make_interval(hours => :hours)
        ORDER BY updated_at DESC
        LIMIT :limit
        """
    )
    result = await db.execute(sql, {"hours": freshness_hours, "limit": sample_size})
    rows = [dict(r._mapping) for r in result.fetchall()]

    if len(rows) < sample_size:
        fallback_sql = text(
            """
            SELECT
                id::text          AS id,
                owner,
                name,
                primary_category,
                integration_tags,
                readme_summary,
                updated_at
            FROM repos
            WHERE is_private = false
              AND (
                  primary_category IS NOT NULL
                  OR integration_tags IS NOT NULL
                  OR readme_summary IS NOT NULL
              )
            ORDER BY updated_at DESC
            LIMIT :limit
            """
        )
        result = await db.execute(fallback_sql, {"limit": sample_size})
        rows = [dict(r._mapping) for r in result.fetchall()]
    return rows


async def _fetch_total_enriched(db: AsyncSession) -> int:
    """Corpus-shape check: how many public repos have any enrichment at all."""
    sql = text(
        """
        SELECT COUNT(*)
        FROM repos
        WHERE is_private = false
          AND (
              readme_summary IS NOT NULL
              OR primary_category IS NOT NULL
              OR integration_tags IS NOT NULL
          )
        """
    )
    result = await db.execute(sql)
    return int(result.scalar() or 0)


def _coerce_tags(raw: Any) -> list[str]:
    """``integration_tags`` is JSONB — SQLAlchemy/asyncpg auto-decode to list/
    dict, but tolerate strings (raw JSON) and None for safety.
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        return [t for t in raw if isinstance(t, str)]
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            return []
        return (
            [t for t in decoded if isinstance(t, str)]
            if isinstance(decoded, list)
            else []
        )
    return []


# ── Checks ────────────────────────────────────────────────────────────────────


def _short(repo: dict[str, Any]) -> str:
    return f"{repo.get('owner') or '?'}/{repo.get('name') or '?'}"


def check_category_in_vocabulary(sample: list[dict[str, Any]]) -> CheckResult:
    failures: list[dict[str, Any]] = []
    for repo in sample:
        cat = repo.get("primary_category")
        if cat is None:
            # Allow null — sample includes any-enrichment rows; null
            # primary_category is a separate corpus-shape concern (DQ gate).
            continue
        if cat not in CANONICAL_CATEGORY_NAMES:
            failures.append({"repo": _short(repo), "primary_category": cat})
    passed = len(failures) == 0
    return CheckResult(
        name="primary_category_in_vocabulary",
        passed=passed,
        floor="100% of non-null primary_category values must match taxonomy.CATEGORIES names",
        observed=f"{len(sample) - len(failures)}/{len(sample)} valid (or null), {len(failures)} out-of-vocab",
        failures=failures,
    )


def check_tags_present_and_length(
    sample: list[dict[str, Any]],
    *,
    tag_max_chars: int,
    tags_total_floor: int,
) -> CheckResult:
    failures: list[dict[str, Any]] = []
    total_tags = 0
    empty_count = 0
    for repo in sample:
        tags = _coerce_tags(repo.get("integration_tags"))
        if not tags:
            empty_count += 1
            failures.append({"repo": _short(repo), "issue": "empty_tags"})
            continue
        total_tags += len(tags)
        for t in tags:
            if len(t) > tag_max_chars:
                failures.append(
                    {
                        "repo": _short(repo),
                        "issue": "tag_too_long",
                        "tag": t[:80],
                        "len": len(t),
                    }
                )
    if total_tags < tags_total_floor:
        failures.append(
            {
                "issue": "total_tags_below_floor",
                "total_tags": total_tags,
                "floor": tags_total_floor,
            }
        )
    passed = len(failures) == 0
    return CheckResult(
        name="integration_tags_present_and_sane",
        passed=passed,
        floor=(
            f">=1 tag per repo, each tag <= {tag_max_chars} chars, "
            f"total >= {tags_total_floor}"
        ),
        observed=(
            f"{len(sample) - empty_count}/{len(sample)} repos have tags, "
            f"total tags = {total_tags}"
        ),
        failures=failures,
    )


def check_summary_length(
    sample: list[dict[str, Any]],
    *,
    min_chars: int,
    max_chars: int,
) -> CheckResult:
    failures: list[dict[str, Any]] = []
    in_range = 0
    for repo in sample:
        s = repo.get("readme_summary")
        if s is None:
            failures.append({"repo": _short(repo), "issue": "null_summary"})
            continue
        n = len(s)
        if n < min_chars:
            failures.append({"repo": _short(repo), "issue": "too_short", "len": n})
        elif n > max_chars:
            failures.append({"repo": _short(repo), "issue": "too_long", "len": n})
        else:
            in_range += 1
    passed = len(failures) == 0
    return CheckResult(
        name="readme_summary_length_in_range",
        passed=passed,
        floor=f"100% of summaries in [{min_chars}, {max_chars}] chars",
        observed=f"{in_range}/{len(sample)} in range, {len(failures)} out of range or null",
        failures=failures,
    )


def check_no_llm_failure_markers(sample: list[dict[str, Any]]) -> CheckResult:
    failures: list[dict[str, Any]] = []
    for repo in sample:
        s = repo.get("readme_summary")
        if not s:
            continue
        s_lower = s.lower()
        for marker in LLM_FAILURE_MARKERS:
            if marker.lower() in s_lower:
                failures.append(
                    {"repo": _short(repo), "marker": marker, "snippet": s[:160]}
                )
                break  # one hit per repo is enough
    passed = len(failures) == 0
    return CheckResult(
        name="no_llm_failure_markers_in_summary",
        passed=passed,
        floor="0 occurrences across sample",
        observed=f"{len(failures)} repos contain a failure marker",
        failures=failures,
    )


def check_total_enriched_floor(total_enriched: int, floor: int) -> CheckResult:
    passed = total_enriched >= floor
    return CheckResult(
        name="total_enriched_corpus_floor",
        passed=passed,
        floor=f">= {floor} enriched public repos",
        observed=f"{total_enriched} enriched public repos",
        failures=(
            []
            if passed
            else [{"issue": "below_floor", "observed": total_enriched, "floor": floor}]
        ),
    )


# ── Orchestration ─────────────────────────────────────────────────────────────


async def run_probe(db: AsyncSession, config: ProbeConfig) -> ProbeReport:
    sample = await _fetch_sample(db, config.sample_size, config.freshness_hours)
    total_enriched = await _fetch_total_enriched(db)

    checks = [
        check_category_in_vocabulary(sample),
        check_tags_present_and_length(
            sample,
            tag_max_chars=config.tag_max_chars,
            tags_total_floor=config.tags_total_floor,
        ),
        check_summary_length(
            sample,
            min_chars=config.summary_min_chars,
            max_chars=config.summary_max_chars,
        ),
        check_no_llm_failure_markers(sample),
        check_total_enriched_floor(total_enriched, config.total_enriched_floor),
    ]

    return ProbeReport(
        run_at=datetime.now(timezone.utc).isoformat(),
        sample_size_target=config.sample_size,
        sample_size_actual=len(sample),
        total_enriched_in_corpus=total_enriched,
        overall_passed=all(c.passed for c in checks),
        checks=checks,
    )


# ── HTTP endpoint ─────────────────────────────────────────────────────────────


@router.post("/admin/enrichment/quality-probe", response_model=dict)
async def enrichment_quality_probe(
    db: AsyncSession = Depends(get_db),
    _admin_key: None = Depends(require_admin_key),
) -> dict:
    """KAN-192 — run the KAN-191 enrichment quality probe in-process.

    Returns ``200`` with the full probe report when every check passes, or
    ``422`` with the same shape when any check fails. The HTTP non-2xx is
    what the caller (GitHub Actions ``nightly_enrichment_quality_probe.yml``)
    pivots on to fail the workflow and trigger the existing Workato → JIRA
    notify-on-failure pipeline.

    Config is read from environment variables on the API side (``PROBE_*``,
    same names as the upstream probe), not from the request body, so that
    changing thresholds is an ops action against the running deployment
    rather than a CI-side knob exposed to anyone with the admin key.
    """
    config = ProbeConfig.from_env()
    report = await run_probe(db, config)
    payload = asdict(report)

    if not report.overall_passed:
        # 422 Unprocessable Entity: the request was well-formed but the
        # probe found data-quality breaches that the caller must surface.
        # Body still includes the full report so the workflow log can show
        # exactly which check tripped without a second round-trip.
        raise HTTPException(status_code=422, detail=payload)

    return payload
