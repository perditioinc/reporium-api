"""Tests for the KAN-192 admin enrichment quality probe endpoint.

The probe logic itself is a port of the KAN-191 ground-truth probe in
``reporium-ingestion``. These tests cover both:

  * The pure check functions (no DB) — verifies the port matches the
    upstream behavior for boundary cases on every check.
  * The HTTP endpoint round-trip (auth + DB + status code) — verifies the
    wiring that GitHub Actions pivots on.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from httpx import AsyncClient
from sqlalchemy import text

from app.database import async_session_factory
from app.routers.admin_enrichment import (
    CANONICAL_CATEGORY_NAMES,
    LLM_FAILURE_MARKERS,
    ProbeConfig,
    check_category_in_vocabulary,
    check_no_llm_failure_markers,
    check_summary_length,
    check_tags_present_and_length,
    check_total_enriched_floor,
)


# ── Pure check functions ──────────────────────────────────────────────────────


def _repo(
    *,
    owner: str = "octocat",
    name: str = "demo",
    primary_category: str | None = None,
    integration_tags=None,
    readme_summary: str | None = None,
) -> dict:
    return {
        "owner": owner,
        "name": name,
        "primary_category": primary_category,
        "integration_tags": integration_tags,
        "readme_summary": readme_summary,
    }


def test_canonical_vocabulary_has_expected_size():
    # The upstream probe enumerates 21 categories — guards against a silent
    # taxonomy drop that would let any LLM output through unchallenged.
    assert len(CANONICAL_CATEGORY_NAMES) == 21
    assert "AI Agents" in CANONICAL_CATEGORY_NAMES
    assert "Data Science & Analytics" in CANONICAL_CATEGORY_NAMES


def test_check_category_passes_when_all_in_vocab():
    sample = [
        _repo(primary_category="AI Agents"),
        _repo(primary_category="RAG & Retrieval"),
        _repo(primary_category=None),  # null is allowed (separate concern)
    ]
    result = check_category_in_vocabulary(sample)
    assert result.passed is True
    assert result.failures == []


def test_check_category_fails_on_out_of_vocab():
    sample = [
        _repo(primary_category="AI Agents"),
        _repo(owner="bad", name="repo", primary_category="Made Up Category"),
    ]
    result = check_category_in_vocabulary(sample)
    assert result.passed is False
    assert len(result.failures) == 1
    assert result.failures[0]["repo"] == "bad/repo"
    assert result.failures[0]["primary_category"] == "Made Up Category"


def test_check_tags_passes_with_enough_total():
    # 20 repos × 5 tags = 100 tags total — exactly at floor
    sample = [_repo(integration_tags=["a", "b", "c", "d", "e"]) for _ in range(20)]
    result = check_tags_present_and_length(
        sample, tag_max_chars=50, tags_total_floor=100
    )
    assert result.passed is True


def test_check_tags_fails_on_empty():
    sample = [_repo(integration_tags=[]), _repo(integration_tags=["ok"])]
    result = check_tags_present_and_length(
        sample, tag_max_chars=50, tags_total_floor=1
    )
    assert result.passed is False
    assert any(f.get("issue") == "empty_tags" for f in result.failures)


def test_check_tags_fails_on_overlong_tag():
    long_tag = "x" * 51
    sample = [_repo(integration_tags=[long_tag, "ok"])]
    result = check_tags_present_and_length(
        sample, tag_max_chars=50, tags_total_floor=1
    )
    assert result.passed is False
    assert any(f.get("issue") == "tag_too_long" for f in result.failures)


def test_check_tags_fails_on_total_below_floor():
    sample = [_repo(integration_tags=["only-one"])]
    result = check_tags_present_and_length(
        sample, tag_max_chars=50, tags_total_floor=100
    )
    assert result.passed is False
    assert any(f.get("issue") == "total_tags_below_floor" for f in result.failures)


def test_check_tags_handles_jsonb_string_form():
    # JSONB usually decodes to list, but tolerate string-of-JSON.
    sample = [_repo(integration_tags='["a", "b", "c"]')]
    result = check_tags_present_and_length(
        sample, tag_max_chars=50, tags_total_floor=1
    )
    # 3 tags total, none too long → passes the floor of 1
    assert result.passed is True


def test_check_summary_passes_in_range():
    sample = [_repo(readme_summary="x" * 100), _repo(readme_summary="y" * 500)]
    result = check_summary_length(sample, min_chars=50, max_chars=2000)
    assert result.passed is True


def test_check_summary_fails_on_too_short():
    sample = [_repo(owner="a", name="b", readme_summary="too short")]
    result = check_summary_length(sample, min_chars=50, max_chars=2000)
    assert result.passed is False
    assert any(f.get("issue") == "too_short" for f in result.failures)


def test_check_summary_fails_on_too_long():
    sample = [_repo(owner="a", name="b", readme_summary="x" * 3000)]
    result = check_summary_length(sample, min_chars=50, max_chars=2000)
    assert result.passed is False
    assert any(f.get("issue") == "too_long" for f in result.failures)


def test_check_summary_fails_on_null():
    sample = [_repo(owner="a", name="b", readme_summary=None)]
    result = check_summary_length(sample, min_chars=50, max_chars=2000)
    assert result.passed is False
    assert any(f.get("issue") == "null_summary" for f in result.failures)


def test_check_no_llm_failure_markers_passes_clean():
    sample = [_repo(readme_summary="A useful library for vector search.")]
    result = check_no_llm_failure_markers(sample)
    assert result.passed is True


@pytest.mark.parametrize("marker", LLM_FAILURE_MARKERS)
def test_check_llm_failure_markers_catches_each(marker):
    sample = [_repo(owner="a", name="b", readme_summary=f"{marker} help with this.")]
    result = check_no_llm_failure_markers(sample)
    assert result.passed is False
    assert len(result.failures) == 1
    assert result.failures[0]["marker"] == marker


def test_check_llm_failure_markers_is_case_insensitive():
    sample = [_repo(owner="a", name="b", readme_summary="i CANNOT do that, sorry.")]
    result = check_no_llm_failure_markers(sample)
    assert result.passed is False


def test_check_total_enriched_floor_passes_at_threshold():
    result = check_total_enriched_floor(1500, 1500)
    assert result.passed is True


def test_check_total_enriched_floor_fails_below():
    result = check_total_enriched_floor(1499, 1500)
    assert result.passed is False
    assert result.failures[0]["issue"] == "below_floor"


# ── Config from env ───────────────────────────────────────────────────────────


def test_probe_config_defaults():
    config = ProbeConfig.from_env()
    # Defaults should match the upstream probe verbatim.
    assert config.sample_size == 20
    assert config.tags_total_floor == 100
    assert config.summary_min_chars == 50
    assert config.summary_max_chars == 2000
    assert config.tag_max_chars == 50
    assert config.total_enriched_floor == 1500
    assert config.freshness_hours == 36


def test_probe_config_env_overrides(monkeypatch):
    monkeypatch.setenv("PROBE_SAMPLE_SIZE", "5")
    monkeypatch.setenv("PROBE_TOTAL_ENRICHED_FLOOR", "10")
    config = ProbeConfig.from_env()
    assert config.sample_size == 5
    assert config.total_enriched_floor == 10


def test_probe_config_invalid_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("PROBE_SAMPLE_SIZE", "not-a-number")
    config = ProbeConfig.from_env()
    assert config.sample_size == 20  # falls back


# ── HTTP endpoint integration ────────────────────────────────────────────────
#
# These tests exercise the full request → DB → response path. They depend on
# the conftest test DB fixture (skipped automatically when the test Postgres
# is unreachable, same as every other admin test).


@pytest.fixture
def test_admin_key(monkeypatch):
    """Set ADMIN_API_KEY to a known value AND switch to production mode so
    require_admin_key actually enforces it (default test config leaves it
    unset and require_admin_key short-circuits in non-prod)."""
    monkeypatch.setenv("ADMIN_API_KEY", "test-admin-key")
    monkeypatch.setenv("ENVIRONMENT", "production")
    yield "test-admin-key"


@pytest.mark.asyncio
async def test_quality_probe_requires_admin_key(client: AsyncClient, test_admin_key):
    response = await client.post("/admin/enrichment/quality-probe")
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_quality_probe_rejects_wrong_key(client: AsyncClient, test_admin_key):
    response = await client.post(
        "/admin/enrichment/quality-probe",
        headers={"X-Admin-Key": "nope"},
    )
    assert response.status_code == 403


async def _seed_enriched_repos(count: int, *, fresh: bool = True) -> list[str]:
    """Insert ``count`` minimal enriched public repos directly via SQL.

    Bypasses the ORM so we can write JSONB/UUID values without bringing in
    the full ingest payload. Returns the inserted IDs for cleanup.
    """
    inserted: list[str] = []
    updated_at = datetime.now(timezone.utc) - (
        timedelta(hours=1) if fresh else timedelta(days=30)
    )
    async with async_session_factory() as session:
        for i in range(count):
            rid = str(uuid.uuid4())
            inserted.append(rid)
            await session.execute(
                text(
                    """
                    INSERT INTO repos (
                        id, owner, name, github_url, primary_category,
                        integration_tags, readme_summary, is_private,
                        updated_at
                    ) VALUES (
                        :id, :owner, :name, :url, :pc,
                        CAST(:tags AS JSONB), :summary, false,
                        :updated_at
                    )
                    """
                ),
                {
                    "id": rid,
                    "owner": "kan192",
                    "name": f"repo-{i}",
                    "url": f"https://github.com/kan192/repo-{i}",
                    "pc": "AI Agents",
                    "tags": '["agents", "llm", "tools", "memory", "planning"]',
                    "summary": "A minimal but realistic README summary used for the KAN-192 probe integration test. " * 2,
                    "updated_at": updated_at,
                },
            )
        await session.commit()
    return inserted


async def _purge_repos(ids: list[str]) -> None:
    if not ids:
        return
    async with async_session_factory() as session:
        await session.execute(
            text("DELETE FROM repos WHERE id = ANY(CAST(:ids AS uuid[]))"),
            {"ids": ids},
        )
        await session.commit()


@pytest.mark.asyncio
async def test_quality_probe_returns_422_on_empty_corpus(
    client: AsyncClient, test_admin_key, monkeypatch
):
    """With a fresh test DB the corpus is empty — total_enriched is 0,
    well below the default 1500 floor, so the probe must FAIL.
    """
    # Lower floors so the test isn't dependent on huge sample seeding.
    monkeypatch.setenv("PROBE_TOTAL_ENRICHED_FLOOR", "1500")
    response = await client.post(
        "/admin/enrichment/quality-probe",
        headers={"X-Admin-Key": test_admin_key},
    )
    assert response.status_code == 422
    body = response.json()
    detail = body["detail"]
    assert detail["overall_passed"] is False
    # The total_enriched check must be one of the failures.
    names = {c["name"] for c in detail["checks"] if not c["passed"]}
    assert "total_enriched_corpus_floor" in names


@pytest.mark.asyncio
async def test_quality_probe_returns_200_when_all_pass(
    client: AsyncClient, test_admin_key, monkeypatch
):
    """With a small seeded corpus and lowered thresholds, every check
    should pass and the endpoint should return 200 with the report."""
    # Lower floors to match what we can realistically seed in a test DB.
    monkeypatch.setenv("PROBE_TOTAL_ENRICHED_FLOOR", "5")
    monkeypatch.setenv("PROBE_TAGS_TOTAL_FLOOR", "5")
    monkeypatch.setenv("PROBE_SAMPLE_SIZE", "5")

    seeded = await _seed_enriched_repos(5, fresh=True)
    try:
        response = await client.post(
            "/admin/enrichment/quality-probe",
            headers={"X-Admin-Key": test_admin_key},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["overall_passed"] is True
        assert body["sample_size_actual"] == 5
        assert body["total_enriched_in_corpus"] >= 5
        # Every check must report passed=True
        assert all(c["passed"] for c in body["checks"])
    finally:
        await _purge_repos(seeded)
