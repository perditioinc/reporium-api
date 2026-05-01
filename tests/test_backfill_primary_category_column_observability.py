"""Observability tests for POST /admin/backfill/primary_category_column.

Covers KAN-API-OBS-BACKFILL: correlation ID propagation (header-supplied or
generated), structured `backfill.end` log line, Prometheus counter increment
with the right outcome label.
"""

import logging
import uuid

import pytest
from httpx import AsyncClient
from sqlalchemy import text

import app.database as db_module
from app.prometheus_metrics import ADMIN_BACKFILL_RUNS_TOTAL
from tests.conftest import AUTH_HEADERS


def _success_count() -> float:
    """Read the current `admin_backfill_runs_total{outcome="success"}` value.

    Works for both the real prometheus_client Counter and the in-process
    `_FallbackMetric` shim so tests stay green whether or not
    prometheus_client is installed.
    """
    counter = ADMIN_BACKFILL_RUNS_TOTAL
    if hasattr(counter, "_metrics"):  # prometheus_client.Counter
        for labels, child in counter._metrics.items():
            if labels == ("success",):
                return child._value.get()
        return 0.0
    # Fallback shim: values is a defaultdict keyed by sorted-tuple-of-pairs.
    return float(counter.values.get((("outcome", "success"),), 0.0))


async def _delete_repos(ids: list[str]) -> None:
    async with db_module.async_session_factory() as session:
        await session.execute(
            text("DELETE FROM repo_categories WHERE repo_id = ANY(:ids)"),
            {"ids": ids},
        )
        await session.execute(
            text("DELETE FROM repos WHERE id = ANY(:ids)"),
            {"ids": ids},
        )
        await session.commit()


@pytest.mark.asyncio
async def test_backfill_logs_end_with_nonzero_duration_and_counter_bumps(
    client: AsyncClient, caplog: pytest.LogCaptureFixture
):
    """Smoke-test the observability surface end-to-end:

    1. The handler logs `backfill.end` with a populated `duration_ms`.
    2. Each call increments `admin_backfill_runs_total{outcome="success"}`
       by exactly 1.
    3. The response body carries `correlation_id` and `duration_ms`.
    """
    drift_id = str(uuid.uuid4())
    inserted_ids = [drift_id]

    try:
        async with db_module.async_session_factory() as session:
            await session.execute(text(
                "INSERT INTO repos (id, name, owner, github_url, is_fork, is_private, primary_category) "
                "VALUES (:id, 'pcc-obs-drift-repo', 'testuser', :url, false, false, NULL) "
                "ON CONFLICT (name) DO UPDATE SET primary_category = NULL"
            ), {"id": drift_id, "url": "https://github.com/testuser/pcc-obs-drift-repo"})
            await session.execute(text(
                "INSERT INTO repo_categories (repo_id, category_id, category_name, is_primary) "
                "VALUES (:id, 'rag-retrieval', 'RAG & Retrieval', true) "
                "ON CONFLICT (repo_id, category_id) DO UPDATE SET is_primary = true"
            ), {"id": drift_id})
            await session.commit()

        before_count = _success_count()

        with caplog.at_level(logging.INFO, logger="app.routers.admin"):
            resp = await client.post(
                "/admin/backfill/primary_category_column",
                headers=AUTH_HEADERS,
            )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "correlation_id" in body
        assert isinstance(body["duration_ms"], (int, float))
        assert body["duration_ms"] >= 0

        end_records = [
            r for r in caplog.records
            if getattr(r, "event", None) == "backfill.end"
        ]
        assert end_records, "expected a backfill.end log record"
        end_rec = end_records[-1]
        assert getattr(end_rec, "outcome", None) == "success"
        assert getattr(end_rec, "duration_ms", 0) > 0
        assert getattr(end_rec, "rows_updated", -1) >= 1

        after_count = _success_count()
        assert after_count - before_count == pytest.approx(1.0, abs=1e-9), (
            f"counter should bump by exactly 1; before={before_count} after={after_count}"
        )
    finally:
        await _delete_repos(inserted_ids)


@pytest.mark.asyncio
async def test_backfill_propagates_caller_supplied_correlation_id(
    client: AsyncClient, caplog: pytest.LogCaptureFixture
):
    """When the caller provides `X-Correlation-ID`, it is preserved verbatim
    in both the structured log lines and the response body."""
    supplied_cid = "obs-test-" + uuid.uuid4().hex[:12]

    headers = dict(AUTH_HEADERS)
    headers["X-Correlation-ID"] = supplied_cid

    with caplog.at_level(logging.INFO, logger="app.routers.admin"):
        resp = await client.post(
            "/admin/backfill/primary_category_column?dry_run=true",
            headers=headers,
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["correlation_id"] == supplied_cid

    matching = [
        r for r in caplog.records
        if getattr(r, "correlation_id", None) == supplied_cid
    ]
    assert any(getattr(r, "event", None) == "backfill.start" for r in matching), (
        "expected backfill.start log with caller correlation_id"
    )
    assert any(getattr(r, "event", None) == "backfill.end" for r in matching), (
        "expected backfill.end log with caller correlation_id"
    )


@pytest.mark.asyncio
async def test_backfill_generates_correlation_id_when_header_absent(
    client: AsyncClient,
):
    """When no `X-Correlation-ID` is supplied, the handler mints a UUID4 and
    surfaces it in the response so the caller can correlate after-the-fact."""
    resp = await client.post(
        "/admin/backfill/primary_category_column?dry_run=true",
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 200, resp.text
    cid = resp.json()["correlation_id"]
    # Should parse as a UUID (raises ValueError otherwise).
    parsed = uuid.UUID(cid)
    assert parsed.version == 4
