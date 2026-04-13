"""Tests for the new observability endpoints in app.routers.platform."""

from __future__ import annotations

import json
import uuid
from types import SimpleNamespace

from sqlalchemy import text

import pytest

from app.database import async_session_factory
from app.models.dependency import RepoDependency
from app.models.repo import RepoCategory
from app.prometheus_metrics import record_http_request
from app.slo_observer import slo_observer


async def _insert_repo(
    *,
    name: str,
    owner: str = "perditioinc",
    forked_from: str | None = None,
    is_fork: bool = False,
    integration_tags: list[str] | None = None,
) -> SimpleNamespace:
    async with async_session_factory() as session:
        repo_id = str(uuid.uuid4())
        await session.execute(
            text("ALTER TABLE repos ADD COLUMN IF NOT EXISTS integration_tags JSONB")
        )
        await session.execute(
            text(
                """
                INSERT INTO repos (id, name, owner, github_url, forked_from, is_fork, is_private, integration_tags)
                VALUES (:id, :name, :owner, :github_url, :forked_from, :is_fork, false, CAST(:integration_tags AS jsonb))
                """
            ),
            {
                "id": repo_id,
                "name": name,
                "owner": owner,
                "github_url": f"https://github.com/{owner}/{name}",
                "forked_from": forked_from,
                "is_fork": is_fork,
                "integration_tags": json.dumps(integration_tags or []),
            },
        )
        await session.commit()
        return SimpleNamespace(id=repo_id)


@pytest.mark.asyncio
async def test_metrics_latency_tracks_graph_edges(client):
    slo_observer.reset()
    for latency in (50.0, 60.0, 80.0, 120.0, 180.0):
        slo_observer.record("/graph/edges", latency, 200)

    resp = await client.get("/metrics/latency")
    assert resp.status_code == 200

    graph_edges = resp.json()["routes"]["/graph/edges"]
    assert graph_edges["observed"]["count"] == 5
    assert graph_edges["observed"]["p95_ms"] is not None
    assert graph_edges["status"] == "ok"
    assert resp.json()["prometheus_endpoint"] == "/metrics/prometheus"


@pytest.mark.asyncio
async def test_metrics_backfill_reports_progress(client):
    repo_a = await _insert_repo(name="repo-a")
    repo_b = await _insert_repo(name="repo-b")
    await _insert_repo(name="repo-c")

    async with async_session_factory() as session:
        session.add(
            RepoDependency(
                repo_id=repo_a.id,
                package_name="numpy",
                package_ecosystem="pypi",
                is_direct=True,
            )
        )
        session.add(
            RepoDependency(
                repo_id=repo_b.id,
                package_name="__none__",
                package_ecosystem="__sentinel__",
                is_direct=False,
            )
        )
        await session.commit()

    resp = await client.get("/metrics/backfill")
    assert resp.status_code == 200

    data = resp.json()
    assert data["available"] is True
    assert data["repos"]["total"] == 3
    assert data["repos"]["scanned"] == 2
    assert data["repos"]["remaining"] == 1
    assert data["repos"]["with_dependencies"] == 1
    assert data["repos"]["marked_no_dependencies"] == 1
    assert data["dependencies"]["rows"] == 1


@pytest.mark.asyncio
async def test_metrics_graph_quality_reports_exact_and_proxy_metrics(client):
    repo_app = await _insert_repo(
        name="rag-app",
        integration_tags=["rag", "vector-database", "python"],
    )
    repo_lib = await _insert_repo(
        name="vectorlib",
        integration_tags=["python"],
    )
    repo_alt = await _insert_repo(
        name="rag-alt",
        integration_tags=["rag", "vector-database", "typescript"],
    )

    async with async_session_factory() as session:
        session.add(
            RepoDependency(
                repo_id=repo_app.id,
                package_name="vectorlib",
                package_ecosystem="pypi",
                is_direct=True,
            )
        )
        session.add_all(
            [
                RepoCategory(
                    repo_id=repo_app.id,
                    category_id="rag",
                    category_name="RAG",
                    is_primary=True,
                ),
                RepoCategory(
                    repo_id=repo_alt.id,
                    category_id="rag",
                    category_name="RAG",
                    is_primary=True,
                ),
            ]
        )
        await session.execute(
            text(
                """
                INSERT INTO repo_edges (source_repo_id, target_repo_id, edge_type, weight)
                VALUES
                    (:app_id, :lib_id, 'DEPENDS_ON', 1.0),
                    (:app_id, :alt_id, 'ALTERNATIVE_TO', 1.0),
                    (:app_id, :alt_id, 'COMPATIBLE_WITH', 0.5)
                """
            ),
            {
                "app_id": str(repo_app.id),
                "lib_id": str(repo_lib.id),
                "alt_id": str(repo_alt.id),
            },
        )
        await session.commit()

    resp = await client.get("/metrics/graph-quality")
    assert resp.status_code == 200

    data = resp.json()
    assert data["available"] is True
    assert data["summary"]["total_edges"] == 3

    depends_on = data["edge_types"]["DEPENDS_ON"]
    assert depends_on["live_edges"] == 1
    assert depends_on["candidate_edges"] == 1
    assert depends_on["matched_edges"] == 1
    assert depends_on["precision"] == 1.0
    assert depends_on["recall"] == 1.0

    alternative = data["edge_types"]["ALTERNATIVE_TO"]
    assert alternative["precision_proxy"] == 1.0

    compatible = data["edge_types"]["COMPATIBLE_WITH"]
    assert compatible["precision_proxy"] == 1.0


@pytest.mark.asyncio
async def test_metrics_prometheus_exposes_http_metrics(client):
    record_http_request(
        path="/graph/edges",
        method="GET",
        status_code=200,
        duration_ms=125.0,
    )

    resp = await client.get("/metrics/prometheus")
    assert resp.status_code == 200
    assert "reporium_http_requests_total" in resp.text
    assert 'route="/graph/edges"' in resp.text
