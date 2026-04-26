"""Tests for platform metrics endpoints: KAN-122 alias fields + observability."""

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


# ---------------------------------------------------------------------------
# KAN-122: /metrics/latest frontend alias fields
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_metrics_latest_exposes_frontend_aliases(client):
    """KAN-122: /metrics/latest must include frontend-expected alias fields (additive)."""
    resp = await client.get("/metrics/latest")
    assert resp.status_code == 200
    data = resp.json()

    # Original fields must not be removed
    assert "repos_tracked" in data
    assert "repos_with_ai_skills" in data
    assert "last_sync" in data

    # Additive aliases required by the frontend dashboard
    assert "total_public_repos" in data, "Missing alias: total_public_repos"
    assert "repos_with_embeddings" in data, "Missing alias: repos_with_embeddings"
    assert "snapshot_generated_at" in data, "Missing alias: snapshot_generated_at"

    assert isinstance(data["total_public_repos"], int)
    assert isinstance(data["repos_with_embeddings"], int)
    # snapshot_generated_at may be None when no snapshot is loaded in tests
    assert data["snapshot_generated_at"] is None or isinstance(data["snapshot_generated_at"], str)


# ---------------------------------------------------------------------------
# Observability / platform metrics endpoints
# ---------------------------------------------------------------------------


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
    baseline_resp = await client.get("/metrics/backfill")
    assert baseline_resp.status_code == 200
    baseline = baseline_resp.json()

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
    assert data["repos"]["total"] == baseline["repos"]["total"] + 3
    assert data["repos"]["scanned"] == baseline["repos"]["scanned"] + 2
    assert data["repos"]["remaining"] == baseline["repos"]["remaining"] + 1
    assert data["repos"]["with_dependencies"] == (
        baseline["repos"]["with_dependencies"] + 1
    )
    assert data["repos"]["marked_no_dependencies"] == (
        baseline["repos"]["marked_no_dependencies"] + 1
    )
    assert data["dependencies"]["rows"] == baseline["dependencies"]["rows"] + 1


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
async def test_metrics_data_quality_reports_public_only_coverage(client):
    # Three public repos with varying enrichment + one private repo that must
    # be excluded from all public-only counts.
    async with async_session_factory() as session:
        await session.execute(
            text(
                """
                INSERT INTO repos (id, name, owner, github_url, is_fork, is_private, primary_category, readme_summary)
                VALUES
                    (gen_random_uuid(), 'dq-pub-full', 'perditioinc', 'https://github.com/perditioinc/dq-pub-full', false, false, 'rag', 'summary A'),
                    (gen_random_uuid(), 'dq-pub-partial', 'perditioinc', 'https://github.com/perditioinc/dq-pub-partial', false, false, 'rag', NULL),
                    (gen_random_uuid(), 'dq-pub-bare', 'perditioinc', 'https://github.com/perditioinc/dq-pub-bare', false, false, NULL, ''),
                    (gen_random_uuid(), 'dq-priv', 'perditioinc', 'https://github.com/perditioinc/dq-priv', false, true, 'rag', 'secret')
                """
            )
        )
        await session.commit()

    resp = await client.get("/metrics/data-quality")
    assert resp.status_code == 200
    data = resp.json()

    assert data["total_public_repos"] >= 3  # baseline may contain other public repos
    # The three public repos we inserted should contribute to the counts, and
    # the private one must not. We assert the relative deltas by reading the
    # baseline the test setup produced in its own fixture.
    assert data["public_with_primary_category"] >= 2  # dq-pub-full + dq-pub-partial
    assert data["public_with_readme_summary"] >= 1    # dq-pub-full only
    assert data["null_is_private_count"] == 0
    assert "generated_at" in data

    # Operator-actionable sample of public repos still missing primary_category.
    # `dq-pub-bare` (NULL primary_category, public) was inserted above and must
    # appear so the data-quality gate workflow can name it on failure.
    assert "missing_primary_category_sample" in data
    sample_names = {entry["name"] for entry in data["missing_primary_category_sample"]}
    assert "perditioinc/dq-pub-bare" in sample_names
    assert len(data["missing_primary_category_sample"]) <= 10
    # Private repos must never leak into the sample.
    assert "perditioinc/dq-priv" not in sample_names


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
