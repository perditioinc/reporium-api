"""P0 PRIVACY REGRESSION GUARD — single-source coverage of every public
surface that reads repos.

Each prior leak (PRs #154/#156/#161, #313, #414, 2026-04-27) was a forgotten
``is_private = false`` filter. This file is the structural backstop: seed a
private repo (with paired public sibling), probe every public endpoint, and
assert the private one never surfaces.

Add a new public endpoint? Add it to ``ENDPOINTS_BY_NAME`` so this file
catches the leak before review.
"""
from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy import text

import app.database as db_module


PRIVATE_NAME = "no-leak-private-probe"
PUBLIC_NAME = "no-leak-public-probe"
PUBLIC_OWNER = "perditioinc"
PRIVATE_OWNER = "perditioinc"


# ---------------------------------------------------------------------------
# Seeding helpers
# ---------------------------------------------------------------------------

async def _insert_repo(
    *,
    name: str,
    owner: str,
    is_private: bool,
    description: str = "",
    primary_category: str | None = None,
    primary_language: str = "Python",
    integration_tags: list[str] | None = None,
    embedding: list[float] | None = None,
) -> str:
    """Insert a repo (and optional embedding) directly via SQL.

    Returns the new repo id (str). Ignores duplicate-name conflicts so tests
    can be re-run against the same DB without manual cleanup between runs.
    """
    repo_id = str(uuid.uuid4())
    async with db_module.async_session_factory() as session:
        await session.execute(
            text(
                """
                INSERT INTO repos
                    (id, name, owner, github_url, description,
                     is_fork, is_private, primary_category,
                     primary_language, integration_tags)
                VALUES
                    (:id, :name, :owner, :github_url, :description,
                     false, :is_private, :primary_category,
                     :primary_language, CAST(:integration_tags AS jsonb))
                ON CONFLICT (name) DO UPDATE
                    SET is_private = EXCLUDED.is_private,
                        description = EXCLUDED.description,
                        primary_category = EXCLUDED.primary_category
                RETURNING id::text
                """
            ),
            {
                "id": repo_id,
                "name": name,
                "owner": owner,
                "github_url": f"https://github.com/{owner}/{name}",
                "description": description,
                "is_private": is_private,
                "primary_category": primary_category,
                "primary_language": primary_language,
                "integration_tags": json.dumps(integration_tags or []),
            },
        )
        # ON CONFLICT DO UPDATE returns the existing id, not the candidate one,
        # so re-read the canonical id after the upsert.
        row = (
            await session.execute(
                text("SELECT id::text FROM repos WHERE name = :name"),
                {"name": name},
            )
        ).first()
        repo_id = row[0]

        if embedding is not None:
            vec_str = "[" + ",".join(f"{v:.6f}" for v in embedding) + "]"
            # `id` is omitted: in production it has DEFAULT gen_random_uuid()
            # (migration 034); in CI the conftest creates only the migration-001
            # schema where `id` does not exist at all. Letting the column
            # default fill it makes this insert work in both environments.
            await session.execute(
                text(
                    """
                    INSERT INTO repo_embeddings (repo_id, model, embedding_vec)
                    VALUES (CAST(:repo_id AS uuid), :model, CAST(:vec AS vector))
                    ON CONFLICT DO NOTHING
                    """
                ),
                {
                    "repo_id": repo_id,
                    "model": "all-MiniLM-L6-v2",
                    "vec": vec_str,
                },
            )
        await session.commit()
    return repo_id


async def _insert_dependency(
    *, repo_id: str, package_name: str, ecosystem: str = "pypi"
) -> None:
    async with db_module.async_session_factory() as session:
        await session.execute(
            text(
                """
                INSERT INTO repo_dependencies
                    (id, repo_id, package_name, package_ecosystem,
                     version_constraint, is_direct)
                VALUES
                    (gen_random_uuid(), CAST(:repo_id AS uuid),
                     :package_name, :ecosystem, '*', true)
                ON CONFLICT DO NOTHING
                """
            ),
            {
                "repo_id": repo_id,
                "package_name": package_name,
                "ecosystem": ecosystem,
            },
        )
        await session.commit()


async def _insert_mention(*, repo_id: str, source: str = "hackernews") -> None:
    async with db_module.async_session_factory() as session:
        await session.execute(
            text(
                """
                INSERT INTO repo_mentions
                    (id, repo_id, source, external_id, title, url, score)
                VALUES
                    (gen_random_uuid(), CAST(:repo_id AS uuid),
                     :source, :external_id, 'fixture', 'https://example.com', 1)
                ON CONFLICT DO NOTHING
                """
            ),
            {
                "repo_id": repo_id,
                "source": source,
                "external_id": f"fixture-{repo_id[:8]}",
            },
        )
        await session.commit()


# ---------------------------------------------------------------------------
# Fixture: seed paired public + private repos with embeddings for every test
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def paired_repos(_setup_db) -> dict[str, str]:
    """Seed a public+private pair with similar embeddings, mentions, and a
    shared dependency. Returns ``{"public_id": ..., "private_id": ...}``.
    """
    # Near-identical embeddings so a similarity query for one will pull the
    # other if the privacy filter is missing — that's the leak vector we're
    # guarding against.
    pub_emb = [0.05] * 384
    priv_emb = [0.0501] * 384

    public_id = await _insert_repo(
        name=PUBLIC_NAME,
        owner=PUBLIC_OWNER,
        is_private=False,
        description="public probe — must always be visible",
        primary_category="Testing",
        primary_language="Python",
        integration_tags=["langchain"],
        embedding=pub_emb,
    )
    private_id = await _insert_repo(
        name=PRIVATE_NAME,
        owner=PRIVATE_OWNER,
        is_private=True,
        description="PRIVATE probe — must NEVER appear in any public surface",
        primary_category="Testing",
        primary_language="Python",
        integration_tags=["langchain"],
        embedding=priv_emb,
    )

    # Both repos share a dependency so the /dependencies/dependents endpoint
    # can be probed with one package name.
    await _insert_dependency(repo_id=public_id, package_name="leakguard-pkg")
    await _insert_dependency(repo_id=private_id, package_name="leakguard-pkg")

    # Both repos have a mention so /repos/{id}/mentions can be probed.
    await _insert_mention(repo_id=public_id)
    await _insert_mention(repo_id=private_id)

    yield {"public_id": public_id, "private_id": private_id}

    # Best-effort cleanup so re-running tests in a developer's local DB stays
    # idempotent. Conftest's DROP SCHEMA teardown handles CI; this is for
    # in-process re-runs.
    async with db_module.async_session_factory() as session:
        for name in (PRIVATE_NAME, PUBLIC_NAME):
            await session.execute(
                text("DELETE FROM repos WHERE name = :name"),
                {"name": name},
            )
        await session.commit()


# ---------------------------------------------------------------------------
# Helpers used by the tests
# ---------------------------------------------------------------------------

def _flatten_strings(payload: Any) -> str:
    """Render any JSON payload as a flat string for substring assertions."""
    return json.dumps(payload, default=str)


def _assert_no_private(payload: Any, *, where: str) -> None:
    flat = _flatten_strings(payload)
    assert PRIVATE_NAME not in flat, (
        f"PRIVACY LEAK at {where}: {PRIVATE_NAME!r} appeared in response. "
        "A public surface is missing the is_private = false filter."
    )


# ===========================================================================
# Tests — list / detail endpoints
# ===========================================================================


@pytest.mark.asyncio
async def test_library_full_excludes_private(client: AsyncClient, paired_repos):
    resp = await client.get("/library/full?page=1&page_size=500")
    assert resp.status_code == 200
    body = resp.json()
    assert any(r.get("name") == PUBLIC_NAME for r in body.get("repos", [])), (
        "public probe should appear in /library/full"
    )
    _assert_no_private(body, where="/library/full")


@pytest.mark.asyncio
async def test_library_excludes_private(client: AsyncClient, paired_repos):
    resp = await client.get("/library?limit=500")
    assert resp.status_code == 200
    body = resp.json()
    _assert_no_private(body, where="/library")


@pytest.mark.asyncio
async def test_repos_list_excludes_private(client: AsyncClient, paired_repos):
    resp = await client.get("/repos?limit=200")
    assert resp.status_code == 200
    body = resp.json()
    _assert_no_private(body, where="/repos")


@pytest.mark.asyncio
async def test_repos_detail_404_for_private(
    client: AsyncClient, paired_repos
):
    resp = await client.get(f"/repos/{PRIVATE_NAME}")
    assert resp.status_code == 404, (
        f"PRIVACY LEAK: /repos/{PRIVATE_NAME} returned {resp.status_code}, "
        "should 404 to avoid confirming the private repo's existence"
    )


@pytest.mark.asyncio
async def test_repos_owner_repo_404_for_private(
    client: AsyncClient, paired_repos
):
    resp = await client.get(f"/repos/{PRIVATE_OWNER}/{PRIVATE_NAME}")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_repos_by_uuid_404_for_private(
    client: AsyncClient, paired_repos
):
    resp = await client.get(f"/repos/{paired_repos['private_id']}")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_search_excludes_private(client: AsyncClient, paired_repos):
    # Match against the description fragment so we'd catch a leak even if the
    # name was filtered out at the response-shaping layer.
    resp = await client.get("/search?q=probe")
    assert resp.status_code == 200
    _assert_no_private(resp.json(), where="/search")


@pytest.mark.asyncio
async def test_search_semantic_excludes_private(
    client: AsyncClient, paired_repos
):
    resp = await client.get("/search/semantic?q=probe")
    assert resp.status_code == 200
    _assert_no_private(resp.json(), where="/search/semantic")


# ===========================================================================
# Tests — mentions / dependencies
# ===========================================================================


@pytest.mark.asyncio
async def test_repo_mentions_404_for_private(
    client: AsyncClient, paired_repos
):
    resp = await client.get(
        f"/repos/{paired_repos['private_id']}/mentions"
    )
    assert resp.status_code == 404, (
        "PRIVACY LEAK: GET /repos/{private_id}/mentions must 404, not "
        "expose the existence of the private repo (UUID-oracle attack)."
    )


@pytest.mark.asyncio
async def test_repo_mentions_200_for_public(
    client: AsyncClient, paired_repos
):
    """Sanity: public repo's mentions endpoint still works."""
    resp = await client.get(
        f"/repos/{paired_repos['public_id']}/mentions"
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_repo_dependencies_404_for_private(
    client: AsyncClient, paired_repos
):
    resp = await client.get(
        f"/repos/{paired_repos['private_id']}/dependencies"
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_repo_dependencies_200_for_public(
    client: AsyncClient, paired_repos
):
    resp = await client.get(
        f"/repos/{paired_repos['public_id']}/dependencies"
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_dependents_excludes_private(client: AsyncClient, paired_repos):
    resp = await client.get(
        "/dependencies/dependents?package=leakguard-pkg"
    )
    assert resp.status_code == 200
    body = resp.json()
    _assert_no_private(body, where="/dependencies/dependents")
    # Also verify the public dependent IS in the response — proves the test
    # is exercising real data, not just an empty result.
    assert any(r.get("name") == PUBLIC_NAME for r in body), (
        "public probe must still appear in dependents list — otherwise the "
        "test passes trivially even when the filter is wrong"
    )


# ===========================================================================
# Tests — recommendations / similar
# ===========================================================================


@pytest.mark.asyncio
async def test_similar_404_for_private_seed(
    client: AsyncClient, paired_repos
):
    """Querying recommendations using a private repo as seed must 404 — the
    seed_r join must not allow private rows through."""
    resp = await client.get(f"/intelligence/similar/{PRIVATE_NAME}")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_similar_results_exclude_private(
    client: AsyncClient, paired_repos
):
    """Querying recommendations from the public seed must NOT surface the
    private sibling in the similar list (defense-in-depth on the target r)."""
    resp = await client.get(
        f"/intelligence/similar/{PUBLIC_NAME}?limit=24&min_similarity=0.0"
    )
    # Either 200 with no private name, or 404 if no neighbours pass the
    # threshold. Both are acceptable; a 200 with PRIVATE_NAME is the leak.
    assert resp.status_code in (200, 404)
    if resp.status_code == 200:
        _assert_no_private(resp.json(), where="/intelligence/similar")


# ===========================================================================
# Tests — graph
# ===========================================================================


@pytest.mark.asyncio
async def test_graph_edges_excludes_private(
    client: AsyncClient, paired_repos
):
    resp = await client.get("/graph/edges?limit=2000")
    # 200 if the graph snapshot is present, 503 if not — either way no private
    # name should leak. We only assert when 200 so empty deployments don't
    # false-fail this guard.
    if resp.status_code == 200:
        _assert_no_private(resp.json(), where="/graph/edges")


# ===========================================================================
# Tests — ASK retrieval (related-edges SQL leak)
# ===========================================================================
#
# /intelligence/ask requires LLM credentials to exercise end-to-end. The leak
# surface that needs guarding is the related-edges hydration SQL — the same
# SQL runs regardless of LLM. We invoke it directly via the test DB so this
# test passes/fails on the structural property (private rows in the JOIN),
# not on Anthropic API availability.


_RELATED_EDGES_SQL = text(
    """
    SELECT
        e1.repo_id::text AS source_id,
        e2.repo_id::text AS target_id,
        r1.name AS source_name,
        r2.name AS target_name,
        1 - (e1.embedding_vec <=> e2.embedding_vec) AS similarity
    FROM repo_embeddings e1
    CROSS JOIN LATERAL (
        SELECT e2_inner.repo_id, e2_inner.embedding_vec
        FROM repo_embeddings e2_inner
        JOIN repos r_inner ON r_inner.id = e2_inner.repo_id
                          AND r_inner.is_private = false
        WHERE e2_inner.repo_id != e1.repo_id
        ORDER BY e1.embedding_vec <=> e2_inner.embedding_vec
        LIMIT 8
    ) e2
    JOIN repos r1 ON r1.id = e1.repo_id AND r1.is_private = false
    JOIN repos r2 ON r2.id = e2.repo_id AND r2.is_private = false
    WHERE e1.repo_id::text = ANY(:ids)
    """
)


@pytest.mark.asyncio
async def test_ask_related_edges_query_excludes_private(paired_repos):
    """Direct SQL probe of the /intelligence/ask related-edges hydration.

    Seeds a public + private repo with near-identical embeddings, runs the
    same SQL the ASK retrieval uses, asserts the private repo is never
    returned as either source or target. This is the structural test for
    the 2026-04-27 leak.
    """
    async with db_module.async_session_factory() as session:
        rows = (
            await session.execute(
                _RELATED_EDGES_SQL,
                {"ids": [paired_repos["public_id"]]},
            )
        ).fetchall()

    target_names = [r.target_name for r in rows]
    source_names = [r.source_name for r in rows]
    assert PRIVATE_NAME not in target_names, (
        f"PRIVACY LEAK in /intelligence/ask related-edges hydration: "
        f"{PRIVATE_NAME!r} returned as target_name. The JOIN on repos r2 "
        "is missing is_private = false."
    )
    assert PRIVATE_NAME not in source_names, (
        f"PRIVACY LEAK in /intelligence/ask related-edges hydration: "
        f"{PRIVATE_NAME!r} returned as source_name."
    )


# ===========================================================================
# Tests — centralized helper unit-level
# ===========================================================================


def test_db_filters_module_exposes_helpers():
    """The centralized helper module exists and exposes the documented API."""
    from app import db_filters

    # SQL constant + ORM predicate factory
    assert db_filters.PUBLIC_REPO_SQL_PREDICATE == "is_private = false"
    expr = db_filters.public_repo_filter()
    assert expr is not None

    # Pre-filtered builder
    stmt = db_filters.public_repos_select()
    compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    assert "is_private" in compiled
    assert "false" in compiled.lower()

    # Aliased SQL fragment helper
    assert db_filters.sql_public_filter() == "r.is_private = false"
    assert db_filters.sql_public_filter("r1") == "r1.is_private = false"


def test_db_filters_sql_fragment_rejects_injection():
    """sql_public_filter validates the alias against SQL injection."""
    from app import db_filters

    with pytest.raises(ValueError):
        db_filters.sql_public_filter("r; DROP TABLE repos --")
    with pytest.raises(ValueError):
        db_filters.sql_public_filter("r OR 1=1")
    with pytest.raises(ValueError):
        db_filters.sql_public_filter("")


# ===========================================================================
# Static guards — defend against future regressions by asserting the privacy
# filters remain present in the source files we hardened.
# ===========================================================================
#
# The DB-dependent tests above exercise behaviour, but they need a Postgres
# instance to fail when the filter is dropped. The static guards below run
# in any environment (CI, local, no DB) and turn red the moment a filter is
# deleted — surfacing the regression at the next test run instead of
# waiting for a CI DB job.

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _read_source(rel_path: str) -> str:
    return (_REPO_ROOT / rel_path).read_text(encoding="utf-8")


def test_intelligence_related_edges_sql_has_privacy_filters():
    """The /intelligence/ask related-edges query must filter both repo joins
    AND the lateral repo_embeddings subquery on is_private = false."""
    src = _read_source("app/routers/intelligence.py")

    # Extract the related-edges SQL block (between the marker comment and the
    # closing parenthesis of the text("""...""") literal).
    match = re.search(
        r"SELECT e1\.repo_id::text AS source_id,.*?\)""\),",
        src,
        re.DOTALL,
    )
    assert match, "related-edges SQL block not found in intelligence.py"
    sql = match.group(0)

    # The two outer JOINs MUST carry the filter inline so a future edit can't
    # silently drop one — using a separate WHERE clause is fine, but inline
    # is what the fix put in place.
    assert "JOIN repos r1 ON r1.id = e1.repo_id AND r1.is_private = false" in sql, (
        "intelligence.py related-edges r1 join lost is_private = false filter"
    )
    assert "JOIN repos r2 ON r2.id = e2.repo_id AND r2.is_private = false" in sql, (
        "intelligence.py related-edges r2 join lost is_private = false filter"
    )
    # The lateral subquery has its own JOIN on repos (r_inner) — without it,
    # private rows can be picked as nearest neighbours and only filtered out
    # at the outer JOIN, distorting result counts.
    assert "r_inner.is_private = false" in sql, (
        "intelligence.py related-edges lateral subquery lost the inner "
        "is_private filter — private repos can be selected as similar "
        "neighbours then filtered, distorting LIMIT semantics"
    )


def test_recommendations_similar_sql_has_seed_privacy_filter():
    """The /intelligence/similar/{name} SQL must filter both seed_r AND r."""
    src = _read_source("app/routers/recommendations.py")
    assert "seed_r.is_private = false" in src, (
        "recommendations.py _SIMILAR_SQL lost seed_r.is_private = false — "
        "a private repo as seed could leak public neighbours back to caller"
    )
    assert "r.is_private = false" in src, (
        "recommendations.py _SIMILAR_SQL lost r.is_private = false — "
        "a public seed could surface private neighbours"
    )


def test_mentions_endpoint_uses_central_helper():
    """GET /repos/{id}/mentions must filter via the centralized predicate."""
    src = _read_source("app/routers/mentions.py")
    assert "from app.db_filters import public_repo_filter" in src, (
        "mentions.py must import public_repo_filter from app.db_filters"
    )
    assert "public_repo_filter()" in src, (
        "mentions.py must use public_repo_filter() in the repo lookup so a "
        "holder of a private UUID cannot use the endpoint as an oracle"
    )


def test_dependencies_endpoints_use_central_helper():
    """Both /repos/{id}/dependencies AND /dependencies/dependents must filter."""
    src = _read_source("app/routers/dependencies.py")
    assert "from app.db_filters import public_repo_filter" in src
    # Two callsites — the per-repo lookup and the dependents query.
    assert src.count("public_repo_filter()") >= 2, (
        "dependencies.py expected to filter both /repos/{id}/dependencies "
        "AND /dependencies/dependents — the second is a confirmed P0 leak "
        "vector (anyone-can-query → returns private repos)"
    )
