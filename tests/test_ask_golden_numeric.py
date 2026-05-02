"""
Numeric golden-set quality gate for POST /intelligence/ask.

This test exists so that cost-cutting PRs (prompt trimming, model downgrades,
top_k reductions, caching tweaks, etc.) can be validated against a stable
numeric quality threshold rather than only response-shape assertions.

How it works
------------
1. Loads ``tests/golden_set_ask.yaml`` (or ``GOLDEN_SET_PATH`` override) — a
   handcrafted set of Q&A pairs grounded in the real Reporium corpus. The
   default is the full 55-entry set; CI uses ``tests/golden_set_ask_ci.yaml``
   (~7 entries) via the env var.
2. For each entry we (concurrently — see KAN-146):
     - Build a mocked DB session that returns the entry's ``fixture_repos``
       from the pgvector similarity query and empty results for the semantic
       cache lookup and the knowledge-graph edge query (mirroring the pattern
       in ``tests/test_intelligence_quality.py``).
     - Patch the embedding model to a zero vector (mocked DB means the vector
       is never actually used server-side).
     - Call the real ``/intelligence/ask`` handler via ``AsyncClient``, which
       triggers a **real** Anthropic call (Haiku/Sonnet as the router chooses).
     - Score the returned answer with ``_score_entry``.

   KAN-146 redesign: entries are dispatched concurrently via ``asyncio.gather``
   with a ``BoundedSemaphore(ASK_GATE_CONCURRENCY)`` capping live Anthropic
   connections (default 5; well under the Haiku tier 1 ~50 RPM limit). The
   serial 30-minute loop that timed out CI (run 25247151618) is gone.

3. Asserts:
     - Average ``quality_score`` across all scored entries is ``>= 0.7``.
     - Total tokens across the suite is ``<= 1.2x`` the sum of per-entry
       ``max_tokens_soft_budget`` values.
     - Every ``expect_status`` edge case returns the expected HTTP code.
     - No entry's ``forbidden_repos`` list appears in ``sources``. This is a
       hard fail — used to catch retrieval bugs where a negated token (e.g.
       "alternatives to pinecone") still returns the negated product as a
       top source. See issue #365 / #367.

Scoring weights (per entry) — ``quality_score`` in ``[0, 1]``:
    0.5 * fraction of ``expected_themes`` substrings present (case-insensitive)
    0.3 * fraction of ``expected_repos`` present in ``sources``
          (full credit if ``expected_repos`` is empty/omitted)
    0.2 * answer-length band score (1.0 inside [50, 2000] chars, graded
          penalty outside)

The 0.7 threshold was chosen as a pragmatic floor:
- A perfect theme hit (0.5) + full-credit repo check (0.3) already clears 0.8.
- 0.7 allows ~one missing theme per answer while still flagging regressions
  where the model drops a required concept or the prompt loses the source
  grounding entirely.
- It is intentionally below what a healthy production run should score
  (~0.85+) so cost-cutting tweaks have room to move without flapping, but
  cannot silently gut answer quality.

Running
-------
Requires ``ANTHROPIC_API_KEY`` in the environment. Skips automatically if
unset (e.g. on forks without secrets).

    # Default — full set:
    pytest tests/test_ask_golden_numeric.py -v

    # Slim CI subset:
    GOLDEN_SET_PATH=tests/golden_set_ask_ci.yaml \\
        pytest tests/test_ask_golden_numeric.py -v
"""
from __future__ import annotations

import asyncio
import contextvars
import os
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
import yaml
from httpx import ASGITransport, AsyncClient

# KAN-146: contextvar carries the per-coroutine mock DB so a single
# process-global `dependency_overrides[get_db]` can route correctly under
# `asyncio.gather`. Without this, the last writer of `dependency_overrides`
# wins and concurrent in-flight requests all see the wrong fixture data.
_CURRENT_MOCK_DB: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "_CURRENT_MOCK_DB", default=None
)

pytestmark = [
    pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not set — golden-set numeric gate requires live Claude access",
    ),
]


GOLDEN_SET_PATH = Path(__file__).parent / "golden_set_ask.yaml"


# ---------------------------------------------------------------------------
# Fixture / mocking helpers (mirrored from test_intelligence_quality.py)
# ---------------------------------------------------------------------------

def _make_db_row(entry: dict[str, Any]) -> MagicMock:
    row = MagicMock()
    row.id = str(uuid.uuid4())
    row.name = entry["name"]
    row.owner = entry["owner"]
    row.forked_from = f"{entry['owner']}/{entry['name']}"
    row.description = entry.get("description") or ""
    row.parent_stars = entry.get("stars", 100)
    row.readme_summary = entry.get("readme_summary") or f"Summary for {entry['name']}"
    row.problem_solved = entry.get("problem_solved") or ""
    row.integration_tags = entry.get("integration_tags") or []
    row.dependencies = entry.get("dependencies") or []
    row.similarity = float(entry.get("similarity", 0.85))
    # KAN-146: explicit defaults for fields read by `_prepare_query` /
    # `_build_sources_block`. Without these, MagicMock auto-vivifies the
    # attributes, the values flow into the sources-block builder, and
    # `', '.join([..., MagicMock()])` raises TypeError. The serial
    # version of this test was skipped when these fields were added to
    # the prompt context, so the defect was masked until KAN-146 turned
    # the gate back on. See `_build_sources_block` in
    # app/routers/intelligence.py.
    row.primary_category = entry.get("primary_category")
    row.language = entry.get("language")
    row.license_spdx = entry.get("license_spdx")
    row.activity_score = entry.get("activity_score")
    row.has_tests = entry.get("has_tests")
    row.has_ci = entry.get("has_ci")
    row.pros_cons = entry.get("pros_cons")
    row.community_health_pct = entry.get("community_health_pct")
    row.contributors_count = entry.get("contributors_count")
    row.issue_close_rate = entry.get("issue_close_rate")
    row.pr_merge_rate = entry.get("pr_merge_rate")
    return row


def _make_mock_db(rows: list[MagicMock]) -> AsyncMock:
    """Mock DB that dispatches results by inspecting the SQL text.

    The original 3-call sequence (cache-first, similarity-fetchall, edges-
    fetchall) silently failed when `_try_smart_route` issued an extra DB
    query — that consumed the cache slot, then the semantic-cache call got
    the similarity result, `result.first()` auto-vivified into a MagicMock,
    and the handler treated it as a cache hit with MagicMock answer/model
    (Pydantic ValidationError).

    Strategy: route based on substrings in the SQL. The pgvector similarity
    query is the only one that returns rows; everything else returns
    empty / None.
    """
    sim_result = MagicMock()
    sim_result.fetchall.return_value = rows

    empty_result = MagicMock()
    empty_result.fetchall.return_value = []
    empty_result.first.return_value = None
    empty_result.scalar.return_value = 0

    async def _execute(stmt, *_args, **_kwargs):
        # Best-effort SQL extraction: text() clauses expose the SQL via str().
        try:
            sql = str(stmt).lower()
        except Exception:
            sql = ""
        # The pgvector similarity query is the only one that uses the `<=>`
        # cosine-distance operator and orders by it. Match on the operator
        # in the ORDER BY to distinguish from the semantic-cache query
        # (which also uses `<=>` but does `LIMIT 1` and reads answer_full).
        is_similarity_query = (
            "<=>" in sql and "answer_full" not in sql
        )
        if is_similarity_query:
            return sim_result
        return empty_result

    mock_db = AsyncMock()
    mock_db.execute = AsyncMock(side_effect=_execute)
    mock_db.commit = AsyncMock()
    return mock_db


def _patch_embedding_model():
    import numpy as np
    mock_model = MagicMock()
    mock_model.encode.return_value = np.zeros(384)
    return patch("app.routers.intelligence.get_embedding_model", return_value=mock_model)


def _patch_log_query():
    return patch("app.routers.intelligence._log_query", new_callable=AsyncMock)


def _patch_create_task():
    def _noop_create_task(coro, *args, **kwargs):
        try:
            coro.close()
        except Exception:
            pass
        return MagicMock()

    return patch("app.routers.intelligence.asyncio.create_task", side_effect=_noop_create_task)


# ---------------------------------------------------------------------------
# Test client (session-scoped, no real DB)
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture(scope="session")
async def client_no_db():
    from app.main import app
    from app.database import check_db_connection  # noqa: F401

    with patch("app.main.check_db_connection", new_callable=AsyncMock):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test", timeout=120.0
        ) as ac:
            yield ac


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_entry(entry: dict[str, Any], answer: str, sources: list[dict]) -> float:
    """Return a quality score in [0, 1] for a single golden-set answer."""
    answer_lower = (answer or "").lower()

    # Theme coverage (0.5 weight)
    themes = entry.get("expected_themes") or []
    if themes:
        hits = sum(1 for t in themes if str(t).lower() in answer_lower)
        theme_score = hits / len(themes)
    else:
        theme_score = 1.0

    # Repo coverage (0.3 weight)
    expected_repos = entry.get("expected_repos") or []
    if expected_repos:
        source_slugs = {
            f"{(s.get('owner') or '').lower()}/{(s.get('name') or '').lower()}"
            for s in sources
        }
        hits = sum(1 for r in expected_repos if str(r).lower() in source_slugs)
        repo_score = hits / len(expected_repos)
    else:
        repo_score = 1.0

    # Length band (0.2 weight) — full credit inside [50, 2000] chars
    length = len(answer or "")
    if 50 <= length <= 2000:
        length_score = 1.0
    elif length < 50:
        length_score = max(0.0, length / 50.0)
    else:  # length > 2000
        # Graded penalty: 1.0 at 2000, 0.0 at 4000+
        length_score = max(0.0, 1.0 - (length - 2000) / 2000.0)

    return 0.5 * theme_score + 0.3 * repo_score + 0.2 * length_score


# ---------------------------------------------------------------------------
# Main gate
# ---------------------------------------------------------------------------

def _resolve_golden_set_path() -> Path:
    """Honour GOLDEN_SET_PATH for the slim CI subset; default to the full set."""
    override = os.getenv("GOLDEN_SET_PATH")
    if override:
        # Allow both repo-relative and absolute paths.
        p = Path(override)
        if not p.is_absolute():
            # Resolve against repo root (parent of tests/ dir).
            p = Path(__file__).parent.parent / p
        return p
    return GOLDEN_SET_PATH


def _load_golden_set() -> list[dict[str, Any]]:
    path = _resolve_golden_set_path()
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a YAML list")
    return [e for e in data if not e.get("skip")]


async def _run_one_entry(
    idx: int,
    entry: dict[str, Any],
    client: AsyncClient,
    sem: asyncio.Semaphore,
) -> dict[str, Any]:
    """Issue a single /intelligence/ask call under the concurrency cap.

    Returns a dict with the per-entry result (status, scoring, forbidden
    leak detection). The aggregate-level assertions live in the caller.

    Concurrency model: `dependency_overrides[get_db]` and the embedding /
    log_query / create_task patches are installed ONCE at the test-function
    level (see `_install_global_patches`). Per-coroutine fixture routing
    happens via the `_CURRENT_MOCK_DB` contextvar — set here, read by the
    global override. This avoids the last-writer-wins race that broke an
    earlier draft of this PR.
    """
    question = entry.get("question", "")
    expect_status = entry.get("expect_status")
    budget = int(entry.get("max_tokens_soft_budget") or 0)

    rows = [_make_db_row(r) for r in (entry.get("fixture_repos") or [])]
    mock_db = _make_mock_db(rows)

    _CURRENT_MOCK_DB.set(mock_db)

    async with sem:
        response = await client.post(
            "/intelligence/ask",
            json={"question": question},
        )

    result: dict[str, Any] = {
        "idx": idx,
        "question": question,
        "expect_status": expect_status,
        "status": response.status_code,
        "budget": budget,
    }

    if expect_status is not None:
        # Edge-case status check; no scoring.
        return result

    if response.status_code != 200:
        result["error"] = f"expected 200, got {response.status_code}: {response.text[:300]}"
        return result

    data = response.json()
    answer = data.get("answer", "") or ""
    sources = data.get("sources") or []
    tokens = data.get("tokens_used") or {}
    used = int(tokens.get("total") or 0)

    forbidden = entry.get("forbidden_repos") or []
    leaked: list[str] = []
    if forbidden:
        source_slugs = {
            f"{(s.get('owner') or '').lower()}/{(s.get('name') or '').lower()}"
            for s in sources
        }
        leaked = [f for f in forbidden if str(f).lower() in source_slugs]

    score = _score_entry(entry, answer, sources)

    result.update(
        {
            "answer_len": len(answer),
            "tokens": used,
            "score": score,
            "difficulty": entry.get("difficulty", "?"),
            "leaked": leaked,
        }
    )
    return result


@pytest.mark.timeout(600)
@pytest.mark.asyncio
async def test_ask_golden_set_numeric_gate(client_no_db: AsyncClient):
    """Aggregate numeric quality gate for /intelligence/ask."""
    from app.main import app
    from app.database import get_db

    golden_set = _load_golden_set()

    # Honour smaller floor for the slim CI subset; preserve >=50 invariant
    # for the full nightly set so silent-truncation regressions still fail.
    is_slim = bool(os.getenv("GOLDEN_SET_PATH"))
    min_entries = 5 if is_slim else 50
    assert len(golden_set) >= min_entries, (
        f"Golden set must contain >= {min_entries} entries, got {len(golden_set)}"
    )

    # KAN-146: install patches ONCE at the test-function level. Per-coroutine
    # fixture data is routed via the `_CURRENT_MOCK_DB` contextvar set inside
    # `_run_one_entry`. Doing the override per-coroutine triggered a
    # last-writer-wins race that hung the run (see PR #458 first iteration).
    async def _override_db():
        db = _CURRENT_MOCK_DB.get()
        if db is None:
            # Fallback so unrelated dependency lookups (e.g. healthcheck)
            # don't crash; never hit during scored entries.
            db = _make_mock_db([])
        yield db

    app.dependency_overrides[get_db] = _override_db

    # KAN-146: bounded-concurrency parallelism replaces serial loop. CI uses
    # slim subset (~7 entries); nightly cron uses full set.
    concurrency = int(os.getenv("ASK_GATE_CONCURRENCY", "5"))
    sem = asyncio.Semaphore(concurrency)

    try:
        # NOTE: `_patch_create_task` is intentionally NOT used in the parallel
        # version. It would patch `asyncio.create_task` globally (via the
        # `app.routers.intelligence.asyncio.create_task` dotted path), which
        # under concurrent gather can no-op tasks that FastAPI / Starlette /
        # anyio rely on internally for request handling — manifesting as a
        # deadlock that hits the per-test timeout. The fire-and-forget tasks
        # the router creates (`_log_query`, `cache.set`) are safe to run for
        # real here: `_log_query` is mocked via `_patch_log_query`, and
        # `cache.set` is a no-op when `REDIS_URL=""` (which the workflow
        # already sets).
        with (
            _patch_embedding_model(),
            _patch_log_query(),
        ):
            # `asyncio.create_task` snapshots the current contextvar context
            # per task — required so each coroutine's `_CURRENT_MOCK_DB.set(...)`
            # is visible only inside that task.
            tasks = [
                asyncio.create_task(
                    _run_one_entry(idx, entry, client_no_db, sem)
                )
                for idx, entry in enumerate(golden_set, start=1)
            ]
            results = await asyncio.gather(*tasks)
    finally:
        app.dependency_overrides.pop(get_db, None)

    # Demux into the same buckets the original serial version produced.
    scored_results: list[dict[str, Any]] = []
    status_results: list[dict[str, Any]] = []
    forbidden_violations: list[dict[str, Any]] = []
    handler_errors: list[dict[str, Any]] = []
    total_tokens = 0
    total_budget = 0

    for r in results:
        if r.get("expect_status") is not None:
            status_results.append(
                {
                    "idx": r["idx"],
                    "question": (r["question"] or "<empty>")[:60],
                    "expected": r["expect_status"],
                    "got": r["status"],
                    "pass": r["status"] == r["expect_status"],
                }
            )
            continue

        if "error" in r:
            handler_errors.append(r)
            continue

        if r.get("leaked"):
            forbidden_violations.append(
                {
                    "idx": r["idx"],
                    "question": r["question"][:60],
                    "leaked": r["leaked"],
                }
            )

        total_tokens += int(r.get("tokens") or 0)
        total_budget += int(r.get("budget") or 0)

        scored_results.append(
            {
                "idx": r["idx"],
                "question": r["question"][:60],
                "difficulty": r.get("difficulty", "?"),
                "score": r["score"],
                "tokens": r.get("tokens") or 0,
                "budget": r.get("budget") or 0,
                "answer_len": r.get("answer_len") or 0,
                "pass": r["score"] >= 0.5,
            }
        )

    # ------------------------------------------------------------------
    # Summary table — always printed so CI logs surface cost-quality data.
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("ASK GOLDEN-SET NUMERIC GATE — per-question results")
    print("=" * 90)
    print(
        f"{'#':>3} {'diff':<8} {'score':>6} {'tokens':>7} {'budget':>7} "
        f"{'len':>5}  question"
    )
    print("-" * 90)
    for r in scored_results:
        marker = "OK" if r["pass"] else "FAIL"
        print(
            f"{r['idx']:>3} {r['difficulty']:<8} {r['score']:>6.3f} "
            f"{r['tokens']:>7} {r['budget']:>7} {r['answer_len']:>5}  "
            f"[{marker}] {r['question']}"
        )

    if status_results:
        print("-" * 90)
        print("Edge-case HTTP status checks")
        print("-" * 90)
        for r in status_results:
            marker = "OK" if r["pass"] else "FAIL"
            print(
                f"{r['idx']:>3} expect={r['expected']} got={r['got']:<4} "
                f"[{marker}] {r['question']}"
            )

    avg_score = (
        sum(r["score"] for r in scored_results) / len(scored_results)
        if scored_results
        else 0.0
    )
    print("-" * 90)
    print(
        f"avg_quality_score = {avg_score:.3f}  "
        f"total_tokens = {total_tokens}  "
        f"total_budget = {total_budget}  "
        f"budget_x1.2 = {int(total_budget * 1.2)}"
    )
    print("=" * 90)

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------
    # Handler errors are unconditional fails (exception during /ask).
    assert not handler_errors, (
        "Handler error(s): "
        + "; ".join(
            f"#{e['idx']} ({(e.get('question') or '<empty>')[:60]!r}) -> {e.get('error')}"
            for e in handler_errors
        )
    )

    # Edge-case statuses must all match.
    bad_statuses = [r for r in status_results if not r["pass"]]
    assert not bad_statuses, f"Edge-case status mismatches: {bad_statuses}"

    # #367: forbidden_repos violations are hard failures. Unlike the aggregate
    # quality_score (which can absorb a few misses), a repo explicitly
    # forbidden appearing in sources means a retrieval bug — e.g. the queried
    # product itself returned as an "alternative" (#365).
    assert not forbidden_violations, (
        "forbidden_repos leaked into sources: "
        + "; ".join(
            f"#{v['idx']} ({v['question']!r}) leaked {v['leaked']}"
            for v in forbidden_violations
        )
    )

    assert scored_results, "No scored entries — golden set produced zero quality samples"

    assert avg_score >= 0.7, (
        f"Average quality_score {avg_score:.3f} < 0.7 threshold. "
        f"Per-entry: "
        + ", ".join(f"#{r['idx']}={r['score']:.2f}" for r in scored_results)
    )

    token_ceiling = int(total_budget * 1.2)
    assert total_tokens <= token_ceiling, (
        f"Total tokens {total_tokens} exceeds 1.2x soft budget ceiling "
        f"({token_ceiling}). Cost regression — investigate before merging."
    )
