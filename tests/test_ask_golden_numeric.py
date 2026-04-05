"""
Numeric golden-set quality gate for POST /intelligence/ask.

This test exists so that cost-cutting PRs (prompt trimming, model downgrades,
top_k reductions, caching tweaks, etc.) can be validated against a stable
numeric quality threshold rather than only response-shape assertions.

How it works
------------
1. Loads ``tests/golden_set_ask.yaml`` — a handcrafted set of 15-18 Q&A pairs
   grounded in the real Reporium corpus.
2. For each entry we:
     - Build a mocked DB session that returns the entry's ``fixture_repos``
       from the pgvector similarity query and empty results for the semantic
       cache lookup and the knowledge-graph edge query (mirroring the pattern
       in ``tests/test_intelligence_quality.py``).
     - Patch the embedding model to a zero vector (mocked DB means the vector
       is never actually used server-side).
     - Call the real ``/intelligence/ask`` handler via ``AsyncClient``, which
       triggers a **real** Anthropic call (Haiku/Sonnet as the router chooses).
     - Score the returned answer with ``_score_entry``.
3. Asserts:
     - Average ``quality_score`` across all scored entries is ``>= 0.7``.
     - Total tokens across the suite is ``<= 1.2x`` the sum of per-entry
       ``max_tokens_soft_budget`` values.
     - Every ``expect_status`` edge case returns the expected HTTP code.

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

    pytest tests/test_ask_golden_numeric.py -v
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
import yaml
from httpx import ASGITransport, AsyncClient

pytestmark = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set — golden-set numeric gate requires live Claude access",
)


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
    return row


def _make_mock_db(rows: list[MagicMock]) -> AsyncMock:
    """Three-call mock: semantic-cache miss, similarity fetchall, edges fetchall."""
    cache_result = MagicMock()
    cache_result.first.return_value = None

    sim_result = MagicMock()
    sim_result.fetchall.return_value = rows

    edges_result = MagicMock()
    edges_result.fetchall.return_value = []

    mock_db = AsyncMock()
    # The handler may execute additional queries (logging, session turns, etc).
    # We return a permissive default after the three expected calls so nothing
    # downstream raises StopIteration.
    default_result = MagicMock()
    default_result.fetchall.return_value = []
    default_result.first.return_value = None

    call_sequence = [cache_result, sim_result, edges_result]

    async def _execute(*_args, **_kwargs):
        if call_sequence:
            return call_sequence.pop(0)
        return default_result

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

def _load_golden_set() -> list[dict[str, Any]]:
    with GOLDEN_SET_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, list):
        raise ValueError(f"{GOLDEN_SET_PATH} must contain a YAML list")
    return [e for e in data if not e.get("skip")]


@pytest.mark.asyncio
async def test_ask_golden_set_numeric_gate(client_no_db: AsyncClient):
    """Aggregate numeric quality gate for /intelligence/ask."""
    from app.main import app
    from app.database import get_db

    golden_set = _load_golden_set()
    assert len(golden_set) >= 15, (
        f"Golden set must contain >= 15 entries, got {len(golden_set)}"
    )

    scored_results: list[dict[str, Any]] = []
    status_results: list[dict[str, Any]] = []
    total_tokens = 0
    total_budget = 0

    for idx, entry in enumerate(golden_set, start=1):
        question = entry.get("question", "")
        expect_status = entry.get("expect_status")
        budget = int(entry.get("max_tokens_soft_budget") or 0)

        rows = [_make_db_row(r) for r in (entry.get("fixture_repos") or [])]
        mock_db = _make_mock_db(rows)

        async def _override_db():
            yield mock_db

        app.dependency_overrides[get_db] = _override_db

        try:
            with (
                _patch_embedding_model(),
                _patch_log_query(),
                _patch_create_task(),
            ):
                response = await client_no_db.post(
                    "/intelligence/ask",
                    json={"question": question},
                )
        finally:
            app.dependency_overrides.pop(get_db, None)

        if expect_status is not None:
            status_results.append(
                {
                    "idx": idx,
                    "question": (question or "<empty>")[:60],
                    "expected": expect_status,
                    "got": response.status_code,
                    "pass": response.status_code == expect_status,
                }
            )
            continue

        assert response.status_code == 200, (
            f"[{idx}] Q={question!r} — expected 200, got "
            f"{response.status_code}: {response.text[:300]}"
        )

        data = response.json()
        answer = data.get("answer", "") or ""
        sources = data.get("sources") or []
        tokens = data.get("tokens_used") or {}
        used = int(tokens.get("total") or 0)

        score = _score_entry(entry, answer, sources)
        total_tokens += used
        total_budget += budget

        scored_results.append(
            {
                "idx": idx,
                "question": question[:60],
                "difficulty": entry.get("difficulty", "?"),
                "score": score,
                "tokens": used,
                "budget": budget,
                "answer_len": len(answer),
                "pass": score >= 0.5,
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
    # Edge-case statuses must all match.
    bad_statuses = [r for r in status_results if not r["pass"]]
    assert not bad_statuses, f"Edge-case status mismatches: {bad_statuses}"

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
