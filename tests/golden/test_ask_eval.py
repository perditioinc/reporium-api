"""
Ask Eval Harness — Sprint 0 baseline measurement.

Purpose
-------
Establishes a repeatable, numerical baseline for the Reporium ``/intelligence/ask``
endpoint BEFORE Sprints 1-3 introduce the router layer, semantic cache, and
evidence packs. Every future sprint will rerun this harness and must prove a
measurable delta (latency, hit rate, or cost) versus the recorded baseline.

Execution policy
----------------
This test is **gated behind an env var** and is NOT part of the default pytest
run. Reason: it issues real network calls to a running API (and transitively to
Anthropic), which makes it slow, flaky, and expensive. Enable it explicitly:

    RUN_GOLDEN_EVAL=1 \\
    ASK_EVAL_BASE_URL=http://localhost:8000 \\
    ASK_EVAL_APP_TOKEN=<your-X-App-Token> \\
        pytest tests/golden/test_ask_eval.py -v -s

When ``RUN_GOLDEN_EVAL`` is unset, the entire module is skipped at collection
time so CI default runs stay cheap.

What this measures (Sprint 0)
-----------------------------
For each question in ``ask_questions.yaml``:
    * ``must_mention`` hit rate — fraction of required substrings found in the
      lowercased answer. Aggregated into a global mention-hit rate.
    * ``must_not_mention`` violations — any hit is logged but non-fatal (we
      report, we don't fail the suite on Sprint 0).
    * Latency (ms) — wall-clock time for the HTTP round-trip.
    * Response size (bytes) — a coarse token proxy until we wire the structured
      ``tokens_used`` numbers through (already present in the response, so we
      record that too when available).

We intentionally do NOT assert ``expected_route`` in Sprint 0 — the router
layer does not exist yet. That field is carried through untouched so Sprint 1
can enable a route-accuracy assertion with zero edits to this file.

Artifact
--------
Results are written to ``tests/golden/.results/latest.json`` (gitignored) for
downstream diffing. The pytest session summary prints pass/fail counts,
median latency, and mention-hit rate.
"""
from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

# ---------------------------------------------------------------------------
# Module-level skip — do not run unless explicitly opted in.
# ---------------------------------------------------------------------------
_RUN = os.getenv("RUN_GOLDEN_EVAL") == "1"

pytestmark = pytest.mark.skipif(
    not _RUN,
    reason=(
        "Golden-set eval is gated behind RUN_GOLDEN_EVAL=1 because it issues "
        "real LLM calls (slow + costly). Set RUN_GOLDEN_EVAL=1, "
        "ASK_EVAL_BASE_URL, and ASK_EVAL_APP_TOKEN to enable."
    ),
)

# httpx is already a dev dep (used elsewhere in tests/). Import lazily after
# the skip guard so collection doesn't fail in environments that don't have it
# installed for unrelated reasons.
import httpx  # noqa: E402

GOLDEN_PATH = Path(__file__).parent / "ask_questions.yaml"
RESULTS_DIR = Path(__file__).parent / ".results"
RESULTS_PATH = RESULTS_DIR / "latest.json"

ASK_PATH = "/intelligence/ask"
DEFAULT_TIMEOUT_S = 60.0

# ---------------------------------------------------------------------------
# Rate-limit pacing
# ---------------------------------------------------------------------------
# The `/intelligence/ask` endpoint is rate-limited at 6/minute, 60/day per IP.
# Fire-and-forget in a tight loop would trip the limiter and record false
# failures, so the runner paces itself and optionally caps the question count.
#
#   ASK_EVAL_SLEEP_SECONDS  seconds to sleep between requests (default 11;
#                           headroom under the 6/min limit). Set to 0 to
#                           opt out entirely (e.g. local runs against a mock).
#   ASK_EVAL_MAX_QUESTIONS  cap on how many questions the run sends. Unset
#                           = unlimited. Useful for a cheap smoke pass
#                           before spending the full daily quota.
_SLEEP = float(os.environ.get("ASK_EVAL_SLEEP_SECONDS", "11"))
_MAX = os.environ.get("ASK_EVAL_MAX_QUESTIONS")
_MAX_INT = int(_MAX) if _MAX else None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_questions() -> list[dict[str, Any]]:
    with GOLDEN_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, list):
        raise ValueError(f"{GOLDEN_PATH} must be a YAML list of question dicts")
    return data


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _mention_hits(answer: str, needles: list[str]) -> tuple[int, int]:
    """Return (hits, total) for substring membership in lowercased answer."""
    if not needles:
        return (0, 0)
    lc = (answer or "").lower()
    hits = sum(1 for n in needles if str(n).lower() in lc)
    return (hits, len(needles))


def _forbidden_hits(answer: str, needles: list[str]) -> list[str]:
    if not needles:
        return []
    lc = (answer or "").lower()
    return [n for n in needles if str(n).lower() in lc]


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def _make_client() -> httpx.Client:
    base_url = os.getenv("ASK_EVAL_BASE_URL", "http://localhost:8000")
    token = os.getenv("ASK_EVAL_APP_TOKEN", "")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-App-Token"] = token
    return httpx.Client(base_url=base_url, headers=headers, timeout=DEFAULT_TIMEOUT_S)


def _ask(client: httpx.Client, question: str) -> tuple[dict[str, Any], int, float, int]:
    """POST /intelligence/ask. Returns (payload, status_code, latency_ms, raw_bytes)."""
    body = {"question": question}
    t0 = time.perf_counter()
    resp = client.post(ASK_PATH, json=body)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    raw = resp.content or b""
    try:
        payload = resp.json() if raw else {}
    except json.JSONDecodeError:
        payload = {"_raw": raw.decode("utf-8", errors="replace")[:2000]}
    return payload, resp.status_code, latency_ms, len(raw)


# ---------------------------------------------------------------------------
# Main eval
# ---------------------------------------------------------------------------

def test_ask_golden_eval_baseline():
    """Run the full golden set and emit a baseline results artifact.

    Sprint 0 policy: the suite PASSES as long as every HTTP call returned
    200. We surface mention-hit rate and latency stats in the summary but
    do not assert a quality floor yet — that threshold is Sprint 1's job,
    once we have a baseline number to choose it from.
    """
    questions = _load_questions()
    # Sprint 0 shipped 50 entries; Sprint 1 expanded to 120. Assert a floor
    # rather than an exact count so adding legitimate new cases doesn't break
    # the eval — drift downward (accidental deletions) still fails loudly.
    assert len(questions) >= 120, f"Expected >=120 questions, got {len(questions)}"

    # Apply the optional cap BEFORE iterating so summary counts match what we
    # actually sent. The full YAML is still validated above.
    if _MAX_INT is not None:
        questions = questions[:_MAX_INT]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    per_entry: list[dict[str, Any]] = []
    http_failures: list[dict[str, Any]] = []

    started_at = time.time()

    print(
        f"[eval] pacing: {_SLEEP}s between requests, "
        f"max={_MAX_INT if _MAX_INT is not None else 'unlimited'}"
    )

    with _make_client() as client:
        for idx, q in enumerate(questions):
            qid = q.get("id", "?")
            question = q.get("question", "")
            category = q.get("category", "?")
            must_mention = list(q.get("must_mention") or [])
            must_not_mention = list(q.get("must_not_mention") or [])

            try:
                payload, status, latency_ms, nbytes = _ask(client, question)
            except httpx.HTTPError as e:
                http_failures.append(
                    {"id": qid, "question": question, "error": repr(e)}
                )
                per_entry.append(
                    {
                        "id": qid,
                        "category": category,
                        "question": question,
                        "status": None,
                        "latency_ms": None,
                        "response_bytes": 0,
                        "mention_hits": 0,
                        "mention_total": len(must_mention),
                        "forbidden_hits": [],
                        "tokens_used": None,
                        "error": repr(e),
                    }
                )
                # Still pace after a transport error so repeated failures
                # don't hammer the endpoint.
                if _SLEEP > 0 and idx < len(questions) - 1:
                    time.sleep(_SLEEP)
                continue

            answer = (payload or {}).get("answer", "") or ""
            tokens_used = (payload or {}).get("tokens_used")

            hits, total = _mention_hits(answer, must_mention)
            forbidden = _forbidden_hits(answer, must_not_mention)

            if status != 200:
                http_failures.append(
                    {"id": qid, "question": question, "status": status,
                     "body": json.dumps(payload)[:500]}
                )

            per_entry.append(
                {
                    "id": qid,
                    "category": category,
                    "question": question,
                    "status": status,
                    "latency_ms": round(latency_ms, 1),
                    "response_bytes": nbytes,
                    "mention_hits": hits,
                    "mention_total": total,
                    "forbidden_hits": forbidden,
                    "tokens_used": tokens_used,
                    "answer_len": len(answer),
                }
            )

            # Pace ourselves under the 6/min, 60/day per-IP rate limit.
            # Skip the trailing sleep after the last question.
            if _SLEEP > 0 and idx < len(questions) - 1:
                time.sleep(_SLEEP)

    finished_at = time.time()

    # -----------------------------------------------------------------------
    # Aggregates
    # -----------------------------------------------------------------------
    ok_entries = [e for e in per_entry if e.get("status") == 200]
    latencies = [e["latency_ms"] for e in ok_entries if e.get("latency_ms") is not None]
    total_hits = sum(e["mention_hits"] for e in per_entry)
    total_needles = sum(e["mention_total"] for e in per_entry)
    mention_hit_rate = (total_hits / total_needles) if total_needles else None
    forbidden_violations = sum(1 for e in per_entry if e["forbidden_hits"])

    summary = {
        "version": "sprint0",
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_s": round(finished_at - started_at, 2),
        "n_questions": len(per_entry),
        "n_http_ok": len(ok_entries),
        "n_http_fail": len(per_entry) - len(ok_entries),
        "mention_hit_rate": mention_hit_rate,
        "forbidden_violations": forbidden_violations,
        "latency_ms": {
            "p50": round(statistics.median(latencies), 1) if latencies else None,
            "p95": round(_percentile(latencies, 95), 1) if latencies else None,
            "max": round(max(latencies), 1) if latencies else None,
            "mean": round(statistics.fmean(latencies), 1) if latencies else None,
        },
        "entries": per_entry,
    }

    RESULTS_PATH.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    # -----------------------------------------------------------------------
    # Printable summary (captured by `-s`)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("ASK GOLDEN-SET EVAL — Sprint 0 baseline")
    print("=" * 88)
    print(f"{'id':<5} {'cat':<10} {'status':>6} {'ms':>8} {'bytes':>7} {'hits':>7}  question")
    print("-" * 88)
    for e in per_entry:
        print(
            f"{e['id']:<5} {e['category']:<10} "
            f"{str(e.get('status') or '-'):>6} "
            f"{(e.get('latency_ms') or 0):>8.1f} "
            f"{e.get('response_bytes', 0):>7} "
            f"{e['mention_hits']}/{e['mention_total']:<5} "
            f"{(e['question'] or '')[:50]}"
        )
    print("-" * 88)
    print(
        f"n={summary['n_questions']}  "
        f"ok={summary['n_http_ok']}  "
        f"fail={summary['n_http_fail']}  "
        f"hit_rate={summary['mention_hit_rate']}  "
        f"forbidden={summary['forbidden_violations']}"
    )
    lat = summary["latency_ms"]
    print(
        f"latency ms  p50={lat['p50']}  p95={lat['p95']}  "
        f"mean={lat['mean']}  max={lat['max']}"
    )
    print(f"artifact: {RESULTS_PATH}")
    print("=" * 88)

    # -----------------------------------------------------------------------
    # Assertions — Sprint 0 is a measurement sprint, not a quality gate.
    # We only fail the run if the endpoint itself is broken.
    # -----------------------------------------------------------------------
    assert not http_failures, (
        f"{len(http_failures)} question(s) returned non-200 or errored — "
        f"eval cannot baseline against a broken endpoint. First: {http_failures[0]}"
    )


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] + (s[hi] - s[lo]) * frac
