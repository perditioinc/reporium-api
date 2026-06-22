"""Local, $0 groundedness / faithfulness eval for /intelligence/ask answers.

reporium#433 (item #17). Given the retrieved context that an /ask answer was
generated from, score how well the answer is *grounded in* (entailed by) that
context. A low score means the model likely hallucinated -- it asserted things
the retrieved repos do not support. This is the hallucination gate that lets an
answer be measured (and, downstream, gated) before it is trusted.

Design choices
--------------
* DOGFOODS the merged ``local-inference`` package: it owns the actual verifier
  (Vectara HHEM-2.1-Open offline cross-encoder as the primary backend, falling
  back to the local Ollama 7B as an NLI judge). We do NOT reimplement scoring
  here -- this module is a thin, reporium-shaped adapter over
  ``local_inference.eval.groundedness.GroundednessScorer``.
* $0 ONLY. Every backend is local hardware. No frontier model, no paid API.
* CI-SAFE. Importing this module never raises and never loads heavy weights.
  ``verifier_available()`` reports whether a local backend can serve a request,
  so tests and CI can ``skip`` cleanly when neither HHEM weights nor a reachable
  Ollama are present (e.g. on a GitHub-hosted runner with no model cache).

Typical use::

    from app.eval.groundedness import grade_answer, verifier_available

    if verifier_available():
        result = grade_answer(context_text, answer_text)
        print(result.score, result.grounded, result.backend)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Optional

# A claim scoring >= this is treated as grounded. Mirrors local-inference's
# DEFAULT_THRESHOLD but is overridable per-call and via env for A/B work.
DEFAULT_GROUNDEDNESS_THRESHOLD = 0.5

Backend = Literal["hhem", "ollama", "auto"]


@dataclass(frozen=True)
class AnswerGroundedness:
    """Groundedness verdict for a single (context, answer) pair.

    ``score`` is P(answer grounded in context) in [0, 1] (1 = fully supported,
    0 = contradicted/unsupported). ``grounded`` applies the threshold.
    ``backend`` records which local $0 verifier served it ('hhem' or 'ollama').
    ``rationale`` is populated only by the Ollama judge. ``latency_ms`` is the
    per-pair scoring latency.
    """

    score: float
    grounded: bool
    backend: str
    threshold: float
    latency_ms: float
    rationale: Optional[str] = None


def _threshold(explicit: float | None) -> float:
    if explicit is not None:
        return explicit
    raw = os.getenv("ASK_GROUNDEDNESS_THRESHOLD", "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return DEFAULT_GROUNDEDNESS_THRESHOLD


def verifier_available(backend: Backend = "auto") -> bool:
    """True iff a local $0 groundedness backend can serve a request right now.

    Tries to construct the scorer (which probes the HHEM cache and/or the
    Ollama endpoint) and reports success. Never raises -- safe to call at the
    top of a CI-skipped test. Construction is cheap when HHEM weights are
    cached; the heavy model load only happens on first ``score``.
    """
    try:
        _get_scorer(backend)
        return True
    except Exception:
        return False


# Module-level cache so repeated calls reuse the loaded HHEM weights.
_SCORER = None
_SCORER_BACKEND: str | None = None


def _get_scorer(backend: Backend = "auto"):
    """Return a cached local-inference GroundednessScorer.

    Raises if the local-inference package is not installed or no local backend
    is available; callers convert that into a skip / non-fatal path.
    """
    global _SCORER, _SCORER_BACKEND
    if _SCORER is not None and _SCORER_BACKEND == backend:
        return _SCORER
    # Imported lazily so this module imports with zero heavy deps and CI that
    # has not installed local-inference still collects the test (then skips).
    from local_inference.eval.groundedness import GroundednessScorer  # noqa: PLC0415

    _SCORER = GroundednessScorer(backend)
    _SCORER_BACKEND = backend
    return _SCORER


def grade_answer(
    context: str,
    answer: str,
    *,
    threshold: float | None = None,
    backend: Backend = "auto",
    timeout: float = 60.0,
) -> AnswerGroundedness:
    """Score how grounded ``answer`` is in ``context`` using a local $0 verifier.

    ``context`` is the retrieved-context block the /ask answer was generated
    from (the same text fed to the LLM); ``answer`` is the model's reply. Higher
    score == more faithful. Raises ``GroundednessError`` (from local-inference)
    if no local backend is available -- gate on ``verifier_available()`` first
    in CI.
    """
    scorer = _get_scorer(backend)
    th = _threshold(threshold)
    res = scorer.score(context, answer, timeout=timeout)
    return AnswerGroundedness(
        score=res.score,
        grounded=res.score >= th,
        backend=res.backend,
        threshold=th,
        latency_ms=res.latency_ms,
        rationale=res.rationale,
    )


__all__ = [
    "AnswerGroundedness",
    "grade_answer",
    "verifier_available",
    "DEFAULT_GROUNDEDNESS_THRESHOLD",
]
