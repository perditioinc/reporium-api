"""reporium#433 item #17: local, $0 groundedness/faithfulness eval for /ask.

These tests exercise ``app.eval.groundedness``, the reporium-side adapter over
the merged ``local-inference`` verifier (HHEM offline cross-encoder primary,
local Ollama 7B NLI fallback). They are CI-SAFE: if neither a local backend is
available NOR the ``local-inference`` package is installed, the verifier tests
skip cleanly instead of failing. The threshold / config logic is pure and runs
everywhere.

Why this matters: a low groundedness score means the /ask answer asserted facts
the retrieved repos do not support -- i.e. the model hallucinated. Scoring it
locally and for $0 is the gate that lets a hallucinated answer be caught before
it is trusted.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from app.eval.groundedness import (
    DEFAULT_GROUNDEDNESS_THRESHOLD,
    AnswerGroundedness,
    grade_answer,
    verifier_available,
)
from app.eval import groundedness as ge


# ---------------------------------------------------------------------------
# Pure / config logic -- runs in any environment (no model, no DB).
# ---------------------------------------------------------------------------

def test_threshold_default_when_unset():
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("ASK_GROUNDEDNESS_THRESHOLD", None)
        assert ge._threshold(None) == DEFAULT_GROUNDEDNESS_THRESHOLD


def test_threshold_explicit_overrides_env():
    with patch.dict(os.environ, {"ASK_GROUNDEDNESS_THRESHOLD": "0.9"}):
        # An explicit arg wins over the env var.
        assert ge._threshold(0.3) == 0.3


def test_threshold_from_env():
    with patch.dict(os.environ, {"ASK_GROUNDEDNESS_THRESHOLD": "0.75"}):
        assert ge._threshold(None) == 0.75


def test_threshold_bad_env_falls_back_to_default():
    with patch.dict(os.environ, {"ASK_GROUNDEDNESS_THRESHOLD": "not-a-number"}):
        assert ge._threshold(None) == DEFAULT_GROUNDEDNESS_THRESHOLD


def test_verifier_available_is_non_raising_bool():
    """Must never raise, even if local-inference is absent -- it's the CI gate."""
    val = verifier_available()
    assert isinstance(val, bool)


def test_grade_answer_threshold_applied_with_stubbed_backend():
    """grade_answer applies the threshold to the backend's score. Stub the
    local-inference scorer so this runs without any model."""

    class _StubResult:
        score = 0.8
        backend = "hhem"
        latency_ms = 5.0
        rationale = None

    class _StubScorer:
        def score(self, context, answer, timeout=60.0):
            return _StubResult()

    with patch.object(ge, "_get_scorer", return_value=_StubScorer()):
        # 0.8 >= 0.5 -> grounded
        r = grade_answer("ctx", "ans", threshold=0.5)
        assert isinstance(r, AnswerGroundedness)
        assert r.score == 0.8
        assert r.grounded is True
        assert r.backend == "hhem"
        # 0.8 < 0.9 -> not grounded
        r2 = grade_answer("ctx", "ans", threshold=0.9)
        assert r2.grounded is False


# ---------------------------------------------------------------------------
# Real verifier -- skipped when no local backend is available (CI-safe).
# ---------------------------------------------------------------------------

requires_verifier = pytest.mark.skipif(
    not verifier_available(),
    reason="no local groundedness backend (HHEM weights uncached AND Ollama unreachable)",
)


# A realistic /ask-shaped retrieved-context block.
_CONTEXT = (
    "vLLM (vllm-project/vllm) is a high-throughput, memory-efficient inference "
    "engine for large language models. It uses PagedAttention to manage the KV "
    "cache and supports continuous batching. It exposes an OpenAI-compatible "
    "HTTP server and is written in Python and CUDA."
)
_GROUNDED = (
    "vLLM is a high-throughput LLM inference engine that uses PagedAttention "
    "and continuous batching and exposes an OpenAI-compatible HTTP server."
)
_HALLUCINATED = (
    "vLLM is a Rust database built by Google in 2012 to store passwords, and it "
    "has 5 million GitHub stars."
)


@requires_verifier
def test_grounded_answer_scores_high():
    r = grade_answer(_CONTEXT, _GROUNDED)
    assert r.backend in ("hhem", "ollama")
    # A faithful answer must clear the default decision threshold.
    assert r.score >= DEFAULT_GROUNDEDNESS_THRESHOLD, r
    assert r.grounded is True


@requires_verifier
def test_hallucinated_answer_scores_low():
    r = grade_answer(_CONTEXT, _HALLUCINATED)
    # A fabricated answer must fall below the threshold.
    assert r.score < DEFAULT_GROUNDEDNESS_THRESHOLD, r
    assert r.grounded is False


@requires_verifier
def test_grounded_strictly_beats_hallucinated():
    """The core property the hallucination gate relies on: a faithful answer
    scores strictly higher than a fabricated one over the same context."""
    g = grade_answer(_CONTEXT, _GROUNDED)
    h = grade_answer(_CONTEXT, _HALLUCINATED)
    assert g.score > h.score, (g, h)
    # And the gap is meaningful, not noise.
    assert (g.score - h.score) > 0.2, (g.score, h.score)
