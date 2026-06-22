#!/usr/bin/env python3
"""Local, $0 groundedness eval demo for /intelligence/ask answers (reporium#433 #17).

Runs the local groundedness verifier (HHEM offline, else Ollama fallback) over a
small fixed set of (retrieved-context, answer) pairs that mimic the /ask shape:
one GROUNDED answer (only states things the context supports) and one
HALLUCINATED answer (asserts facts the context never mentions). It prints the
real score for each and asserts the grounded answer scores strictly higher than
the hallucinated one -- the property the hallucination gate relies on.

This is a manual / CI-on-demand tool, NOT part of the default pytest run, so it
never blocks CI on model availability. Run it with the verifier present::

    python scripts/groundedness_eval.py

Exit code 0 == the verifier separated grounded from hallucinated; 2 == no local
backend available (so nothing was proven); 1 == the separation property failed.
Every backend is local hardware -- no paid API, no frontier model.
"""

from __future__ import annotations

import sys

# --- Fixtures: realistic /ask context + answers -----------------------------
# CONTEXT is the kind of retrieved-repo block the /ask prompt builds. The two
# answers are deliberately one faithful and one fabricated.
CONTEXT = (
    "vLLM (vllm-project/vllm) is a high-throughput and memory-efficient "
    "inference engine for large language models. It introduces PagedAttention "
    "to manage the KV cache and supports continuous batching of incoming "
    "requests. It is written in Python and CUDA and has over 30000 stars on "
    "GitHub. vLLM exposes an OpenAI-compatible HTTP server.\n"
    "Ollama (ollama/ollama) lets you run open large language models locally. "
    "It bundles model weights and a runtime and exposes a local REST API on "
    "port 11434."
)

GROUNDED_ANSWER = (
    "vLLM is a high-throughput inference engine for large language models that "
    "uses PagedAttention and continuous batching, and it exposes an "
    "OpenAI-compatible HTTP server. Ollama runs open LLMs locally and serves a "
    "REST API on port 11434."
)

HALLUCINATED_ANSWER = (
    "vLLM is a Rust database created by Google in 2012 for storing user "
    "passwords, and it has 5 million stars. Ollama is a paid cloud service that "
    "trains models on port 8080 and was acquired by Microsoft."
)


def main() -> int:
    try:
        from app.eval.groundedness import grade_answer, verifier_available
    except Exception as exc:  # pragma: no cover - import guard
        print(f"[skip] could not import the eval helper: {exc}")
        return 2

    if not verifier_available():
        print(
            "[skip] no local groundedness backend available "
            "(HHEM weights not cached AND Ollama at 127.0.0.1:11434 unreachable). "
            "Nothing was proven; this is expected on a bare CI runner."
        )
        return 2

    grounded = grade_answer(CONTEXT, GROUNDED_ANSWER)
    hallucinated = grade_answer(CONTEXT, HALLUCINATED_ANSWER)

    print(f"backend           : {grounded.backend}")
    print(f"threshold         : {grounded.threshold}")
    print(
        f"GROUNDED answer   : score={grounded.score:.4f} "
        f"grounded={grounded.grounded} latency={grounded.latency_ms:.0f}ms"
    )
    print(
        f"HALLUCINATED ans  : score={hallucinated.score:.4f} "
        f"grounded={hallucinated.grounded} latency={hallucinated.latency_ms:.0f}ms"
    )
    print(f"separation (g-h)  : {grounded.score - hallucinated.score:+.4f}")

    ok = grounded.score > hallucinated.score
    if ok:
        print("[pass] grounded answer scored strictly higher than the hallucinated one.")
        return 0
    print("[fail] verifier did NOT separate grounded from hallucinated.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
