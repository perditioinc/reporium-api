"""Local, $0 evaluation helpers for reporium-api.

Currently exposes a groundedness / faithfulness scorer for /intelligence/ask
answers (reporium#433, item #17). All backends are local (HHEM offline cross
encoder, or the local Ollama 7B as a fallback NLI judge); no frontier model
and no paid API are ever called.
"""
