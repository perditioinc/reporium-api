"""
Reusable Locust harness for reporium-api.

Usage examples are documented in tests/load/README.md.
"""

from __future__ import annotations

import os

from locust import HttpUser, between, task


INTELLIGENCE_ASK_PATH = os.getenv("LOAD_INTELLIGENCE_ASK_PATH", "/intelligence/ask")
ASK_QUESTION = os.getenv(
    "LOAD_INTELLIGENCE_QUESTION",
    "What repositories help with AI evaluation and observability?"
)
ASK_TOP_K = int(os.getenv("LOAD_INTELLIGENCE_TOP_K", "8"))
OPTIONAL_BEARER_TOKEN = os.getenv("LOAD_TEST_BEARER_TOKEN")


def _json_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if OPTIONAL_BEARER_TOKEN:
        headers["Authorization"] = f"Bearer {OPTIONAL_BEARER_TOKEN}"
    return headers


class ReporiumApiUser(HttpUser):
    wait_time = between(1, 1)

    @task(3)
    def library_full(self) -> None:
        self.client.get(
            "/library/full",
            name="GET /library/full",
            headers={"Accept": "application/json"},
        )

    @task(1)
    def intelligence_ask(self) -> None:
        self.client.post(
            INTELLIGENCE_ASK_PATH,
            name=f"POST {INTELLIGENCE_ASK_PATH}",
            json={
                "question": ASK_QUESTION,
                "top_k": ASK_TOP_K,
            },
            headers=_json_headers(),
        )
