"""
Locust load test suite for reporium-api.

Usage examples are documented in tests/load/README.md.
"""

from __future__ import annotations

from locust import HttpUser, between, task


class ReporiumUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def library_full(self) -> None:
        self.client.get(
            "/library/full?page=1&pageSize=200",
            name="GET /library/full",
        )

    @task(2)
    def library(self) -> None:
        self.client.get(
            "/library",
            name="GET /library",
        )

    @task(2)
    def stats(self) -> None:
        self.client.get(
            "/stats",
            name="GET /stats",
        )

    @task(1)
    def intelligence_ask(self) -> None:
        self.client.post(
            "/intelligence/ask",
            name="POST /intelligence/ask",
            json={"question": "what AI tools do you use for testing?"},
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )

    @task(1)
    def health(self) -> None:
        self.client.get(
            "/health",
            name="GET /health",
        )
