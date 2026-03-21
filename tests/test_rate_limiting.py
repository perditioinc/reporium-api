"""Tests for rate limiting middleware."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_rate_limit_headers_present(client: AsyncClient):
    """Every response must include rate limit policy header."""
    response = await client.get("/health")
    assert response.status_code == 200
    assert "X-RateLimit-Policy" in response.headers
    assert response.headers["X-RateLimit-Policy"] == "200/hour;30/minute"


@pytest.mark.asyncio
async def test_health_always_returns_ok(client: AsyncClient):
    """Health endpoint must always return 200."""
    for _ in range(5):
        response = await client.get("/health")
        assert response.status_code == 200
