"""Tests for CORS middleware — verifies allowed and blocked origins."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_cors_allowed_origin_reporium(client: AsyncClient):
    """reporium.com must be allowed."""
    response = await client.get("/health", headers={"Origin": "https://reporium.com"})
    assert response.headers.get("access-control-allow-origin") == "https://reporium.com"


@pytest.mark.asyncio
async def test_cors_allowed_origin_www(client: AsyncClient):
    """www.reporium.com must be allowed."""
    response = await client.get("/health", headers={"Origin": "https://www.reporium.com"})
    assert response.headers.get("access-control-allow-origin") == "https://www.reporium.com"


@pytest.mark.asyncio
async def test_cors_allowed_origin_github_pages(client: AsyncClient):
    """perditioinc.github.io must be allowed."""
    response = await client.get("/health", headers={"Origin": "https://perditioinc.github.io"})
    assert response.headers.get("access-control-allow-origin") == "https://perditioinc.github.io"


@pytest.mark.asyncio
async def test_cors_blocked_unknown_origin(client: AsyncClient):
    """Unknown origins must not receive allow-origin header."""
    response = await client.get("/health", headers={"Origin": "https://evil.example.com"})
    assert response.headers.get("access-control-allow-origin") is None
