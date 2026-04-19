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
async def test_cors_allowed_origin_reposhark_vercel(client: AsyncClient):
    """reposhark.vercel.app (public frontend) must be allowed."""
    response = await client.get("/health", headers={"Origin": "https://reposhark.vercel.app"})
    assert response.headers.get("access-control-allow-origin") == "https://reposhark.vercel.app"


@pytest.mark.asyncio
async def test_cors_allowed_origin_reposhark_preview(client: AsyncClient):
    """reposhark-<hash>.vercel.app preview deploys must be allowed."""
    response = await client.get("/health", headers={"Origin": "https://reposhark-git-main.vercel.app"})
    assert response.headers.get("access-control-allow-origin") == "https://reposhark-git-main.vercel.app"


@pytest.mark.asyncio
async def test_cors_allowed_origin_reporium_vercel(client: AsyncClient):
    """Legacy reporium.vercel.app must remain allowed."""
    response = await client.get("/health", headers={"Origin": "https://reporium.vercel.app"})
    assert response.headers.get("access-control-allow-origin") == "https://reporium.vercel.app"


@pytest.mark.asyncio
async def test_cors_blocked_unknown_origin(client: AsyncClient):
    """Unknown origins must not receive allow-origin header."""
    response = await client.get("/health", headers={"Origin": "https://evil.example.com"})
    assert response.headers.get("access-control-allow-origin") is None
