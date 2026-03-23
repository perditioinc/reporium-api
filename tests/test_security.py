"""Security tests — ensure private repos are never exposed."""
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_library_full_no_private_repo_names(client: AsyncClient):
    """No known private repo names should appear in /library/full."""
    PRIVATE_NAMES = {
        'didymo-ai-agent', 'didymo-ai-api', 'didymo-ai-auth', 'didymo-ai-gcp-tts',
        'simon-brain', 'whatsapp-webhook', 'whatsapp-template-generator',
        'ticket-generator', 'ticket-issuer', 'mind-guard-app', 'ideas-2026',
        '18degrees-ecom', 'perditio-infra', 'perditio-web', 'perditio-services',
    }
    resp = await client.get("/library/full")
    if resp.status_code == 200:
        data = resp.json()
        repo_names = {r["name"] for r in data.get("repos", [])}
        exposed = repo_names & PRIVATE_NAMES
        assert len(exposed) == 0, f"PRIVATE REPOS EXPOSED: {exposed}"


@pytest.mark.asyncio
async def test_library_full_all_repos_not_fork(client: AsyncClient):
    """Every repo in /library/full must be a non-fork."""
    resp = await client.get("/library/full")
    if resp.status_code == 200:
        for repo in resp.json().get("repos", []):
            assert repo.get("isFork") is False, f"{repo['name']} is a fork"


@pytest.mark.asyncio
async def test_health_does_not_leak_secrets(client: AsyncClient):
    """Health endpoint must not expose connection strings or keys."""
    resp = await client.get("/health")
    body = resp.text
    assert "postgresql" not in body.lower()
    assert "redis://" not in body.lower()
    assert "ghp_" not in body
    assert "sk-ant-" not in body
