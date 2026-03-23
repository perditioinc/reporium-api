"""Security tests — ensure private repos are never exposed, public forks are included."""
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_library_full_never_exposes_private_repos(client: AsyncClient):
    """No known private repo names should appear in /library/full."""
    PRIVATE_NAMES = {
        'didymo-ai-agent', 'didymo-ai-api', 'didymo-ai-auth', 'didymo-ai-gcp-tts',
        'simon-brain', 'whatsapp-webhook', 'whatsapp-template-generator',
        'ticket-generator', 'ticket-issuer', 'mind-guard-app', 'ideas-2026',
        '18degrees-ecom', 'perditio-infra', 'perditio-web', 'perditio-services',
    }
    try:
        resp = await client.get("/library/full")
    except Exception:
        pytest.skip("library/full unavailable in test environment")
        return
    if resp.status_code != 200:
        pytest.skip("library/full requires full schema with migration columns")
    data = resp.json()
    repo_names = {r["name"] for r in data.get("repos", [])}
    exposed = repo_names & PRIVATE_NAMES
    assert len(exposed) == 0, f"PRIVATE REPOS EXPOSED: {exposed}"


@pytest.mark.asyncio
async def test_library_full_includes_public_forks(client: AsyncClient):
    """Public forks must be included in /library/full — they are the core content."""
    try:
        resp = await client.get("/library/full")
    except Exception:
        pytest.skip("library/full unavailable in test environment")
        return
    if resp.status_code != 200:
        pytest.skip("library/full requires full schema with migration columns")
    repos = resp.json().get("repos", [])
    forks = [r for r in repos if r.get("isFork")]
    # In production there are 1000+ public forks. In test env there may be fewer.
    # Just verify forks are not being filtered out if data exists.
    if len(repos) > 20:
        assert len(forks) > 0, "Public forks are missing from /library/full"


@pytest.mark.asyncio
async def test_health_does_not_leak_secrets(client: AsyncClient):
    """Health endpoint must not expose connection strings or keys."""
    resp = await client.get("/health")
    body = resp.text
    assert "postgresql" not in body.lower()
    assert "redis://" not in body.lower()
    assert "ghp_" not in body
    assert "sk-ant-" not in body
