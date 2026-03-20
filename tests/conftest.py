import os
from collections.abc import AsyncGenerator

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/reporium_test")
os.environ.setdefault("INGESTION_API_KEY", "test-api-key")
os.environ.setdefault("GH_USERNAME", "testuser")

from app.database import Base
from app.main import app

TEST_API_KEY = "test-api-key"
AUTH_HEADERS = {"Authorization": f"Bearer {TEST_API_KEY}"}

TEST_DB_URL = os.environ["DATABASE_URL"]

_engine = create_async_engine(TEST_DB_URL, echo=False)


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def _setup_db():
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await _engine.dispose()


@pytest_asyncio.fixture
async def client(_setup_db) -> AsyncGenerator[AsyncClient, None]:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


TEST_REPO_FIXTURE = {
    "name": "test-repo",
    "owner": "testuser",
    "description": "A test repository",
    "is_fork": True,
    "forked_from": "upstream/test-repo",
    "primary_language": "Python",
    "github_url": "https://github.com/testuser/test-repo",
    "fork_sync_state": "behind",
    "behind_by": 5,
    "ahead_by": 0,
    "parent_stars": 1000,
    "parent_forks": 200,
    "parent_is_archived": False,
    "commits_last_7_days": 3,
    "commits_last_30_days": 12,
    "commits_last_90_days": 40,
    "activity_score": 75,
    "tags": ["ai", "python", "llm"],
    "categories": [
        {"category_id": "ai-agents", "category_name": "AI Agents", "is_primary": True}
    ],
    "builders": [
        {"login": "testuser", "display_name": "Test User", "org_category": "individual", "is_known_org": False}
    ],
    "ai_dev_skills": ["prompt-engineering", "rag"],
    "pm_skills": ["product-strategy"],
    "languages": [
        {"language": "Python", "bytes": 50000, "percentage": 90.0},
        {"language": "Shell", "bytes": 5000, "percentage": 10.0},
    ],
    "commits": [],
}
