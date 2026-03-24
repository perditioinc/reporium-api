import asyncio
import os
from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

os.environ["DATABASE_URL"] = "postgresql+asyncpg://postgres:postgres@localhost:5432/reporium_test"
os.environ["INGESTION_API_KEY"] = "test-api-key"
os.environ["GH_USERNAME"] = "testuser"
os.environ["REDIS_URL"] = ""  # disable Redis in tests
os.environ["RATELIMIT_ENABLED"] = "0"  # disable rate limiting in tests

import app.database as db_module
from app.database import Base, get_db
from app.main import app

TEST_API_KEY = "test-api-key"
AUTH_HEADERS = {"Authorization": f"Bearer {TEST_API_KEY}"}

TEST_DB_URL = os.environ["DATABASE_URL"]


@pytest.fixture(scope="session")
def event_loop():
    """Single event loop for all tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def _setup_db(event_loop):
    # Replace the app engine with one on this event loop
    await db_module.engine.dispose()
    db_module.engine = create_async_engine(TEST_DB_URL, echo=False, pool_pre_ping=True)
    db_module.async_session_factory.configure(bind=db_module.engine)

    async with db_module.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    # Dispose engine first to drain connection pool before loop teardown
    await db_module.engine.dispose()
    # Recreate for drop_all, then dispose again
    db_module.engine = create_async_engine(TEST_DB_URL, echo=False, pool_pre_ping=True)
    async with db_module.engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await db_module.engine.dispose()


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
    "open_issues_count": 42,
    "license_spdx": "MIT",
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
