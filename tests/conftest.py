import os
from collections.abc import AsyncGenerator

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/reporium_test")
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


@pytest_asyncio.fixture(scope="session")
async def _setup_db():
    await db_module.engine.dispose()
    # NullPool: no connection pooling → each session gets a fresh connection.
    # This prevents "Future attached to a different loop" errors when pytest-asyncio
    # creates a new event loop per test while the engine is session-scoped.
    db_module.engine = create_async_engine(TEST_DB_URL, echo=False, poolclass=NullPool)
    db_module.async_session_factory.configure(bind=db_module.engine)

    async with db_module.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Install pgvector and add embedding_vec columns that are managed
        # outside the ORM model (via raw migrations in production).
        await conn.execute(
            text("CREATE EXTENSION IF NOT EXISTS vector")
        )
        await conn.execute(
            text("ALTER TABLE taxonomy_values ADD COLUMN IF NOT EXISTS embedding_vec vector(384)")
        )
        await conn.execute(
            text("ALTER TABLE repo_embeddings ADD COLUMN IF NOT EXISTS embedding_vec vector(384)")
        )
    yield
    await db_module.engine.dispose()
    db_module.engine = create_async_engine(TEST_DB_URL, echo=False, poolclass=NullPool)
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
