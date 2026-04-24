import os
import socket
from collections.abc import AsyncGenerator
from urllib.parse import urlparse

import pytest
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
os.environ["ENVIRONMENT"] = "test"  # skip embedding model pre-warm in tests

import importlib

import app.database as db_module
from app.database import Base, get_db
from app.main import app

# Ensure all ORM models are imported so Base.metadata.create_all() creates
# every table, including ones not transitively imported by main.py at startup.
importlib.import_module("app.models.query_log")
importlib.import_module("app.models.audit_log")
importlib.import_module("app.models.dependency")
importlib.import_module("app.models.mention")
importlib.import_module("app.models.session")

TEST_API_KEY = "test-api-key"
AUTH_HEADERS = {"Authorization": f"Bearer {TEST_API_KEY}"}

TEST_DB_URL = os.environ["DATABASE_URL"]


def _test_db_available() -> bool:
    """Quick TCP probe to see if the test Postgres is reachable.

    If HAS_TEST_DB=1 is set explicitly (e.g. in CI), trust it and don't probe.
    Otherwise, try a short socket connect to the DATABASE_URL host:port.
    This lets local dev runs skip DB-dependent tests cleanly instead of
    failing with ConnectionRefusedError x200+.
    """
    if os.getenv("HAS_TEST_DB") == "1":
        return True
    # Allow explicit opt-out for CI safety — treat CI as always having DB.
    if os.getenv("CI") == "true":
        return True
    try:
        # DATABASE_URL looks like postgresql+asyncpg://user:pw@host:port/db
        parsed = urlparse(TEST_DB_URL.replace("postgresql+asyncpg://", "postgresql://"))
        host = parsed.hostname or "localhost"
        port = parsed.port or 5432
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except (OSError, socket.timeout):
        return False


_TEST_DB_AVAILABLE = _test_db_available()


@pytest_asyncio.fixture(scope="session")
async def _setup_db():
    if not _TEST_DB_AVAILABLE:
        pytest.skip(
            "Test Postgres not reachable at DATABASE_URL. "
            "Start Postgres and set HAS_TEST_DB=1 (or run in CI) to enable DB-dependent tests."
        )
    await db_module.engine.dispose()
    # NullPool: no connection pooling → each session gets a fresh connection.
    # This prevents "Future attached to a different loop" errors when pytest-asyncio
    # creates a new event loop per test while the engine is session-scoped.
    db_module.engine = create_async_engine(
        TEST_DB_URL,
        echo=False,
        poolclass=NullPool,
        connect_args={"command_timeout": 30},
    )
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
        # Create repo_edges table (managed outside ORM by build_knowledge_graph.py)
        await conn.execute(
            text("""
                CREATE TABLE IF NOT EXISTS repo_edges (
                    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
                    source_repo_id UUID NOT NULL REFERENCES repos(id),
                    target_repo_id UUID NOT NULL REFERENCES repos(id),
                    edge_type TEXT NOT NULL,
                    weight FLOAT DEFAULT 1.0,
                    evidence JSONB DEFAULT '{}',
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE(source_repo_id, target_repo_id, edge_type)
                )
            """)
        )
        await conn.execute(
            text("CREATE INDEX IF NOT EXISTS idx_repo_edges_source ON repo_edges(source_repo_id)")
        )
        await conn.execute(
            text("CREATE INDEX IF NOT EXISTS idx_repo_edges_target ON repo_edges(target_repo_id)")
        )
        await conn.execute(
            text("CREATE INDEX IF NOT EXISTS idx_repo_edges_type ON repo_edges(edge_type)")
        )
        # query_log.question_embedding_vec is added by migration in production;
        # the ORM model doesn't include it, so we add it manually here.
        await conn.execute(
            text("ALTER TABLE query_log ADD COLUMN IF NOT EXISTS question_embedding_vec vector(384)")
        )
    yield
    # Teardown is best-effort: the test database is ephemeral in CI (a fresh
    # service container per workflow run) and trashed at the end of local dev
    # sessions. If asyncpg times out mid-drop (the per-table drops + CASCADEs
    # accumulate seconds and have repeatedly tripped the 30s `command_timeout`
    # in CI — see issues #428/#426/#424/#421/#417/...), don't let that flake
    # turn the whole pytest session into a failure. The next run will drop
    # the leftover tables anyway via `IF NOT EXISTS` semantics in setup.
    try:
        await db_module.engine.dispose()
        db_module.engine = create_async_engine(
            TEST_DB_URL,
            echo=False,
            poolclass=NullPool,
            # Bump teardown timeout: drop_all() walks ~25 tables sequentially
            # and has been hitting the previous 30s ceiling on slow runners.
            connect_args={"command_timeout": 120},
        )
        async with db_module.engine.begin() as conn:
            # Drop repo_edges first — it has FK constraints to repos and is not
            # tracked by the ORM, so drop_all() would fail with DependentObjects.
            await conn.execute(text("DROP TABLE IF EXISTS repo_edges CASCADE"))
            await conn.run_sync(Base.metadata.drop_all)
        await db_module.engine.dispose()
    except Exception as e:  # noqa: BLE001 — teardown should never fail the run
        # Print rather than raise: pytest captures stdout per-test, but a
        # session-fixture exception during teardown surfaces as a session-level
        # ERROR that flips the whole run to red even if every test passed.
        print(f"[conftest] non-fatal teardown error (test DB is ephemeral): {e!r}")


@pytest_asyncio.fixture
async def client(_setup_db) -> AsyncGenerator[AsyncClient, None]:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


TEST_REPO_FIXTURE = {
    "name": "test-repo",
    "owner": "testuser",
    "description": "A test repository",
    "is_fork": True,
    # is_private is REQUIRED by the ingest schema (no Pydantic default) as a
    # structural guard against the 2026-04-23 private-repo leak. Tests that
    # want to ingest a private repo override this to True.
    "is_private": False,
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
