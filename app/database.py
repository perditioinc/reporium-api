import asyncio
import logging
from collections.abc import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import settings

logger = logging.getLogger(__name__)

engine = create_async_engine(
    settings.database_url,
    echo=settings.environment == "development",
    pool_pre_ping=True,
    # db-f1-micro has max_connections=25. Keep pool small so multiple Cloud Run
    # instances don't exhaust connections. 5 + 2 overflow = 7 per instance;
    # at max 3 concurrent instances that is 21 — well under the limit.
    #
    # containerConcurrency in deploy/service.yaml is set to 8 (≤ pool_size+max_overflow)
    # to prevent pool starvation: requests beyond 7 concurrent DB connections
    # would queue and hit the 30s pool_timeout under normal f1-micro latency,
    # causing transient 500s (~20% error rate measured 2026-04-20).
    pool_size=5,
    max_overflow=2,
    # Fail fast on pool exhaustion (10s) rather than waiting the default 30s.
    # Under pool saturation a fast 500 is better than a 30s hanging request
    # that delays the client's retry and blocks Uvicorn workers.
    pool_timeout=10,
    pool_recycle=1800,
    # asyncpg statement-level timeout: abort any single query that runs longer
    # than 20s. Prevents a rogue slow query from holding a connection and
    # starving the pool even when concurrency is within bounds.
    connect_args={"command_timeout": 20},
)

async_session_factory = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        yield session


async def check_db_connection() -> None:
    """Verify the DB connection is healthy on startup.

    Retries up to 3 times with exponential backoff (1s, 2s, 4s).
    Logs a warning and continues if all attempts fail — does not crash the app.
    """
    delays = [1, 2, 4]
    for attempt, delay in enumerate(delays, start=1):
        try:
            async with async_session_factory() as session:
                await session.execute(text("SELECT 1"))
            logger.info("DB connection healthy (attempt %d)", attempt)
            return
        except Exception as exc:
            logger.warning(
                "DB connection check failed (attempt %d/%d): %s",
                attempt,
                len(delays),
                exc,
            )
            if attempt < len(delays):
                await asyncio.sleep(delay)

    logger.warning(
        "DB connection could not be verified after %d attempts — continuing anyway",
        len(delays),
    )
