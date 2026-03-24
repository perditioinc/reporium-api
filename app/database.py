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
