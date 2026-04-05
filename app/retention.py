"""Background retention tasks for query_log and other PII stores."""
import asyncio
import logging
import os

from sqlalchemy import text

from app.database import async_session_factory

logger = logging.getLogger(__name__)

RETENTION_DAYS = int(os.getenv("QUERY_LOG_RETENTION_DAYS", "90"))
RETENTION_INTERVAL_SECONDS = 24 * 60 * 60  # 24 hours


async def purge_old_query_logs(days: int = RETENTION_DAYS) -> int:
    """Delete query_log rows older than ``days`` days. Returns row count."""
    async with async_session_factory() as db:
        # Parameterized interval to avoid SQL injection
        result = await db.execute(
            text("DELETE FROM query_log WHERE created_at < NOW() - (:days || ' days')::interval"),
            {"days": days},
        )
        await db.commit()
        return result.rowcount or 0


async def retention_loop() -> None:
    """Run purge every 24 hours. Fire and forget."""
    if os.getenv("ENABLE_RETENTION_PURGE", "true").lower() != "true":
        logger.info("Retention purge disabled")
        return
    while True:
        try:
            count = await purge_old_query_logs()
            logger.info(
                "Retention purge complete: deleted %d rows older than %d days",
                count,
                RETENTION_DAYS,
            )
        except Exception:
            logger.exception("Retention purge failed (will retry next cycle)")
        await asyncio.sleep(RETENTION_INTERVAL_SECONDS)
