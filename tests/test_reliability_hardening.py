"""Tests for KAN-ask-reliability hardening changes.

Covers:
- Database connection pool sizing
- Fire-and-forget task exception callback
- (Startup env validation is tested implicitly via the lifespan check)
"""

import asyncio
import logging

import pytest


class TestPoolSizing:
    """Verify the async engine is configured with explicit pool parameters."""

    def test_pool_size_is_set(self):
        from app.database import engine

        pool = engine.pool
        assert pool.size() == 20, f"Expected pool_size=20, got {pool.size()}"

    def test_max_overflow_is_set(self):
        from app.database import engine

        pool = engine.pool
        assert pool._max_overflow == 10, f"Expected max_overflow=10, got {pool._max_overflow}"

    def test_pool_recycle_is_set(self):
        from app.database import engine

        pool = engine.pool
        assert pool._recycle == 3600, f"Expected pool_recycle=3600, got {pool._recycle}"

    def test_pool_pre_ping_is_enabled(self):
        from app.database import engine

        assert engine.pool._pre_ping is True, "pool_pre_ping should be True"


class TestTaskDoneCallback:
    """Verify _task_done_callback logs warnings for failed tasks and stays silent for successful ones."""

    @pytest.mark.asyncio
    async def test_logs_warning_for_failed_task(self, caplog):
        from app.routers.intelligence import _task_done_callback

        async def _failing():
            raise RuntimeError("boom")

        task = asyncio.create_task(_failing())
        try:
            await task
        except RuntimeError:
            pass
        with caplog.at_level(logging.WARNING, logger="app.routers.intelligence"):
            _task_done_callback(task)

        assert any("boom" in record.message for record in caplog.records), (
            "Expected a warning log containing 'boom'"
        )
        assert any("Background task" in record.message for record in caplog.records), (
            "Expected log message to mention 'Background task'"
        )

    @pytest.mark.asyncio
    async def test_no_log_for_successful_task(self, caplog):
        from app.routers.intelligence import _task_done_callback

        async def _ok():
            return 42

        task = asyncio.create_task(_ok())
        await task
        with caplog.at_level(logging.WARNING, logger="app.routers.intelligence"):
            _task_done_callback(task)

        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warning_records) == 0, (
            f"Expected no warning logs for a successful task, got: {warning_records}"
        )

    @pytest.mark.asyncio
    async def test_no_log_for_cancelled_task(self, caplog):
        from app.routers.intelligence import _task_done_callback

        async def _slow():
            await asyncio.sleep(100)

        task = asyncio.create_task(_slow())
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        with caplog.at_level(logging.WARNING, logger="app.routers.intelligence"):
            _task_done_callback(task)

        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warning_records) == 0, (
            f"Expected no warning logs for a cancelled task, got: {warning_records}"
        )
