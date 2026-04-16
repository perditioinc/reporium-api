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
    """Verify the async engine is configured with explicit pool parameters.

    The test engine uses NullPool (to prevent event-loop issues in pytest-asyncio),
    so we inspect the database.py source code to confirm the production values are set
    rather than reading live pool attributes.
    """

    def _get_database_source(self) -> str:
        import inspect
        import app.database as db_module
        return inspect.getsource(db_module)

    def test_pool_size_is_set(self):
        src = self._get_database_source()
        # pool_size=5: Cloud SQL f1-micro has max_connections=25; 5+2 overflow per
        # instance keeps the total under 25 even with 3 concurrent Cloud Run instances.
        assert "pool_size=5" in src, "database.py must set pool_size=5 (f1-micro max_connections=25 constraint)"

    def test_max_overflow_is_set(self):
        src = self._get_database_source()
        assert "max_overflow=2" in src, "database.py must set max_overflow=2 (f1-micro constraint)"

    def test_pool_recycle_is_set(self):
        src = self._get_database_source()
        assert "pool_recycle=1800" in src, "database.py must set pool_recycle=1800"

    def test_pool_pre_ping_is_enabled(self):
        src = self._get_database_source()
        assert "pool_pre_ping=True" in src, "database.py must set pool_pre_ping=True"


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
