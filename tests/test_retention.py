"""Tests for app.retention — query_log 90-day retention purge."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app import retention


@pytest.mark.asyncio
async def test_purge_old_query_logs_executes_parameterized_delete():
    """purge_old_query_logs should issue a parameterized DELETE and commit."""
    exec_result = MagicMock()
    exec_result.rowcount = 7

    mock_db = AsyncMock()
    mock_db.execute = AsyncMock(return_value=exec_result)
    mock_db.commit = AsyncMock()

    # async_session_factory() is used as an async context manager.
    session_cm = MagicMock()
    session_cm.__aenter__ = AsyncMock(return_value=mock_db)
    session_cm.__aexit__ = AsyncMock(return_value=None)

    factory = MagicMock(return_value=session_cm)

    with patch.object(retention, "async_session_factory", factory):
        count = await retention.purge_old_query_logs(days=90)

    assert count == 7
    factory.assert_called_once()
    mock_db.execute.assert_awaited_once()
    mock_db.commit.assert_awaited_once()

    # Verify the SQL and bound parameters.
    call_args = mock_db.execute.call_args
    sql_clause = call_args.args[0]
    params = call_args.args[1]
    assert "DELETE FROM query_log" in str(sql_clause)
    assert "created_at" in str(sql_clause)
    assert params == {"days": 90}


@pytest.mark.asyncio
async def test_purge_old_query_logs_custom_days():
    """Custom `days` argument flows through to the SQL parameters."""
    exec_result = MagicMock()
    exec_result.rowcount = 0

    mock_db = AsyncMock()
    mock_db.execute = AsyncMock(return_value=exec_result)
    mock_db.commit = AsyncMock()

    session_cm = MagicMock()
    session_cm.__aenter__ = AsyncMock(return_value=mock_db)
    session_cm.__aexit__ = AsyncMock(return_value=None)

    factory = MagicMock(return_value=session_cm)

    with patch.object(retention, "async_session_factory", factory):
        count = await retention.purge_old_query_logs(days=30)

    assert count == 0
    params = mock_db.execute.call_args.args[1]
    assert params == {"days": 30}


@pytest.mark.asyncio
async def test_purge_old_query_logs_handles_none_rowcount():
    """If the driver returns rowcount=None, we should coerce to 0."""
    exec_result = MagicMock()
    exec_result.rowcount = None

    mock_db = AsyncMock()
    mock_db.execute = AsyncMock(return_value=exec_result)
    mock_db.commit = AsyncMock()

    session_cm = MagicMock()
    session_cm.__aenter__ = AsyncMock(return_value=mock_db)
    session_cm.__aexit__ = AsyncMock(return_value=None)

    factory = MagicMock(return_value=session_cm)

    with patch.object(retention, "async_session_factory", factory):
        count = await retention.purge_old_query_logs()

    assert count == 0


@pytest.mark.asyncio
async def test_retention_loop_disabled_returns_immediately(monkeypatch):
    """When ENABLE_RETENTION_PURGE is false, the loop should exit immediately."""
    monkeypatch.setenv("ENABLE_RETENTION_PURGE", "false")
    # Should return without calling purge / sleep.
    with patch.object(retention, "purge_old_query_logs", new=AsyncMock()) as mock_purge, \
         patch("asyncio.sleep", new=AsyncMock()) as mock_sleep:
        await retention.retention_loop()
    mock_purge.assert_not_awaited()
    mock_sleep.assert_not_awaited()
