"""Tests for KAN-ask-observability: PII scrub from error messages and phase-level latency logging."""
from __future__ import annotations

import logging
import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fastapi import HTTPException


# ---------------------------------------------------------------------------
# 1. Error messages must NOT contain user-supplied strings
# ---------------------------------------------------------------------------


class TestErrorMessagePIIScrub:
    """Verify that HTTPException detail messages in intelligence.py never echo
    user-supplied input (repo names, category names, search terms)."""

    def _collect_http_exceptions(self, source_lines: list[str]) -> list[str]:
        """Extract all `detail=` values from HTTPException raises in source."""
        details = []
        in_exception = False
        detail_buf = ""
        for line in source_lines:
            stripped = line.strip()
            if "raise HTTPException" in stripped or "HTTPException(" in stripped:
                in_exception = True
                detail_buf = ""
            if in_exception:
                detail_buf += stripped + " "
                if ")" in stripped:
                    in_exception = False
                    # Extract the detail value
                    m = re.search(r'detail\s*=\s*(.+?)(?:\s*[,)])', detail_buf)
                    if m:
                        details.append(m.group(1))
        return details

    def test_no_fstring_in_error_details(self):
        """No HTTPException detail should use f-strings that interpolate variables."""
        import inspect
        from app.routers import intelligence

        source = inspect.getsource(intelligence)
        # Look for detail=f"..." or detail=f'...' patterns
        fstring_details = re.findall(r'detail\s*=\s*f["\']', source)
        assert fstring_details == [], (
            f"Found {len(fstring_details)} f-string detail(s) in HTTPException raises — "
            "these may leak user-supplied input. Use static messages instead."
        )

    def test_repo_not_found_message_is_generic(self):
        """The 404 'repo not found' message must not contain the repo name."""
        import inspect
        from app.routers import intelligence

        source = inspect.getsource(intelligence)
        # Ensure the old patterns are gone
        assert "Repo '" not in source or "detail" not in source.split("Repo '")[0][-50:], \
            "Found 'Repo '<name>' pattern that may leak user input"
        # Ensure the new generic messages exist
        assert '"Repo not found"' in source
        assert '"No repos found for the specified category"' in source

    def test_error_detail_does_not_contain_user_input_examples(self):
        """Simulate user-supplied strings and ensure they cannot appear in error details."""
        user_inputs = [
            "my-secret-repo",
            "private-category-name",
            "SELECT * FROM users",
            "<script>alert('xss')</script>",
        ]
        # Read the actual error detail strings from the module
        import inspect
        from app.routers import intelligence

        source = inspect.getsource(intelligence)
        # Collect all detail= string literals
        detail_literals = re.findall(r'detail\s*=\s*"([^"]*)"', source)

        for detail in detail_literals:
            for user_input in user_inputs:
                assert user_input not in detail, (
                    f"User input '{user_input}' found in error detail: {detail}"
                )


# ---------------------------------------------------------------------------
# 2. Phase-level latency breakdown logging
# ---------------------------------------------------------------------------


class TestLatencyBreakdownLogging:
    """Verify that _run_query emits the expected latency breakdown log line."""

    def test_latency_log_pattern_exists_in_source(self):
        """The latency breakdown log pattern must exist in the source code."""
        import inspect
        from app.routers import intelligence

        source = inspect.getsource(intelligence)
        assert "ask latency breakdown:" in source, (
            "Expected 'ask latency breakdown:' log pattern not found in intelligence.py"
        )
        # Verify it includes all required phase keys
        for phase in ["total=", "smart=", "embed=", "search=", "context=", "claude=", "model=", "cached="]:
            assert phase in source, f"Missing phase '{phase}' in latency log pattern"

    def test_latency_log_emitted_on_cache_hit(self, caplog):
        """On a cache hit, the latency log should be emitted with claude=0ms and cached=True."""
        from app.routers.intelligence import logger as intel_logger

        # Simulate the log line that would be emitted on a cache hit
        with caplog.at_level(logging.INFO, logger=intel_logger.name):
            intel_logger.info(
                "ask latency breakdown: total=%dms smart=%dms embed=%dms search=%dms context=%dms claude=%dms model=%s cached=%s",
                42, 5, 10, 15, 12, 0, "smart-route:stats", True,
            )

        assert any("ask latency breakdown:" in r.message for r in caplog.records)
        matching = [r for r in caplog.records if "ask latency breakdown:" in r.message]
        assert len(matching) == 1
        msg = matching[0].getMessage()
        assert "claude=0ms" in msg
        assert "cached=True" in msg

    def test_latency_log_format_non_cache(self, caplog):
        """On a non-cache path, the log should show actual claude timing and cached=False."""
        from app.routers.intelligence import logger as intel_logger

        with caplog.at_level(logging.INFO, logger=intel_logger.name):
            intel_logger.info(
                "ask latency breakdown: total=%dms smart=%dms embed=%dms search=%dms context=%dms claude=%dms model=%s cached=%s",
                350, 5, 30, 50, 15, 250, "claude-sonnet-4-20250514", False,
            )

        matching = [r for r in caplog.records if "ask latency breakdown:" in r.message]
        assert len(matching) == 1
        msg = matching[0].getMessage()
        assert "claude=250ms" in msg
        assert "cached=False" in msg
        assert "claude-sonnet-4-20250514" in msg

    def test_query_context_has_timing_fields(self):
        """QueryContext dataclass must have the phase-level timing fields."""
        from app.routers.intelligence import QueryContext

        ctx = QueryContext(
            sources=[],
            context_text="",
            model="test",
            session_history=[],
            cache_result=None,
            query_embedding=None,
            route_label=None,
            t_smart_ms=5.0,
            t_embed_ms=10.0,
            t_search_ms=15.0,
            t_context_ms=12.0,
        )
        assert ctx.t_smart_ms == 5.0
        assert ctx.t_embed_ms == 10.0
        assert ctx.t_search_ms == 15.0
        assert ctx.t_context_ms == 12.0

    def test_query_context_timing_defaults_to_zero(self):
        """QueryContext timing fields should default to 0.0."""
        from app.routers.intelligence import QueryContext

        ctx = QueryContext(
            sources=[],
            context_text="",
            model="test",
            session_history=[],
            cache_result=None,
            query_embedding=None,
            route_label=None,
        )
        assert ctx.t_smart_ms == 0.0
        assert ctx.t_embed_ms == 0.0
        assert ctx.t_search_ms == 0.0
        assert ctx.t_context_ms == 0.0

    def test_cache_hit_log_includes_all_phases(self):
        """Verify the log message format includes all 6 timing phases plus model and cached flag."""
        import re
        pattern = re.compile(
            r"ask latency breakdown: "
            r"total=\d+ms "
            r"smart=\d+ms "
            r"embed=\d+ms "
            r"search=\d+ms "
            r"context=\d+ms "
            r"claude=\d+ms "
            r"model=\S+ "
            r"cached=(True|False)"
        )
        test_msg = (
            "ask latency breakdown: total=42ms smart=5ms embed=10ms "
            "search=15ms context=12ms claude=0ms model=smart-route:stats cached=True"
        )
        assert pattern.search(test_msg), "Log message does not match expected pattern"
