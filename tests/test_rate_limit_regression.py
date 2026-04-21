"""
Regression guard: rate-limit decorators on ingest/* and ingest/events/* endpoints.

These tests assert the presence of the @limiter.limit decorator on every
POST/PUT/DELETE endpoint in ingest.py that was missing rate limiting (P1 finding
from security-sweep-batch1, 2026-04-20).

Strategy: inspect source code of each handler to confirm the decorator literal
is present. This prevents silent revert of rate-limit decorators.

Note: RATELIMIT_ENABLED=0 in the test environment, so 429 responses are not
triggered. We're verifying the decorator is wired, not functional behavior.
"""

import inspect

import pytest


def _get_limit_string(handler) -> str:
    """Extract the rate-limit string from a slowapi-decorated handler's source."""
    return inspect.getsource(handler)


class TestIngestRateLimitDecorators:
    """Asserts that each previously-unprotected ingest endpoint has a rate-limit decorator."""

    def test_enrich_repo_has_rate_limit(self):
        from app.routers import ingest
        src = _get_limit_string(ingest.enrich_repo)
        assert "600/minute" in src, (
            "enrich_repo must have @limiter.limit('600/minute') — "
            "P1 security finding: missing rate limit on X-Ingest-Key gated endpoint"
        )

    def test_ingest_trend_snapshot_has_rate_limit(self):
        from app.routers import ingest
        src = _get_limit_string(ingest.ingest_trend_snapshot)
        assert "600/minute" in src, (
            "ingest_trend_snapshot must have @limiter.limit('600/minute') — "
            "P1 security finding: missing rate limit on X-Ingest-Key gated endpoint"
        )

    def test_ingest_gaps_has_rate_limit(self):
        from app.routers import ingest
        src = _get_limit_string(ingest.ingest_gaps)
        assert "600/minute" in src, (
            "ingest_gaps must have @limiter.limit('600/minute') — "
            "P1 security finding: missing rate limit on X-Ingest-Key gated endpoint"
        )

    def test_ingest_log_has_rate_limit(self):
        from app.routers import ingest
        src = _get_limit_string(ingest.ingest_log)
        assert "600/minute" in src, (
            "ingest_log must have @limiter.limit('600/minute') — "
            "P1 security finding: missing rate limit on X-Ingest-Key gated endpoint"
        )

    def test_repo_ingested_event_has_rate_limit(self):
        from app.routers import ingest
        src = _get_limit_string(ingest.repo_ingested_event)
        assert "600/minute" in src, (
            "repo_ingested_event must have @limiter.limit('600/minute') — "
            "P1 security finding: missing rate limit on X-Ingest-Key gated event endpoint"
        )

    def test_repo_added_event_has_rate_limit(self):
        from app.routers import ingest
        src = _get_limit_string(ingest.repo_added_event)
        assert "600/minute" in src, (
            "repo_added_event must have @limiter.limit('600/minute') — "
            "P1 security finding: missing rate limit on X-Ingest-Key gated event endpoint"
        )

    def test_ingest_repos_existing_rate_limit_unchanged(self):
        """Verify the pre-existing 200/minute limit on /ingest/repos was not changed."""
        from app.routers import ingest
        src = _get_limit_string(ingest.ingest_repos)
        assert "200/minute" in src, (
            "ingest_repos must retain @limiter.limit('200/minute') — "
            "do not change existing rate limits"
        )

    def test_all_newly_limited_endpoints_have_request_param(self):
        """slowapi requires `request: Request` as first param for the limiter to work."""
        from app.routers import ingest

        handlers = [
            ("enrich_repo", ingest.enrich_repo),
            ("ingest_trend_snapshot", ingest.ingest_trend_snapshot),
            ("ingest_gaps", ingest.ingest_gaps),
            ("ingest_log", ingest.ingest_log),
            ("repo_ingested_event", ingest.repo_ingested_event),
            ("repo_added_event", ingest.repo_added_event),
        ]
        for name, handler in handlers:
            sig = inspect.signature(handler)
            params = list(sig.parameters.keys())
            assert "request" in params, (
                f"{name} must have `request: Request` parameter for slowapi to function"
            )
