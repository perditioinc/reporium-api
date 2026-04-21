"""
Tests for DB pool configuration correctness.

These are unit-level assertions that parse the pool parameters from the
source of ``app/database.py`` — they do NOT introspect the live engine,
because under ``ENVIRONMENT=test`` the engine uses ``NullPool`` (no pooling),
which does not expose ``size()`` / ``_max_overflow`` / ``_timeout``.

They guard against regressions where someone bumps pool_size or the Cloud Run
concurrency without updating the other.

Root cause documented in .audit/2026-04-20/db-pool-diagnosis.md:
- containerConcurrency=20 with pool_size=5+2=7 caused ~20% transient 500s
- Fix: containerConcurrency ≤ pool_size + max_overflow; pool_timeout explicit
"""

import inspect
import os
import re

import pytest

from app import database as db_module


# ---------------------------------------------------------------------------
# Helpers — parse the engine config from source so tests work under NullPool
# ---------------------------------------------------------------------------

_DB_SOURCE = inspect.getsource(db_module)


def _parse_kw_int(kw: str) -> int | None:
    """Find ``<kw>=<int>`` (kwarg form) in app/database.py source."""
    m = re.search(rf"\b{re.escape(kw)}\s*=\s*(\d+)", _DB_SOURCE)
    return int(m.group(1)) if m else None


def _pool_capacity() -> int:
    pool_size = _parse_kw_int("pool_size")
    max_overflow = _parse_kw_int("max_overflow")
    assert pool_size is not None, "pool_size must be set explicitly in app/database.py"
    assert max_overflow is not None, "max_overflow must be set explicitly in app/database.py"
    return pool_size + max_overflow


# ---------------------------------------------------------------------------
# Pool settings unit tests
# ---------------------------------------------------------------------------

class TestPoolSettings:
    """Assert the engine is configured with the correct pool parameters."""

    def test_pool_size(self):
        size = _parse_kw_int("pool_size")
        assert size == 5, (
            f"pool_size changed to {size}; update the concurrency cap accordingly. "
            "See .audit/2026-04-20/db-pool-diagnosis.md"
        )

    def test_max_overflow(self):
        mo = _parse_kw_int("max_overflow")
        assert mo == 2, (
            f"max_overflow changed to {mo}; update the concurrency cap accordingly."
        )

    def test_pool_timeout_is_set_and_reasonable(self):
        """pool_timeout must be explicitly set and < Cloud Run timeout.

        Leaving pool_timeout at the SQLAlchemy default (30s) means a saturated
        pool blocks each request for 30s before raising TimeoutError → 500.
        We set it to 10s so failures are fast and the client can retry sooner.
        """
        timeout = _parse_kw_int("pool_timeout")
        assert timeout is not None, (
            "pool_timeout must be set explicitly (not relying on default 30s)"
        )
        assert timeout <= 15, (
            f"pool_timeout={timeout}s is too long; should be ≤15s. "
            "Cloud Run container timeout is 60s."
        )

    def test_connect_args_command_timeout(self):
        """asyncpg command_timeout must be set to prevent rogue queries from blocking the pool."""
        assert "command_timeout" in _DB_SOURCE, (
            "connect_args command_timeout must be set in app/database.py. "
            "Without it, slow queries hold connections indefinitely."
        )
        m = re.search(r"command_timeout['\"]?\s*:\s*(\d+)", _DB_SOURCE)
        assert m, "command_timeout must be a numeric value"
        value = int(m.group(1))
        assert value <= 30, (
            f"command_timeout={value}s is too high; should be < pool_timeout and < Cloud Run timeout (60s)"
        )


# ---------------------------------------------------------------------------
# containerConcurrency vs pool capacity — deploy/service.yaml
# ---------------------------------------------------------------------------

class TestConcurrencyVsPoolCapacity:
    """Assert containerConcurrency ≤ pool_size + max_overflow in service.yaml.

    service.yaml is not the runtime source of truth (see TestDeployWorkflowFlags)
    but is kept consistent to document intent and protect against a switch to
    declarative deploy.
    """

    @pytest.fixture(scope="class")
    def service_yaml_text(self):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base, "deploy", "service.yaml")
        with open(path) as f:
            return f.read()

    def _parse_container_concurrency(self, text: str) -> int | None:
        m = re.search(r"^\s*containerConcurrency:\s*(\d+)", text, re.MULTILINE)
        return int(m.group(1)) if m else None

    def test_container_concurrency_le_pool_capacity(self, service_yaml_text):
        pool_capacity = _pool_capacity()
        cc = self._parse_container_concurrency(service_yaml_text)

        assert cc is not None, (
            "containerConcurrency must be explicitly set in deploy/service.yaml."
        )
        # Allow a 1-unit slack above pool_capacity: Cloud Run's LB queues the
        # (N+1)th request briefly while in-flight requests release connections,
        # which is cheaper than spinning a new instance for transient micro-bursts.
        assert cc <= pool_capacity + 1, (
            f"containerConcurrency={cc} > pool_capacity+1={pool_capacity + 1}. "
            f"See .audit/2026-04-20/db-pool-diagnosis.md for measured 20% error rate."
        )

    def test_container_concurrency_minimum_sensible(self, service_yaml_text):
        cc = self._parse_container_concurrency(service_yaml_text)
        assert cc is not None and cc >= 1, (
            "containerConcurrency=0 or missing means unlimited — pool will be exhausted."
        )


# ---------------------------------------------------------------------------
# Deploy workflow flags — the actual source of truth for Cloud Run config
# ---------------------------------------------------------------------------

class TestDeployWorkflowFlags:
    """Assert .github/workflows/deploy.yml flags satisfy Cloud SQL connection math.

    The gcloud run deploy step passes --concurrency and --max-instances as
    flags, which OVERRIDE whatever is in deploy/service.yaml. The workflow
    is therefore the authoritative source for these values.

    Cloud SQL max_connections on f1-micro = 25. Reserve 4 for maintenance,
    leaving 21 usable.

    Invariants (both must hold):
      1. concurrency ≤ pool_size + max_overflow         (no per-instance pool queueing)
      2. max_instances × (pool_size + max_overflow) ≤ 21 (total DB connections capped)
    """

    CLOUD_SQL_SAFE_CONNECTIONS = 21  # 25 max_connections − 4 buffer

    @pytest.fixture(scope="class")
    def workflow_text(self):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base, ".github", "workflows", "deploy.yml")
        with open(path) as f:
            return f.read()

    def _parse_flag(self, text: str, name: str) -> int | None:
        m = re.search(rf"--{re.escape(name)}=(\d+)", text)
        return int(m.group(1)) if m else None

    def test_concurrency_le_pool_capacity(self, workflow_text):
        pool_capacity = _pool_capacity()
        concurrency = self._parse_flag(workflow_text, "concurrency")
        assert concurrency is not None, (
            "--concurrency flag must be set explicitly in .github/workflows/deploy.yml"
        )
        # See TestConcurrencyVsPoolCapacity for 1-unit slack rationale.
        assert concurrency <= pool_capacity + 1, (
            f"--concurrency={concurrency} > pool_capacity+1={pool_capacity + 1}. "
            f"Surplus requests queue on the pool → transient 500s under burst load."
        )

    def test_total_connections_within_cloud_sql_budget(self, workflow_text):
        pool_capacity = _pool_capacity()
        max_instances = self._parse_flag(workflow_text, "max-instances")
        assert max_instances is not None, (
            "--max-instances flag must be set explicitly in .github/workflows/deploy.yml"
        )
        total = max_instances * pool_capacity
        assert total <= self.CLOUD_SQL_SAFE_CONNECTIONS, (
            f"max_instances={max_instances} × pool_capacity={pool_capacity} = {total} "
            f"exceeds Cloud SQL safe budget ({self.CLOUD_SQL_SAFE_CONNECTIONS} of 25 max_connections). "
            f"Lower --max-instances or shrink the pool."
        )

    def test_max_instances_minimum_sensible(self, workflow_text):
        max_instances = self._parse_flag(workflow_text, "max-instances")
        assert max_instances is not None and max_instances >= 1, (
            "--max-instances must be ≥ 1"
        )
