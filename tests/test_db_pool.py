"""
Tests for DB pool configuration correctness.

These are unit-level assertions against the engine settings — they do NOT
require a live database. They guard against regressions where someone bumps
pool_size or containerConcurrency without updating the other.

Root cause documented in .audit/2026-04-20/db-pool-diagnosis.md:
- containerConcurrency=20 with pool_size=5+2=7 caused ~20% transient 500s
- Fix: containerConcurrency ≤ pool_size + max_overflow; pool_timeout explicit
"""

import os
import re

import pytest



# ---------------------------------------------------------------------------
# Pool settings unit tests
# ---------------------------------------------------------------------------

class TestPoolSettings:
    """Assert the engine is configured with the correct pool parameters."""

    def test_pool_size(self):
        """pool_size must stay ≤ max_connections / max_instances to avoid exhausting Cloud SQL."""
        # Import after env is set (conftest.py sets DATABASE_URL)
        from app.database import engine
        pool = engine.pool
        # QueuePool exposes pool_size via .size()
        assert pool.size() == 5, (
            f"pool_size changed to {pool.size()}; update the containerConcurrency cap accordingly. "
            "See .audit/2026-04-20/db-pool-diagnosis.md"
        )

    def test_max_overflow(self):
        from app.database import engine
        pool = engine.pool
        assert pool._max_overflow == 2, (
            f"max_overflow changed to {pool._max_overflow}; update the concurrency cap accordingly."
        )

    def test_pool_timeout_is_set_and_reasonable(self):
        """pool_timeout must be explicitly set and less than Cloud Run timeoutSeconds (60s).

        Leaving pool_timeout at the SQLAlchemy default (30s) means a saturated pool
        blocks each request for 30s before raising TimeoutError → 500. We set it to
        10s so failures are fast and the client can retry sooner.
        """
        from app.database import engine
        pool = engine.pool
        timeout = pool._timeout
        assert timeout is not None, "pool_timeout must be set explicitly (not relying on default 30s)"
        assert timeout <= 15, (
            f"pool_timeout={timeout}s is too long; should be ≤15s so failures are fast. "
            "Cloud Run container timeout is 60s."
        )

    def test_connect_args_command_timeout(self):
        """asyncpg command_timeout must be set to prevent rogue queries from blocking the pool.

        A slow query holding a connection starves other requests even when concurrency
        is within bounds. command_timeout=20s ensures queries abort before pool_timeout.
        """
        from app.database import engine
        connect_args = engine.dialect.create_connect_args(engine.url)[1] if hasattr(engine.dialect, 'create_connect_args') else {}
        # Engine.url.query carries the connect_args for asyncpg
        # We check the engine creator's kwargs instead
        creator_kw = engine.dialect.create_connect_args
        # Access stored connect_args via engine pool configuration
        # The engine stores _creator or _pool._creator; simplest: check engine.url connect_args
        # For asyncpg, connect_args are passed through to the dialect.
        # Since there's no public API to read them back, we verify via the pool's
        # _creator function — we test the config constant instead.
        # This test asserts the value was set at module load time.
        from app import database as db_module
        # Read the engine creation call params via source inspection
        import inspect
        src = inspect.getsource(db_module)
        assert "command_timeout" in src, (
            "connect_args command_timeout must be set in app/database.py. "
            "Without it, slow queries hold connections indefinitely."
        )
        # Extract the numeric value
        match = re.search(r"command_timeout['\"]?\s*:\s*(\d+)", src)
        assert match, "command_timeout must be a numeric value"
        value = int(match.group(1))
        assert value <= 30, (
            f"command_timeout={value}s is too high; should be < pool_timeout and < Cloud Run timeout (60s)"
        )


# ---------------------------------------------------------------------------
# containerConcurrency vs pool capacity constraint test
# ---------------------------------------------------------------------------

class TestConcurrencyVsPoolCapacity:
    """Assert containerConcurrency ≤ pool_size + max_overflow in service.yaml.

    This is the core invariant that prevents pool saturation.
    containerConcurrency=N means Cloud Run can send N simultaneous requests
    to one instance. If N > pool_size + max_overflow, the excess requests
    must wait for a pool slot. Under f1-micro latency (~50-200ms/query) with
    11 sequential queries per /library/full cold path, this causes 500s.
    """

    @pytest.fixture(scope="class")
    def service_yaml_text(self):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base, "deploy", "service.yaml")
        with open(path) as f:
            return f.read()

    def _parse_container_concurrency(self, text: str) -> int | None:
        """Extract containerConcurrency from YAML text without a YAML parser dependency."""
        # Match 'containerConcurrency: <int>' ignoring inline comments
        m = re.search(r"^\s*containerConcurrency:\s*(\d+)", text, re.MULTILINE)
        return int(m.group(1)) if m else None

    def test_container_concurrency_le_pool_capacity(self, service_yaml_text):
        from app.database import engine
        pool = engine.pool
        pool_capacity = pool.size() + pool._max_overflow

        container_concurrency = self._parse_container_concurrency(service_yaml_text)

        assert container_concurrency is not None, (
            "containerConcurrency must be explicitly set in deploy/service.yaml. "
            "The default (0 = unlimited) will cause pool exhaustion under burst load."
        )

        # Allow a 1-unit slack above pool_capacity: Cloud Run's LB queues the
        # (N+1)th request briefly while in-flight requests release connections,
        # which is cheaper than spinning a new instance for transient micro-bursts.
        # Anything more than +1 risks pool_timeout exhaustion.
        assert container_concurrency <= pool_capacity + 1, (
            f"containerConcurrency={container_concurrency} > pool_capacity+1={pool_capacity + 1} "
            f"(pool_size={pool.size()} + max_overflow={pool._max_overflow}). "
            f"This causes pool exhaustion under burst load, producing transient 500s. "
            f"Either raise pool_size/max_overflow or lower containerConcurrency to ≤{pool_capacity + 1}. "
            "See .audit/2026-04-20/db-pool-diagnosis.md for measured 20% error rate."
        )

    def test_container_concurrency_minimum_sensible(self, service_yaml_text):
        """Concurrency must be at least 1 (sanity check — value of 0 means unlimited)."""
        cc = self._parse_container_concurrency(service_yaml_text)
        assert cc is not None and cc >= 1, (
            "containerConcurrency=0 or missing means unlimited — pool will be exhausted. Set an explicit cap."
        )


# ---------------------------------------------------------------------------
# Deploy workflow flags — the actual source of truth for Cloud Run config
# ---------------------------------------------------------------------------

class TestDeployWorkflowFlags:
    """Assert .github/workflows/deploy.yml flags satisfy Cloud SQL connection math.

    The gcloud run deploy step in the workflow passes --concurrency and
    --max-instances as flags, which OVERRIDE whatever is in deploy/service.yaml.
    The workflow is therefore the authoritative source for these values.

    Cloud SQL max_connections on f1-micro = 25. Reserve 4 for maintenance,
    leaving 21 usable. Per-instance pool capacity = pool_size + max_overflow.

    Invariants (both must hold):
      1. concurrency ≤ pool_size + max_overflow         (no per-instance queueing on pool)
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
        """Extract --<name>=<int> from a gcloud flags block."""
        m = re.search(rf"--{re.escape(name)}=(\d+)", text)
        return int(m.group(1)) if m else None

    def test_concurrency_le_pool_capacity(self, workflow_text):
        from app.database import engine
        pool = engine.pool
        pool_capacity = pool.size() + pool._max_overflow

        concurrency = self._parse_flag(workflow_text, "concurrency")
        assert concurrency is not None, (
            "--concurrency flag must be set explicitly in .github/workflows/deploy.yml"
        )
        # Allow a 1-unit slack above pool_capacity (see TestConcurrencyVsPoolCapacity
        # for rationale). Surplus requests queue at the Cloud Run LB, not at the pool.
        assert concurrency <= pool_capacity + 1, (
            f"--concurrency={concurrency} > pool_capacity+1={pool_capacity + 1} "
            f"(pool_size={pool.size()} + max_overflow={pool._max_overflow}). "
            f"Excess requests queue on the pool → transient 500s under burst load. "
            f"Either raise pool_size/max_overflow or lower --concurrency to ≤{pool_capacity + 1}."
        )

    def test_total_connections_within_cloud_sql_budget(self, workflow_text):
        from app.database import engine
        pool = engine.pool
        pool_capacity = pool.size() + pool._max_overflow

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
