"""
Lightweight circuit breaker for external API calls (no extra dependencies).

States
------
CLOSED    — normal operation; failures are counted.
OPEN      — fast-fail; raises HTTP 503 until the cooldown expires.
HALF_OPEN — one probe request is allowed through to test recovery.

Default thresholds (overridable via constructor):
  failure_threshold : 5  consecutive failures → OPEN
  recovery_timeout  : 60 seconds in OPEN before moving to HALF_OPEN
"""

import logging
import time
import threading
from enum import Enum

from fastapi import HTTPException

logger = logging.getLogger(__name__)


class _State(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreaker:
    """
    Thread-safe circuit breaker.

    Usage::

        cb = CircuitBreaker()

        with cb:
            result = some_external_call()
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        name: str = "circuit_breaker",
    ) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._name = name

        self._state = _State.CLOSED
        self._failure_count = 0
        self._opened_at: float | None = None

        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public context-manager interface
    # ------------------------------------------------------------------

    def __enter__(self) -> "CircuitBreaker":
        self._before_call()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is None:
            self._on_success()
        elif not issubclass(exc_type, HTTPException):
            # Only count non-HTTP errors as circuit-breaker failures.
            # HTTPException (400, 422, …) reflects bad input, not a service outage.
            self._on_failure()
        # Never suppress the exception — let it propagate.
        return False

    # ------------------------------------------------------------------
    # Internal state machine
    # ------------------------------------------------------------------

    def _before_call(self) -> None:
        with self._lock:
            if self._state is _State.CLOSED:
                return  # happy path

            if self._state is _State.OPEN:
                elapsed = time.monotonic() - (self._opened_at or 0)
                if elapsed >= self._recovery_timeout:
                    logger.info(
                        "[%s] Recovery timeout elapsed — moving to HALF_OPEN", self._name
                    )
                    self._state = _State.HALF_OPEN
                    # Fall through: allow this probe request
                else:
                    remaining = int(self._recovery_timeout - elapsed)
                    logger.warning(
                        "[%s] OPEN — fast-failing (retry in ~%ds)", self._name, remaining
                    )
                    raise HTTPException(
                        status_code=503,
                        detail="AI service temporarily unavailable — try again shortly",
                    )

            # HALF_OPEN: allow the probe through (no extra action here)

    def _on_success(self) -> None:
        with self._lock:
            if self._state is _State.HALF_OPEN:
                logger.info("[%s] Probe succeeded — resetting to CLOSED", self._name)
            self._state = _State.CLOSED
            self._failure_count = 0
            self._opened_at = None

    def _on_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            logger.warning(
                "[%s] Failure recorded (%d/%d)",
                self._name,
                self._failure_count,
                self._failure_threshold,
            )

            if self._state is _State.HALF_OPEN or self._failure_count >= self._failure_threshold:
                self._state = _State.OPEN
                self._opened_at = time.monotonic()
                logger.error(
                    "[%s] Tripped OPEN after %d failure(s)", self._name, self._failure_count
                )

    # ------------------------------------------------------------------
    # Introspection helpers (useful for health checks / tests)
    # ------------------------------------------------------------------

    @property
    def state(self) -> str:
        return self._state.value

    @property
    def failure_count(self) -> int:
        return self._failure_count


# ---------------------------------------------------------------------------
# Module-level singleton — shared across all requests in a process
# ---------------------------------------------------------------------------
anthropic_breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=60.0,
    name="anthropic",
)
