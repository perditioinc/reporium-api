"""Shared utility functions across routers."""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from app.config import settings

if TYPE_CHECKING:
    import anthropic

logger = logging.getLogger(__name__)
_logger = logging.getLogger(__name__)


def get_anthropic_key() -> str:
    """Get Anthropic API key from env or config."""
    key = os.getenv("ANTHROPIC_API_KEY") or getattr(settings, "anthropic_api_key", None)
    if not key:
        raise ValueError("ANTHROPIC_API_KEY not configured")
    return key


# KAN-197 / Issue #215: Lazy singleton Anthropic client shared across routers.
# Avoids creating a new client per request and keeps a single source of truth.
_anthropic_client: "anthropic.Anthropic | None" = None


def get_anthropic_client() -> "anthropic.Anthropic":
    """Return the process-wide lazy singleton Anthropic client."""
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        _anthropic_client = anthropic.Anthropic(api_key=get_anthropic_key())
    return _anthropic_client


def log_nonfatal(
    operation: str,
    *,
    session_id: str | None = None,
    extra_context: str | None = None,
) -> None:
    """Log a non-fatal exception from a fire-and-forget handler.

    Must be called from within an ``except`` block — uses ``exc_info=True``
    so the active exception traceback is included at WARNING level.
    """
    msg = f"{operation} failed (non-fatal)"
    if session_id:
        msg += f" for session {session_id}"
    if extra_context:
        msg += f" ({extra_context})"
    _logger.warning(msg, exc_info=True)


def vec_to_pg(vec) -> str:
    """Convert a float sequence (list or numpy array) to a pgvector-compatible string."""
    items = vec.tolist() if hasattr(vec, "tolist") else vec
    return "[" + ",".join(f"{x:.8f}" for x in items) + "]"
