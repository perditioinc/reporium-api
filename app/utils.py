"""Shared utility functions across routers."""
import logging
import os

from app.config import settings

logger = logging.getLogger(__name__)


def get_anthropic_key() -> str:
    """Get Anthropic API key from env or config."""
    key = os.getenv("ANTHROPIC_API_KEY") or getattr(settings, "anthropic_api_key", None)
    if not key:
        raise ValueError("ANTHROPIC_API_KEY not configured")
    return key


def vec_to_pg(vec) -> str:
    """Convert a float sequence (list or numpy array) to a pgvector-compatible string."""
    items = vec.tolist() if hasattr(vec, "tolist") else vec
    return "[" + ",".join(f"{x:.8f}" for x in items) + "]"
