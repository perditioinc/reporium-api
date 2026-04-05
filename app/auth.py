import hashlib
import hmac
import logging
import os

from fastapi import HTTPException, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.security.api_key import APIKeyHeader
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token

from app.config import settings

logger = logging.getLogger(__name__)
security = HTTPBearer()

_IS_PRODUCTION = os.getenv("ENVIRONMENT") == "production"


def _secrets_equal(provided: str | None, expected: str | None) -> bool:
    """
    Constant-time comparison of two secret strings.

    hmac.compare_digest requires both operands to be non-empty bytes of equal
    length — otherwise it effectively returns False in constant time. We still
    guard against None / empty explicitly so callers can differentiate
    "missing credential" from "wrong credential" for HTTP status selection.
    """
    if not provided or not expected:
        return False
    return hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))


async def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> str:
    # Issue #237: timing-safe comparison of the bearer token against the
    # configured ingestion API key.
    if not _secrets_equal(credentials.credentials, settings.ingestion_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    return credentials.credentials


# ---------------------------------------------------------------------------
# Admin key - protects admin-only endpoints (POST /admin/*, /admin/taxonomy/*)
# ---------------------------------------------------------------------------

_ADMIN_KEY_HEADER = APIKeyHeader(name="X-Admin-Key", auto_error=False)


async def require_admin_key(
    x_admin_key: str | None = Security(_ADMIN_KEY_HEADER),
) -> None:
    admin_key = os.getenv("ADMIN_API_KEY")
    if not admin_key:
        if _IS_PRODUCTION:
            logger.error("ADMIN_API_KEY not set in production — rejecting request")
            raise HTTPException(status_code=500, detail="Server misconfiguration")
        return  # No key configured — allow in dev mode only
    # Issue #237: timing-safe comparison.
    if not _secrets_equal(x_admin_key, admin_key):
        raise HTTPException(status_code=403, detail="Invalid admin key")


# ---------------------------------------------------------------------------
# Ingest key - protects ingest pipeline endpoints (POST /ingest/*)
# Accept both X-Ingest-Key and the legacy X-Admin-Key for backward compatibility.
# ---------------------------------------------------------------------------

_INGEST_KEY_HEADER = APIKeyHeader(name="X-Ingest-Key", auto_error=False)
_LEGACY_INGEST_KEY_HEADER = APIKeyHeader(name="X-Admin-Key", auto_error=False)


async def require_ingest_key(
    x_ingest_key: str | None = Security(_INGEST_KEY_HEADER),
    x_admin_key: str | None = Security(_LEGACY_INGEST_KEY_HEADER),
) -> None:
    ingest_key = os.getenv("INGEST_API_KEY")
    if not ingest_key:
        if _IS_PRODUCTION:
            logger.error("INGEST_API_KEY not set in production — rejecting request")
            raise HTTPException(status_code=500, detail="Server misconfiguration")
        return  # No key configured — allow in dev mode only
    # Issue #237: timing-safe comparison for both headers. Both branches run
    # regardless of the first result so an attacker cannot distinguish which
    # header was matched via response timing.
    match_new = _secrets_equal(x_ingest_key, ingest_key)
    match_legacy = _secrets_equal(x_admin_key, ingest_key)
    if match_new or match_legacy:
        return
    raise HTTPException(status_code=403, detail="Invalid ingest key")


# ---------------------------------------------------------------------------
# App token - lightweight guard on expensive LLM endpoints
# ---------------------------------------------------------------------------

_APP_TOKEN_HEADER = APIKeyHeader(name="X-App-Token", auto_error=False)


async def require_app_token(
    x_app_token: str | None = Security(_APP_TOKEN_HEADER),
) -> None:
    """
    Lightweight app verification for expensive endpoints.
    The token is a simple shared secret set via APP_API_TOKEN env var.
    Not full auth — just prevents random scripts from hitting LLM endpoints.
    """
    expected = os.getenv("APP_API_TOKEN", "")
    if not expected:
        if _IS_PRODUCTION:
            logger.error("APP_API_TOKEN not set in production — rejecting")
            raise HTTPException(status_code=500, detail="Server misconfiguration")
        return  # Dev mode
    # Issue #237: timing-safe comparison.
    if not _secrets_equal(x_app_token, expected):
        raise HTTPException(status_code=403, detail="App token required")


def hash_app_token(token: str | None) -> str | None:
    """
    Return a stable SHA-256 hex digest of the X-App-Token, or None.

    Used by /intelligence/ask session-memory code (issue #235) to bind each
    ask_sessions row to the token that created it so one app token cannot
    read another's conversation history.
    """
    if not token:
        return None
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


async def get_app_token_hash(
    x_app_token: str | None = Security(_APP_TOKEN_HEADER),
) -> str | None:
    """
    FastAPI dependency: return the SHA-256 hex of the presented X-App-Token.

    Does NOT validate the token — `require_app_token` already does that via
    its own Depends() on the same endpoint. Returns None in dev mode when no
    token is required. Safe to use alongside `require_app_token`.
    """
    return hash_app_token(x_app_token)


async def require_pubsub_push(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Security(HTTPBearer(auto_error=False)),
) -> None:
    audience = settings.pubsub_audience
    if not audience:
        if _IS_PRODUCTION:
            logger.error("pubsub_audience not set in production — rejecting request")
            raise HTTPException(status_code=500, detail="Server misconfiguration")
        return  # No audience configured — allow in dev/manual mode only
    if credentials is None or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Missing Pub/Sub bearer token")

    try:
        id_token.verify_oauth2_token(
            credentials.credentials,
            google_requests.Request(),
            audience=audience,
        )
    except Exception as exc:
        raise HTTPException(status_code=403, detail="Invalid Pub/Sub bearer token") from exc

    # Pub/Sub push headers are optional in local/manual testing, so no additional
    # hard requirement is enforced once the OIDC token is valid for the audience.
    _ = request
