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


async def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> str:
    if credentials.credentials != settings.ingestion_api_key:
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
    if x_admin_key != admin_key:
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
    if x_ingest_key == ingest_key or x_admin_key == ingest_key:
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
    if x_app_token != expected:
        raise HTTPException(status_code=403, detail="App token required")


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
