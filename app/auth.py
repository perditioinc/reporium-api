import os

from fastapi import HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.security.api_key import APIKeyHeader

from app.config import settings

security = HTTPBearer()


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
# Admin key — protects admin-only endpoints (POST /admin/*, /admin/taxonomy/*)
# ---------------------------------------------------------------------------

_ADMIN_KEY_HEADER = APIKeyHeader(name="X-Admin-Key", auto_error=False)


async def require_admin_key(
    x_admin_key: str | None = Security(_ADMIN_KEY_HEADER),
) -> None:
    admin_key = os.getenv("ADMIN_API_KEY")
    if not admin_key:
        return  # No key configured — allow (dev mode)
    if x_admin_key != admin_key:
        raise HTTPException(status_code=403, detail="Invalid admin key")


# ---------------------------------------------------------------------------
# Ingest key — protects ingest pipeline endpoints (POST /ingest/repos, /enrich)
# Separate credential from admin key so the ingestion pipeline has its own token.
# ---------------------------------------------------------------------------

_INGEST_KEY_HEADER = APIKeyHeader(name="X-Admin-Key", auto_error=False)


async def require_ingest_key(
    x_admin_key: str | None = Security(_INGEST_KEY_HEADER),
) -> None:
    ingest_key = os.getenv("INGEST_API_KEY")
    if not ingest_key:
        return  # No key configured — allow (dev mode)
    if x_admin_key != ingest_key:
        raise HTTPException(status_code=403, detail="Invalid ingest key")
