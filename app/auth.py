from fastapi import HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

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
