import logging
import os

from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


def _get_secret(secret_id: str, project_id: str = "perditio-platform") -> str:
    """Read a secret from GCP Secret Manager."""
    from google.cloud import secretmanager

    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")


def _resolve_database_url() -> str:
    """Resolve DATABASE_URL from env or Secret Manager in production."""
    env_url = os.getenv("DATABASE_URL")
    if env_url:
        return env_url

    if os.getenv("ENVIRONMENT") == "production":
        project = os.getenv("GCP_PROJECT", "perditio-platform")
        logger.info("Loading DATABASE_URL from Secret Manager")
        return _get_secret("reporium-db-url-async", project)

    return "postgresql+asyncpg://postgres:postgres@localhost:5432/reporium"


def _resolve_secrets():
    """Resolve secrets from env or Secret Manager in production."""
    if os.getenv("ENVIRONMENT") == "production":
        project = os.getenv("GCP_PROJECT", "perditio-platform")
        if not os.getenv("DATABASE_URL"):
            logger.info("Loading DATABASE_URL from Secret Manager")
            os.environ["DATABASE_URL"] = _get_secret("reporium-db-url-async", project)
        if not os.getenv("INGESTION_API_KEY"):
            logger.info("Loading INGESTION_API_KEY from Secret Manager")
            os.environ["INGESTION_API_KEY"] = _get_secret("reporium-ingestion-api-key", project)
        if not os.getenv("ANTHROPIC_API_KEY"):
            try:
                logger.info("Loading ANTHROPIC_API_KEY from Secret Manager")
                os.environ["ANTHROPIC_API_KEY"] = _get_secret("anthropic-api-key", project).strip()
            except Exception:
                logger.warning("ANTHROPIC_API_KEY not found in Secret Manager")
    os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/reporium")


_resolve_secrets()


class Settings(BaseSettings):
    # Database
    database_url: str
    database_provider: str = "local"  # local | supabase | gcp | neon

    # Redis (optional — graceful fallback if not set)
    redis_url: str | None = None
    cache_ttl_seconds: int = 300  # 5 minutes default

    # Auth
    ingestion_api_key: str

    # GitHub (for any direct API calls)
    gh_token: str | None = None
    gh_username: str = "perditioinc"

    # GCP
    gcp_project: str = "perditio-platform"

    # Mode
    environment: str = "development"  # development, production

    class Config:
        env_file = ".env"


settings = Settings()
