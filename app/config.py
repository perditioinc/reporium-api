from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database
    database_url: str
    database_provider: str = "local"  # local | supabase | gcp

    # Redis (optional — graceful fallback if not set)
    redis_url: str | None = None
    cache_ttl_seconds: int = 300  # 5 minutes default

    # Auth
    ingestion_api_key: str

    # GitHub (for any direct API calls)
    gh_token: str | None = None
    gh_username: str = "perditioinc"

    # Mode
    environment: str = "development"  # development, production

    class Config:
        env_file = ".env"


settings = Settings()
