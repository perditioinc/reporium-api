import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.cache import cache
from app.database import engine
from app.routers import ingest, library, platform, repos, search, trends, wiki

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await cache.connect()
    yield
    await cache.disconnect()
    await engine.dispose()


app = FastAPI(
    title="Reporium API",
    description="The central API for the Reporium platform — tracks 818+ AI development tools on GitHub",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(library.router)
app.include_router(repos.router)
app.include_router(search.router)
app.include_router(trends.router)
app.include_router(wiki.router)
app.include_router(platform.router)
app.include_router(ingest.router)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)


@app.get("/health")
async def health() -> dict:
    from sqlalchemy import text
    from app.database import async_session_factory
    from app.models.trend import IngestionLog
    from sqlalchemy import select

    db_ok = False
    try:
        async with async_session_factory() as session:
            await session.execute(text("SELECT 1"))
            db_ok = True
    except Exception as e:
        logger.warning(f"DB health check failed: {e}")

    last_ingestion = None
    if db_ok:
        try:
            async with async_session_factory() as session:
                stmt = (
                    select(IngestionLog)
                    .order_by(IngestionLog.started_at.desc())
                    .limit(1)
                )
                result = await session.execute(stmt)
                log = result.scalar_one_or_none()
                if log:
                    last_ingestion = {
                        "started_at": log.started_at.isoformat(),
                        "status": log.status,
                        "mode": log.mode,
                    }
        except Exception:
            pass

    return {
        "status": "ok" if db_ok else "degraded",
        "database": "ok" if db_ok else "error",
        "cache": "ok" if await cache.is_available() else "disabled",
        "last_ingestion": last_ingestion,
    }
