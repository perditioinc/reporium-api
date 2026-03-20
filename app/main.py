import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

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
    description="The central API for the Reporium platform — tracks 826+ AI development tools on GitHub",
    version="1.0.0",
    docs_url=None,      # disable default — we serve custom dark theme
    redoc_url=None,     # disable default — we serve custom route
    lifespan=lifespan,
)

# Mount static files for dark theme CSS
static_dir = Path(__file__).resolve().parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui() -> HTMLResponse:
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Reporium API",
        swagger_ui_parameters={
            "syntaxHighlight.theme": "monokai",
            "deepLinking": True,
            "displayRequestDuration": True,
            "defaultModelsExpandDepth": 1,
            "defaultModelExpandDepth": 1,
        },
        swagger_css_url="/static/swagger-dark.css",
        swagger_js_url="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/5.11.0/swagger-ui-bundle.js",
    )


@app.get("/redoc", include_in_schema=False)
async def custom_redoc() -> HTMLResponse:
    return get_redoc_html(
        openapi_url="/openapi.json",
        title="Reporium API — ReDoc",
        with_google_fonts=False,
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
