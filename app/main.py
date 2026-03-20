import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

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
    docs_url=None,      # disable default — we serve Scalar
    redoc_url=None,     # disable default — Scalar replaces both
    lifespan=lifespan,
)


@app.get("/docs", include_in_schema=False)
async def scalar_docs() -> HTMLResponse:
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Reporium API</title>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <style>
            body { margin: 0; background: #0a0a0f; }
        </style>
    </head>
    <body>
        <script
            id="api-reference"
            data-url="/openapi.json"
            data-configuration='{
                "theme": "purple",
                "darkMode": true,
                "layout": "modern",
                "showSidebar": true,
                "searchHotKey": "k",
                "customCss": "
                    :root {
                        --scalar-background-1: #0a0a0f;
                        --scalar-background-2: #0d1117;
                        --scalar-background-3: #161b22;
                        --scalar-background-4: #1c2128;
                        --scalar-border-color: #21262d;

                        --scalar-color-1: #ffffff;
                        --scalar-color-2: #cdd9e5;
                        --scalar-color-3: #8b949e;
                        --scalar-color-accent: #a78bfa;

                        --scalar-sidebar-background-1: #0a0a0f;
                        --scalar-sidebar-background-2: #0d1117;
                        --scalar-sidebar-color-1: #ffffff;
                        --scalar-sidebar-color-2: #8b949e;
                        --scalar-sidebar-border-color: #21262d;
                        --scalar-sidebar-item-hover-background: #161b22;
                        --scalar-sidebar-item-active-background: #1c2128;

                        --scalar-button-1: #a78bfa;
                        --scalar-button-1-color: #ffffff;
                        --scalar-button-1-hover: #8b5cf6;

                        --scalar-color-green: #3fb950;
                        --scalar-color-red: #f85149;
                        --scalar-color-yellow: #d29922;
                        --scalar-color-blue: #58a6ff;
                        --scalar-color-orange: #db6d28;
                        --scalar-color-purple: #a78bfa;

                        --scalar-code-language-color-supersede: #cdd9e5;
                        --scalar-scrollbar-color: #21262d;
                        --scalar-scrollbar-color-hover: #30363d;
                    }

                    .light-mode, .dark-mode {
                        color-scheme: dark;
                    }

                    .scalar-app, .scalar-api-reference {
                        background: #0a0a0f !important;
                    }

                    .section-header {
                        background: #0a0a0f !important;
                        border-bottom: 1px solid #21262d !important;
                    }

                    .sidebar {
                        border-right: 1px solid #21262d !important;
                    }

                    .endpoint-path {
                        color: #ffffff !important;
                        font-weight: 600 !important;
                    }

                    .method-get { background: rgba(56, 189, 248, 0.15) !important; color: #38bdf8 !important; border: 1px solid #38bdf8 !important; border-radius: 4px; }
                    .method-post { background: rgba(63, 185, 80, 0.15) !important; color: #3fb950 !important; border: 1px solid #3fb950 !important; border-radius: 4px; }
                    .method-put { background: rgba(210, 153, 34, 0.15) !important; color: #d29922 !important; border: 1px solid #d29922 !important; border-radius: 4px; }
                    .method-delete { background: rgba(248, 81, 73, 0.15) !important; color: #f85149 !important; border: 1px solid #f85149 !important; border-radius: 4px; }
                    .method-patch { background: rgba(167, 139, 250, 0.15) !important; color: #a78bfa !important; border: 1px solid #a78bfa !important; border-radius: 4px; }

                    code, .code-block, pre {
                        background: #0d1117 !important;
                        border: 1px solid #21262d !important;
                        color: #cdd9e5 !important;
                        border-radius: 6px !important;
                    }

                    ::-webkit-scrollbar { width: 6px; height: 6px; }
                    ::-webkit-scrollbar-track { background: #0a0a0f; }
                    ::-webkit-scrollbar-thumb { background: #21262d; border-radius: 3px; }
                    ::-webkit-scrollbar-thumb:hover { background: #30363d; }
                "
            }'>
        </script>
        <script src="https://cdn.jsdelivr.net/npm/@scalar/api-reference"></script>
    </body>
    </html>
    """
    return HTMLResponse(html)


@app.get("/redoc", include_in_schema=False)
async def redoc_redirect() -> HTMLResponse:
    """Redirect /redoc to /docs — Scalar replaces both."""
    return HTMLResponse(
        '<html><head><meta http-equiv="refresh" content="0;url=/docs"></head></html>'
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
