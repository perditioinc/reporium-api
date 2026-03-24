import json
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from app.cache import cache
from app.database import check_db_connection, engine
from app.routers import admin, analytics, ingest, intelligence, library, library_full, platform, repos, search, taxonomy, trends, wiki


class _JsonFormatter(logging.Formatter):
    """Emit each log record as a single-line JSON object for Cloud Run structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def _configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(_JsonFormatter())
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(handler)


_configure_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    await cache.connect()
    await check_db_connection()
    yield
    await cache.disconnect()
    await engine.dispose()


limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per hour", "30 per minute"],
    storage_uri="memory://",
)

app = FastAPI(
    title="Reporium API",
    description=(
        "The central API for the Reporium platform — tracks 1,400+ AI development tools on GitHub.\n\n"
        "## Rate Limits\n"
        "| Endpoint | Limit |\n"
        "|----------|-------|\n"
        "| `GET /repos`, `/stats`, `/library`, `/library/full` | 200/hour, 30/minute |\n"
        "| `GET /search` | 200/hour, 30/minute |\n"
        "| `POST /intelligence/ask`, `/intelligence/query` | 200/hour, 30/minute |\n"
        "| `POST /ingest/*` | 200/hour, 30/minute |\n"
        "| `GET /health` | No limit |\n\n"
        "Rate limit headers are included in every response.\n\n"
        "## Repo Lookups\n"
        "All repo endpoints use `owner/name` (e.g. `perditioinc/reporium-api`). "
        "Bare-name lookups are not supported."
    ),
    version="1.1.0",
    docs_url=None,      # disable default — we serve Scalar
    redoc_url=None,     # disable default — Scalar replaces both
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(GZipMiddleware, minimum_size=1000)  # compress responses > 1KB


@app.get("/docs", include_in_schema=False)
async def scalar_docs() -> HTMLResponse:
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Reporium API</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <style>
        body { margin: 0; background: #0a0a0f; }
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
            --scalar-scrollbar-color: #21262d;
            --scalar-scrollbar-color-hover: #30363d;
        }
        .light-mode, .dark-mode { color-scheme: dark; }
        .scalar-app, .scalar-api-reference { background: #0a0a0f !important; }
        .section-header { background: #0a0a0f !important; border-bottom: 1px solid #21262d !important; }
        .sidebar { border-right: 1px solid #21262d !important; }
        .endpoint-path { color: #ffffff !important; font-weight: 600 !important; }
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: #0a0a0f; }
        ::-webkit-scrollbar-thumb { background: #21262d; border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: #30363d; }
    </style>
</head>
<body>
    <script id="api-reference" data-url="/openapi.json"></script>
    <script src="https://cdn.jsdelivr.net/npm/@scalar/api-reference"></script>
    <script>
        document.addEventListener('scalar:loaded', function() {
            document.querySelector('.scalar-app')?.setAttribute('data-theme', 'dark');
        });
        // Force dark mode via Scalar config
        window.scalarConfig = {
            theme: 'purple',
            darkMode: true,
            showSidebar: true,
            searchHotKey: 'k'
        };
    </script>
</body>
</html>"""
    return HTMLResponse(html)


@app.get("/redoc", include_in_schema=False)
async def redoc_redirect() -> HTMLResponse:
    """Redirect /redoc to /docs — Scalar replaces both."""
    return HTMLResponse(
        '<html><head><meta http-equiv="refresh" content="0;url=/docs"></head></html>'
    )


_ALLOWED_ORIGINS = [
    "https://reporium.com",
    "https://www.reporium.com",
    "https://perditioinc.github.io",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info(
        "request",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
        },
    )
    return response


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-RateLimit-Policy"] = "200/hour;30/minute"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = (
        "default-src 'none'; "
        "script-src 'none'; "
        "style-src 'none'; "
        "frame-ancestors 'none'"
    )
    return response

app.include_router(library.router)
app.include_router(repos.router)
app.include_router(search.router)
app.include_router(analytics.router)
app.include_router(trends.router)
app.include_router(wiki.router)
app.include_router(platform.router)
app.include_router(ingest.router)
app.include_router(ingest.events_router)
app.include_router(intelligence.router)
app.include_router(library_full.router)
app.include_router(taxonomy.router, prefix="/taxonomy")
app.include_router(admin.router)


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
