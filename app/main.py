import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_redoc_html
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
    docs_url=None,      # disable default — we serve custom dark theme
    redoc_url=None,     # disable default — we serve custom route
    lifespan=lifespan,
)


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui() -> HTMLResponse:
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Reporium API</title>
        <meta charset="utf-8"/>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link rel="stylesheet" type="text/css" href="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/5.11.0/swagger-ui.css" >
        <style>
            /* Base dark theme */
            html { box-sizing: border-box; overflow: -moz-scrollbars-vertical; overflow-y: scroll; }
            body { margin: 0; background: #1a1a2e; }

            /* Topbar */
            .swagger-ui .topbar {
                background: #16213e;
                border-bottom: 2px solid #7c3aed;
                padding: 8px 0;
            }
            .swagger-ui .topbar .download-url-wrapper { display: none; }
            .swagger-ui .topbar-wrapper {
                display: flex;
                align-items: center;
                padding: 0 20px;
            }
            .swagger-ui .topbar-wrapper img { display: none; }
            .swagger-ui .topbar-wrapper::before {
                content: 'Reporium API';
                color: #7c3aed;
                font-size: 1.4em;
                font-weight: 700;
                font-family: sans-serif;
                letter-spacing: 0.5px;
            }

            /* Main wrapper */
            .swagger-ui { background: #1a1a2e; color: #e2e8f0; font-family: sans-serif; }
            .swagger-ui .wrapper { max-width: 1460px; width: 100%; padding: 0 20px; }

            /* Info block */
            .swagger-ui .info {
                background: #16213e;
                border: 1px solid #7c3aed;
                border-radius: 8px;
                padding: 20px 30px;
                margin: 20px 0;
            }
            .swagger-ui .info .title {
                color: #7c3aed;
                font-size: 2em;
            }
            .swagger-ui .info p { color: #94a3b8; }
            .swagger-ui .info a { color: #7c3aed; }
            .swagger-ui .info .base-url { color: #64748b; }

            /* Scheme container */
            .swagger-ui .scheme-container {
                background: #16213e;
                border-bottom: 1px solid #2d3748;
                box-shadow: none;
                padding: 15px 0;
            }

            /* Operation tags */
            .swagger-ui .opblock-tag {
                color: #e2e8f0 !important;
                border-bottom: 1px solid #2d3748 !important;
                font-size: 1.1em;
            }
            .swagger-ui .opblock-tag:hover { background: rgba(124, 58, 237, 0.1) !important; }

            /* Operation blocks */
            .swagger-ui .opblock {
                border-radius: 6px;
                margin: 6px 0;
                box-shadow: none;
                border: 1px solid;
            }
            .swagger-ui .opblock.opblock-get {
                background: rgba(37, 99, 235, 0.15);
                border-color: #2563eb;
            }
            .swagger-ui .opblock.opblock-post {
                background: rgba(5, 150, 105, 0.15);
                border-color: #059669;
            }
            .swagger-ui .opblock.opblock-put {
                background: rgba(217, 119, 6, 0.15);
                border-color: #d97706;
            }
            .swagger-ui .opblock.opblock-delete {
                background: rgba(220, 38, 38, 0.15);
                border-color: #dc2626;
            }
            .swagger-ui .opblock-summary {
                padding: 8px 20px;
            }
            .swagger-ui .opblock-summary-method {
                border-radius: 4px;
                font-weight: 700;
                min-width: 70px;
                text-align: center;
            }
            .swagger-ui .opblock-summary-path {
                color: #e2e8f0 !important;
                font-weight: 500;
            }
            .swagger-ui .opblock-summary-description {
                color: #94a3b8 !important;
            }
            .swagger-ui .opblock-body {
                background: #0f172a;
                border-top: 1px solid #2d3748;
            }

            /* Authorize button */
            .swagger-ui .btn.authorize {
                background: transparent;
                border: 2px solid #7c3aed;
                color: #7c3aed;
                border-radius: 6px;
            }
            .swagger-ui .btn.authorize:hover {
                background: rgba(124, 58, 237, 0.1);
            }
            .swagger-ui .btn.authorize svg { fill: #7c3aed; }

            /* Execute button */
            .swagger-ui .btn.execute {
                background: #7c3aed;
                border-color: #7c3aed;
                border-radius: 6px;
                color: white;
            }
            .swagger-ui .btn.execute:hover { background: #6d28d9; }

            /* Inputs */
            .swagger-ui input[type=text],
            .swagger-ui input[type=password],
            .swagger-ui textarea,
            .swagger-ui select {
                background: #0f172a !important;
                color: #e2e8f0 !important;
                border: 1px solid #2d3748 !important;
                border-radius: 4px;
            }

            /* Parameters */
            .swagger-ui .parameters-col_description p { color: #94a3b8; }
            .swagger-ui .parameter__name { color: #7c3aed; font-weight: 600; }
            .swagger-ui .parameter__type { color: #38bdf8; }
            .swagger-ui .parameter__in { color: #64748b; font-style: italic; }
            .swagger-ui table thead tr td,
            .swagger-ui table thead tr th {
                color: #94a3b8;
                border-bottom: 1px solid #2d3748;
            }
            .swagger-ui .response-col_status { color: #e2e8f0; }
            .swagger-ui .response-col_description p { color: #94a3b8; }

            /* Models */
            .swagger-ui section.models {
                background: #16213e;
                border: 1px solid #2d3748;
                border-radius: 6px;
            }
            .swagger-ui section.models h4 { color: #e2e8f0; }
            .swagger-ui .model-box { background: #0f172a; }
            .swagger-ui .model { color: #e2e8f0; }
            .swagger-ui .model-title { color: #7c3aed; }
            .swagger-ui span.prop-type { color: #38bdf8; }
            .swagger-ui span.prop-format { color: #64748b; }

            /* Code / response */
            .swagger-ui .highlight-code { background: #0f172a; }
            .swagger-ui .microlight { background: #0f172a !important; color: #e2e8f0 !important; }
            .swagger-ui .responses-inner { background: #0f172a; }
            .swagger-ui .response-control-media-type__accept-message { color: #94a3b8; }

            /* Scrollbar */
            ::-webkit-scrollbar { width: 8px; height: 8px; }
            ::-webkit-scrollbar-track { background: #1a1a2e; }
            ::-webkit-scrollbar-thumb { background: #2d3748; border-radius: 4px; }
            ::-webkit-scrollbar-thumb:hover { background: #4a5568; }

            /* Copy button icons */
            .swagger-ui .copy-to-clipboard { background: #2d3748; border-radius: 4px; }
            .swagger-ui .copy-to-clipboard button { background: transparent; }
        </style>
    </head>
    <body>
        <div id="swagger-ui"></div>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/5.11.0/swagger-ui-bundle.js"> </script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/5.11.0/swagger-ui-standalone-preset.js"> </script>
        <script>
        window.onload = function() {
            window.ui = SwaggerUIBundle({
                url: "/openapi.json",
                dom_id: '#swagger-ui',
                deepLinking: true,
                displayRequestDuration: true,
                defaultModelsExpandDepth: 1,
                defaultModelExpandDepth: 1,
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIStandalonePreset
                ],
                plugins: [
                    SwaggerUIBundle.plugins.DownloadUrl
                ],
                layout: "StandaloneLayout"
            })
        }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(html)


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
