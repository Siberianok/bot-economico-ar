"""FastAPI entrypoint for Observatorio Economico."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api import (
    routes_alerts,
    routes_config,
    routes_health,
    routes_market,
    routes_portfolio,
    routes_projections,
    routes_screener,
    routes_signals,
)
from backend.app.config import get_settings
from backend.app.database import init_database
from backend.app.schemas.envelope import envelope
from backend.app.utils.errors import ApiError
from backend.app.utils.time import utc_now_iso


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Backend modular para Telegram Bot + Telegram Mini App del Observatorio Economico.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api_cors_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    init_database()


@app.exception_handler(ApiError)
async def _api_error_handler(_: Request, exc: ApiError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=envelope(
            status="error",
            timestamp=utc_now_iso(),
            source=exc.source,
            freshness="unknown",
            data=None,
            warnings=[exc.message],
        ),
    )


@app.exception_handler(Exception)
async def _unexpected_error_handler(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content=envelope(
            status="error",
            timestamp=utc_now_iso(),
            source="backend",
            freshness="unknown",
            data=None,
            warnings=[f"Error interno controlado: {exc.__class__.__name__}"],
        ),
    )


api_prefix = "/api/v1"
app.include_router(routes_health.router, prefix=api_prefix)
app.include_router(routes_market.router, prefix=api_prefix)
app.include_router(routes_screener.router, prefix=api_prefix)
app.include_router(routes_portfolio.router, prefix=api_prefix)
app.include_router(routes_alerts.router, prefix=api_prefix)
app.include_router(routes_signals.router, prefix=api_prefix)
app.include_router(routes_config.router, prefix=api_prefix)
app.include_router(routes_projections.projections_router, prefix=api_prefix)
app.include_router(routes_projections.validations_router, prefix=api_prefix)

