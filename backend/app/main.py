"""FastAPI entrypoint for ObeconBot."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api import (
    routes_alerts,
    routes_config,
    routes_health,
    routes_market,
    routes_news,
    routes_portfolio,
    routes_projections,
    routes_screener,
    routes_session,
    routes_signals,
    routes_summary,
    routes_telegram,
)
from backend.app.bot.telegram_bot import telegram_runtime
from backend.app.config import get_settings
from backend.app.database import init_database
from backend.app.schemas.envelope import envelope
from backend.app.utils.errors import ApiError
from backend.app.utils.time import utc_now_iso


settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_database()
    await telegram_runtime.start()
    try:
        yield
    finally:
        await telegram_runtime.stop()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Backend modular para Telegram Bot + Telegram Mini App de ObeconBot.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api_cors_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def root_health() -> dict[str, str]:
    return {"status": "ok", "service": settings.app_name}


@app.get("/healthz", include_in_schema=False)
def liveness() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz", include_in_schema=False)
def readiness() -> JSONResponse:
    telegram = telegram_runtime.status()
    ready = not telegram["enabled"] or telegram["state"] == "running"
    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "status": "ready" if ready else "not_ready",
            "telegram": telegram["state"],
        },
    )


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


@app.exception_handler(RequestValidationError)
async def _validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    first_error = exc.errors()[0] if exc.errors() else {}
    location = ".".join(str(part) for part in first_error.get("loc", []) if part != "body")
    message = str(first_error.get("msg", "datos inválidos"))
    detail = (
        f"Solicitud inválida en {location}: {message}"
        if location
        else f"Solicitud inválida: {message}"
    )
    return JSONResponse(
        status_code=422,
        content=envelope(
            status="error",
            timestamp=utc_now_iso(),
            source="validation",
            freshness="unknown",
            data=None,
            warnings=[detail],
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
app.include_router(routes_news.router, prefix=api_prefix)
app.include_router(routes_summary.router, prefix=api_prefix)
app.include_router(routes_screener.router, prefix=api_prefix)
app.include_router(routes_session.router, prefix=api_prefix)
app.include_router(routes_portfolio.router, prefix=api_prefix)
app.include_router(routes_alerts.router, prefix=api_prefix)
app.include_router(routes_signals.router, prefix=api_prefix)
app.include_router(routes_config.router, prefix=api_prefix)
app.include_router(routes_projections.projections_router, prefix=api_prefix)
app.include_router(routes_projections.validations_router, prefix=api_prefix)
app.include_router(routes_telegram.router)
