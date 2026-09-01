"""Health routes."""

from __future__ import annotations

from fastapi import APIRouter

from backend.app.config import get_settings
from backend.app.bot.telegram_bot import telegram_runtime
from backend.app.schemas.envelope import envelope
from backend.app.utils.time import utc_now_iso


router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
def health() -> dict:
    settings = get_settings()
    telegram = telegram_runtime.status()
    degraded = telegram["enabled"] and telegram["state"] != "running"
    return envelope(
        status="partial" if degraded else "ok",
        timestamp=utc_now_iso(),
        source="backend",
        freshness="current",
        data={
            "app": settings.app_name,
            "version": settings.app_version,
            "telegram": telegram,
        },
        warnings=["Telegram runtime no operativo"] if degraded else [],
    )
