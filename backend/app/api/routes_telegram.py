"""Telegram webhook route for the combined free-tier runtime."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Header

from backend.app.bot.telegram_bot import telegram_runtime
from backend.app.schemas.envelope import envelope
from backend.app.utils.errors import ApiError
from backend.app.utils.time import utc_now_iso


router = APIRouter(prefix="/telegram", tags=["telegram"])


@router.post("/webhook", include_in_schema=False)
async def telegram_webhook(
    payload: dict[str, Any],
    x_telegram_bot_api_secret_token: str | None = Header(
        default=None,
        alias="X-Telegram-Bot-Api-Secret-Token",
    ),
) -> dict[str, Any]:
    if not x_telegram_bot_api_secret_token or not telegram_runtime.matches_secret(
        x_telegram_bot_api_secret_token
    ):
        raise ApiError("Webhook no encontrado", status_code=404, source="telegram")
    if telegram_runtime.state != "running":
        raise ApiError("Telegram runtime no disponible", status_code=503, source="telegram")

    try:
        await telegram_runtime.enqueue_update(payload)
    except RuntimeError as exc:
        raise ApiError(str(exc), status_code=503, source="telegram") from exc

    return envelope(
        status="ok",
        timestamp=utc_now_iso(),
        source="telegram",
        freshness="current",
        data={"accepted": True},
        warnings=[],
    )
