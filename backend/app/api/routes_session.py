"""Telegram Mini App session route."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.app.services.session_service import get_session_status
from backend.app.utils.telegram_auth import TelegramIdentity, get_telegram_identity


router = APIRouter(prefix="/session", tags=["session"])


@router.get("")
def session_get(
    identity: TelegramIdentity = Depends(get_telegram_identity),
) -> dict:
    return get_session_status(identity)
