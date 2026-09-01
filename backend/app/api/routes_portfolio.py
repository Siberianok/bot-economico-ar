"""Portfolio routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.app.services.portfolio_service import get_portfolio_summary
from backend.app.utils.telegram_auth import TelegramIdentity, get_telegram_identity


router = APIRouter(prefix="/portfolio", tags=["portfolio"])


@router.get("/summary")
async def portfolio_summary(
    identity: TelegramIdentity = Depends(get_telegram_identity),
) -> dict:
    return await get_portfolio_summary(identity.user_id)
