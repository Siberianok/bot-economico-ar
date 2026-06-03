"""Market routes."""

from __future__ import annotations

from fastapi import APIRouter

from backend.app.services.market_service import get_market_pulse


router = APIRouter(prefix="/market", tags=["market"])


@router.get("/pulse")
async def market_pulse() -> dict:
    return await get_market_pulse()

