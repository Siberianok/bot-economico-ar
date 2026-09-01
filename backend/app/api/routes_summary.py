"""Daily summary routes."""

from __future__ import annotations

from fastapi import APIRouter

from backend.app.services.summary_service import get_daily_summary


router = APIRouter(prefix="/summary", tags=["summary"])


@router.get("")
async def daily_summary() -> dict:
    return await get_daily_summary()
