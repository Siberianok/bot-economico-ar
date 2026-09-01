"""Screener routes."""

from __future__ import annotations

from fastapi import APIRouter

from backend.app.services.screener_service import get_screener


router = APIRouter(prefix="/screener", tags=["screener"])


@router.get("")
async def screener(kind: str) -> dict:
    return await get_screener(kind)
