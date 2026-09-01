"""News routes."""

from __future__ import annotations

from fastapi import APIRouter, Query

from backend.app.services.news_service import get_news


router = APIRouter(prefix="/news", tags=["news"])


@router.get("")
async def news(limit: int = Query(default=12, ge=1, le=30)) -> dict:
    return await get_news(limit=limit)
