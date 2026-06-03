"""Screener routes."""

from __future__ import annotations

from fastapi import APIRouter

from backend.app.services.screener_service import get_screener


router = APIRouter(prefix="/screener", tags=["screener"])


@router.get("")
def screener(kind: str) -> dict:
    return get_screener(kind)

