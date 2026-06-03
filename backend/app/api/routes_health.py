"""Health routes."""

from __future__ import annotations

from fastapi import APIRouter

from backend.app.config import get_settings
from backend.app.schemas.envelope import envelope
from backend.app.utils.time import utc_now_iso


router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
def health() -> dict:
    settings = get_settings()
    return envelope(
        status="ok",
        timestamp=utc_now_iso(),
        source="backend",
        freshness="current",
        data={"app": settings.app_name, "version": settings.app_version},
        warnings=[],
    )

