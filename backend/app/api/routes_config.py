"""Configuration routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.config import get_settings
from backend.app.database import get_db
from backend.app.schemas.config import ConfigUpdate
from backend.app.schemas.envelope import envelope
from backend.app.services.config_service import get_or_create_preference, update_preference
from backend.app.utils.time import utc_now_iso


router = APIRouter(prefix="/config", tags=["config"])


@router.get("")
def config_get(db: Session = Depends(get_db)) -> dict:
    settings = get_settings()
    return envelope(
        status="ok",
        timestamp=utc_now_iso(),
        source="config_service",
        freshness="current",
        data={
            "api": {
                "app": settings.app_name,
                "version": settings.app_version,
                "miniapp_url": settings.miniapp_url,
                "legacy_live_fetch_enabled": settings.legacy_live_fetch_enabled,
                "dev_telegram_user_id": settings.dev_telegram_user_id,
            },
            "preferences": get_or_create_preference(db),
        },
        warnings=[],
    )


@router.patch("")
def config_patch(payload: ConfigUpdate, db: Session = Depends(get_db)) -> dict:
    return envelope(
        status="ok",
        timestamp=utc_now_iso(),
        source="config_service",
        freshness="current",
        data=update_preference(db, payload),
        warnings=[],
    )

