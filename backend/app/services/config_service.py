"""Runtime preference service."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.config import get_settings
from backend.app.models import AppPreference
from backend.app.schemas.config import ConfigUpdate


def _serialize(pref: AppPreference) -> dict[str, Any]:
    return {
        "user_id": pref.user_id,
        "base_currency": pref.base_currency,
        "benchmark": pref.benchmark,
        "signal_min_score": pref.signal_min_score,
        "theme": pref.theme,
        "updated_at": pref.updated_at.isoformat().replace("+00:00", "Z"),
    }


def get_or_create_preference(db: Session, user_id: str | None = None) -> dict[str, Any]:
    settings = get_settings()
    uid = user_id or settings.dev_telegram_user_id
    pref = db.execute(select(AppPreference).where(AppPreference.user_id == uid)).scalars().first()
    if pref is None:
        pref = AppPreference(
            user_id=uid,
            signal_min_score=settings.signal_min_score,
            updated_at=datetime.now(timezone.utc),
        )
        db.add(pref)
        db.commit()
        db.refresh(pref)
    return _serialize(pref)


def update_preference(db: Session, payload: ConfigUpdate, user_id: str | None = None) -> dict[str, Any]:
    settings = get_settings()
    uid = user_id or settings.dev_telegram_user_id
    pref = db.execute(select(AppPreference).where(AppPreference.user_id == uid)).scalars().first()
    if pref is None:
        get_or_create_preference(db, uid)
        pref = db.execute(select(AppPreference).where(AppPreference.user_id == uid)).scalars().one()
    for key, value in payload.model_dump(exclude_unset=True).items():
        setattr(pref, key, value)
    pref.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(pref)
    return _serialize(pref)

