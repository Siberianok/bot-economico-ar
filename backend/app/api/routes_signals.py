"""Signal routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.schemas.envelope import envelope
from backend.app.services.signal_service import list_signals, refresh_market_signals
from backend.app.utils.time import utc_now_iso


router = APIRouter(prefix="/signals", tags=["signals"])


@router.get("")
async def signals(db: Session = Depends(get_db)) -> dict:
    refresh = await refresh_market_signals(db)
    items = list_signals(db)
    return envelope(
        status="ok" if items else "not_available",
        timestamp=utc_now_iso(),
        source="signal_service",
        freshness=refresh["freshness"] if items else "unknown",
        data={"items": items},
        warnings=refresh["warnings"],
    )
