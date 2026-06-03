"""Signal routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.schemas.envelope import envelope
from backend.app.services.signal_service import list_signals
from backend.app.utils.time import utc_now_iso


router = APIRouter(prefix="/signals", tags=["signals"])


@router.get("")
def signals(db: Session = Depends(get_db)) -> dict:
    return envelope(
        status="ok",
        timestamp=utc_now_iso(),
        source="signal_service",
        freshness="current",
        data={"items": list_signals(db)},
        warnings=["Senales financieras live pendientes de conectar a fuentes reales"],
    )

