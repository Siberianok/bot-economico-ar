"""Alert routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.schemas.alerts import AlertCreate, AlertEvaluationInput, AlertUpdate
from backend.app.schemas.envelope import envelope
from backend.app.services.alert_service import create_alert, delete_alert, list_alerts, record_alert_check, update_alert
from backend.app.utils.errors import ApiError
from backend.app.utils.time import utc_now_iso


router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.get("")
def alerts(db: Session = Depends(get_db)) -> dict:
    return envelope(
        status="ok",
        timestamp=utc_now_iso(),
        source="alert_service",
        freshness="current",
        data={"items": list_alerts(db)},
        warnings=[],
    )


@router.post("")
def alerts_create(payload: AlertCreate, db: Session = Depends(get_db)) -> dict:
    return envelope(
        status="ok",
        timestamp=utc_now_iso(),
        source="alert_service",
        freshness="current",
        data=create_alert(db, payload),
        warnings=["Canal Telegram preparado; envio real se integrara en fase siguiente"],
    )


@router.patch("/{alert_id}")
def alerts_update(alert_id: int, payload: AlertUpdate, db: Session = Depends(get_db)) -> dict:
    updated = update_alert(db, alert_id, payload)
    if updated is None:
        raise ApiError("Alerta no encontrada", status_code=404, source="alert_service")
    return envelope(
        status="ok",
        timestamp=utc_now_iso(),
        source="alert_service",
        freshness="current",
        data=updated,
        warnings=[],
    )


@router.delete("/{alert_id}")
def alerts_delete(alert_id: int, db: Session = Depends(get_db)) -> dict:
    deleted = delete_alert(db, alert_id)
    if not deleted:
        raise ApiError("Alerta no encontrada", status_code=404, source="alert_service")
    return envelope(
        status="ok",
        timestamp=utc_now_iso(),
        source="alert_service",
        freshness="current",
        data={"deleted": True, "id": alert_id},
        warnings=[],
    )


@router.post("/{alert_id}/check")
def alerts_check(alert_id: int, payload: AlertEvaluationInput, db: Session = Depends(get_db)) -> dict:
    result = record_alert_check(db, alert_id, payload.current_value, payload.source)
    if result is None:
        raise ApiError("Alerta no encontrada", status_code=404, source="alert_service")
    return envelope(
        status="ok",
        timestamp=utc_now_iso(),
        source="alert_service",
        freshness="current",
        data=result,
        warnings=[],
    )

