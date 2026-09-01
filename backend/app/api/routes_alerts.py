"""Alert routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.schemas.alerts import AlertCreate, AlertEvaluationInput, AlertUpdate
from backend.app.schemas.envelope import envelope
from backend.app.services.alert_service import (
    create_alert,
    delete_alert,
    get_alert_center,
    record_alert_check,
    update_alert,
)
from backend.app.utils.errors import ApiError
from backend.app.utils.telegram_auth import TelegramIdentity, get_telegram_identity
from backend.app.utils.time import utc_now_iso


router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.get("")
def alerts(
    db: Session = Depends(get_db),
    identity: TelegramIdentity = Depends(get_telegram_identity),
) -> dict:
    center = get_alert_center(db, identity.user_id)
    return envelope(
        status=center["status"],
        timestamp=utc_now_iso(),
        source=center["source"],
        freshness=center["freshness"],
        data=center["data"],
        warnings=center["warnings"],
    )


@router.post("")
def alerts_create(
    payload: AlertCreate,
    db: Session = Depends(get_db),
    identity: TelegramIdentity = Depends(get_telegram_identity),
) -> dict:
    return envelope(
        status="ok",
        timestamp=utc_now_iso(),
        source="alert_service",
        freshness="current",
        data=create_alert(db, payload, identity.user_id),
        warnings=["Borrador guardado; delivery Telegram no conectado"],
    )


@router.patch("/{alert_id}")
def alerts_update(
    alert_id: int,
    payload: AlertUpdate,
    db: Session = Depends(get_db),
    identity: TelegramIdentity = Depends(get_telegram_identity),
) -> dict:
    updated = update_alert(db, alert_id, payload, identity.user_id)
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
def alerts_delete(
    alert_id: int,
    db: Session = Depends(get_db),
    identity: TelegramIdentity = Depends(get_telegram_identity),
) -> dict:
    deleted = delete_alert(db, alert_id, identity.user_id)
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
def alerts_check(
    alert_id: int,
    payload: AlertEvaluationInput,
    db: Session = Depends(get_db),
    identity: TelegramIdentity = Depends(get_telegram_identity),
) -> dict:
    result = record_alert_check(db, alert_id, payload.current_value, identity.user_id)
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
