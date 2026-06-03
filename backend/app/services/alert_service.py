"""Initial alert engine and persistence."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from backend.app.models import AlertRule, AlertTrigger
from backend.app.schemas.alerts import AlertCreate, AlertUpdate


def _serialize_dt(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat().replace("+00:00", "Z")


def serialize_alert(alert: AlertRule, include_history: bool = True) -> dict[str, Any]:
    return {
        "id": alert.id,
        "user_id": alert.user_id,
        "category": alert.category,
        "asset": alert.asset,
        "metric": alert.metric,
        "condition_type": alert.condition_type,
        "operator": alert.operator,
        "target_value": alert.target_value,
        "prealert_tolerance_pct": alert.prealert_tolerance_pct,
        "cooldown_minutes": alert.cooldown_minutes,
        "confirmation_reads": alert.confirmation_reads,
        "channel": alert.channel,
        "status": alert.status,
        "last_triggered_at": _serialize_dt(alert.last_triggered_at),
        "last_checked_at": _serialize_dt(alert.last_checked_at),
        "last_value": alert.last_value,
        "trigger_count": alert.trigger_count,
        "created_at": _serialize_dt(alert.created_at),
        "updated_at": _serialize_dt(alert.updated_at),
        "history": [
            {
                "id": trigger.id,
                "value": trigger.value,
                "reason": trigger.reason,
                "source": trigger.source,
                "created_at": _serialize_dt(trigger.created_at),
            }
            for trigger in sorted(alert.triggers, key=lambda item: item.created_at, reverse=True)
        ]
        if include_history
        else [],
    }


def list_alerts(db: Session) -> list[dict[str, Any]]:
    rows = db.execute(select(AlertRule).order_by(desc(AlertRule.created_at))).scalars().all()
    return [serialize_alert(row) for row in rows]


def create_alert(db: Session, payload: AlertCreate) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    alert = AlertRule(
        user_id=payload.user_id,
        category=payload.category,
        asset=payload.asset,
        metric=payload.metric,
        condition_type=payload.condition_type,
        operator=payload.operator,
        target_value=payload.target_value,
        prealert_tolerance_pct=payload.prealert_tolerance_pct,
        cooldown_minutes=payload.cooldown_minutes,
        confirmation_reads=payload.confirmation_reads,
        channel=payload.channel,
        status=payload.status,
        created_at=now,
        updated_at=now,
    )
    db.add(alert)
    db.commit()
    db.refresh(alert)
    return serialize_alert(alert)


def update_alert(db: Session, alert_id: int, payload: AlertUpdate) -> dict[str, Any] | None:
    alert = db.get(AlertRule, alert_id)
    if alert is None:
        return None
    for key, value in payload.model_dump(exclude_unset=True).items():
        setattr(alert, key, value)
    alert.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(alert)
    return serialize_alert(alert)


def delete_alert(db: Session, alert_id: int) -> bool:
    alert = db.get(AlertRule, alert_id)
    if alert is None:
        return False
    db.delete(alert)
    db.commit()
    return True


def _compare(value: float, operator: str, target: float) -> bool:
    if operator == ">=":
        return value >= target
    if operator == "<=":
        return value <= target
    if operator == ">":
        return value > target
    if operator == "<":
        return value < target
    if operator == "==":
        return abs(value - target) < 1e-9
    return False


def evaluate_alert(alert: AlertRule, current_value: float, *, now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    if alert.status != "active":
        return {"triggered": False, "reason": "paused", "cooldown": False}
    if alert.last_triggered_at is not None and alert.cooldown_minutes > 0:
        last = alert.last_triggered_at
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        if now - last < timedelta(minutes=alert.cooldown_minutes):
            return {"triggered": False, "reason": "cooldown", "cooldown": True}
    if alert.condition_type == "approximation":
        tolerance = alert.prealert_tolerance_pct if alert.prealert_tolerance_pct is not None else 0.0
        if alert.target_value == 0:
            distance_pct = 0.0 if current_value == 0 else 100.0
        else:
            distance_pct = abs((current_value - alert.target_value) / alert.target_value) * 100.0
        return {
            "triggered": distance_pct <= tolerance,
            "reason": "approximation" if distance_pct <= tolerance else "outside_tolerance",
            "distance_pct": distance_pct,
            "cooldown": False,
        }
    triggered = _compare(current_value, alert.operator, alert.target_value)
    return {
        "triggered": triggered,
        "reason": "exact" if triggered else "condition_not_met",
        "cooldown": False,
    }


def record_alert_check(db: Session, alert_id: int, current_value: float, source: str = "api") -> dict[str, Any] | None:
    alert = db.get(AlertRule, alert_id)
    if alert is None:
        return None
    now = datetime.now(timezone.utc)
    evaluation = evaluate_alert(alert, current_value, now=now)
    alert.last_checked_at = now
    alert.last_value = current_value
    alert.updated_at = now
    if evaluation.get("triggered"):
        alert.last_triggered_at = now
        alert.trigger_count += 1
        db.add(AlertTrigger(alert_id=alert.id, value=current_value, reason=str(evaluation["reason"]), source=source))
    db.commit()
    db.refresh(alert)
    return {"alert": serialize_alert(alert), "evaluation": evaluation}

