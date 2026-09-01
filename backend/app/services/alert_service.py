"""Initial alert engine and persistence."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
import re
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from backend.app.models import AlertRule, AlertTrigger
from backend.app.schemas.alerts import AlertCreate, AlertUpdate
from backend.app.services.legacy_adapter import legacy_adapter


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
        "delivery": {
            "status": "not_connected",
            "channel": "telegram",
        },
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


def list_alerts(db: Session, user_id: str) -> list[dict[str, Any]]:
    rows = db.execute(
        select(AlertRule)
        .where(AlertRule.user_id == user_id)
        .order_by(desc(AlertRule.created_at))
    ).scalars().all()
    return [serialize_alert(row) for row in rows]


_FX_LABELS = {
    "oficial": "Dólar Oficial",
    "mayorista": "Dólar Mayorista",
    "blue": "Dólar Blue",
    "mep": "Dólar MEP",
    "ccl": "Dólar CCL",
    "tarjeta": "Dólar Tarjeta",
    "cripto": "Dólar Cripto",
}
_METRIC_LABELS = {
    "riesgo": "Riesgo País",
    "inflacion": "Inflación Mensual",
    "reservas": "Reservas BCRA",
}


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _epoch_iso(value: Any) -> str | None:
    parsed = _finite_float(value)
    if parsed is None or parsed <= 0:
        return None
    try:
        return datetime.fromtimestamp(parsed, timezone.utc).isoformat().replace("+00:00", "Z")
    except (OverflowError, OSError, ValueError):
        return None


def _legacy_condition_label(condition: dict[str, Any]) -> str:
    condition_type = str(condition.get("type") or "").lower()
    if condition_type == "variation":
        indicator = condition.get("indicator") or {}
        asset = indicator.get("symbol") or indicator.get("type") or indicator.get("kind") or "indicador"
        return (
            f"{str(asset).upper()} {condition.get('window') or ''} "
            f"{condition.get('op') or ''} {condition.get('value') or ''}"
        ).strip()
    if condition_type == "cross":
        left = condition.get("left") or {}
        right = condition.get("right") or {}
        left_label = left.get("symbol") or left.get("type") or left.get("kind") or "indicador"
        right_label = right.get("symbol") or right.get("type") or right.get("kind") or "indicador"
        direction = "alcista" if condition.get("direction") == "up" else "bajista"
        return f"Cruce {direction}: {str(left_label).upper()} / {str(right_label).upper()}"
    return "Condición compuesta"


def _legacy_rule_identity(rule: dict[str, Any]) -> tuple[str, str | None, str | None]:
    kind = str(rule.get("kind") or "unknown").lower()
    if kind == "fx":
        fx_type = str(rule.get("type") or "").lower()
        side = str(rule.get("side") or "").lower()
        title = _FX_LABELS.get(fx_type, fx_type.upper() or "Dólar")
        return f"{title} · {side.capitalize() or 'Precio'}", fx_type or None, "ARS"
    if kind == "metric":
        metric_type = str(rule.get("type") or "").lower()
        unit = "pb" if metric_type == "riesgo" else "MUS$" if metric_type == "reservas" else "%"
        return _METRIC_LABELS.get(metric_type, metric_type.capitalize() or "Métrica"), metric_type or None, unit
    if kind == "crypto":
        symbol = str(rule.get("symbol") or "").upper()
        quote = str(rule.get("quote") or "USDT").upper()
        return symbol or "Cripto", symbol or None, quote
    if kind == "composite":
        logic = str(rule.get("logic") or "AND").upper()
        return f"Alerta compuesta {logic}", None, None
    symbol = str(rule.get("symbol") or "").upper()
    return symbol or "Instrumento", symbol or None, "ARS"


def serialize_legacy_alert(
    rule: dict[str, Any],
    index: int,
    state: dict[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    now_ts = now.timestamp()
    title, asset, unit = _legacy_rule_identity(rule)
    pause_until = _finite_float(rule.get("pause_until"))
    global_pause_until = _finite_float(state.get("global_pause_until"))
    paused_indefinitely = bool(rule.get("pause_indef")) or bool(state.get("global_pause"))
    effective_pause_until = None
    pause_scope = None
    if paused_indefinitely:
        status = "paused"
        pause_scope = "global" if state.get("global_pause") else "rule"
    elif global_pause_until is not None and global_pause_until > now_ts:
        status = "paused"
        pause_scope = "global"
        effective_pause_until = global_pause_until
    elif pause_until is not None and pause_until > now_ts:
        status = "paused"
        pause_scope = "rule"
        effective_pause_until = pause_until
    elif rule.get("armed") is False:
        status = "waiting_rearm"
    else:
        status = "active"

    kind = str(rule.get("kind") or "unknown").lower()
    defaults = state.get("defaults") or {}
    cooldown_seconds = _finite_float(rule.get("cooldown_secs"))
    if cooldown_seconds is None:
        cooldown_seconds = _finite_float(defaults.get("rearm_cooldown_seconds"))
    tolerance = _finite_float(defaults.get("price_tolerance_pct"))
    conditions = rule.get("conditions") if isinstance(rule.get("conditions"), list) else []

    return {
        "id": f"legacy-{index + 1}",
        "position": index + 1,
        "kind": kind,
        "title": title,
        "asset": asset,
        "operator": rule.get("op"),
        "target_value": _finite_float(rule.get("value")),
        "unit": unit,
        "mode": rule.get("mode"),
        "segment": rule.get("segment"),
        "status": status,
        "pause_scope": pause_scope,
        "pause_until": _epoch_iso(effective_pause_until),
        "armed": rule.get("armed") if isinstance(rule.get("armed"), bool) else None,
        "cooldown_seconds": int(cooldown_seconds) if cooldown_seconds is not None else None,
        "tolerance_pct": round(tolerance * 100, 4) if tolerance is not None else None,
        "last_triggered_at": _epoch_iso(rule.get("last_trigger_ts")),
        "last_trigger_value": _finite_float(rule.get("last_trigger_price")),
        "logic": str(rule.get("logic") or "").upper() or None,
        "conditions": [
            _legacy_condition_label(condition)
            for condition in conditions
            if isinstance(condition, dict)
        ],
        "source": "legacy.ALERTS.read_only",
        "write_surface": "telegram:/alertas_menu",
    }


def get_alert_center(db: Session, user_id: str) -> dict[str, Any]:
    drafts = list_alerts(db, user_id)
    legacy_result = legacy_adapter.inspect_alert_state(user_id)
    warnings = list(legacy_result.warnings)

    legacy_items: list[dict[str, Any]] = []
    legacy_state = legacy_result.data if isinstance(legacy_result.data, dict) else {}
    if legacy_result.status == "ok":
        legacy_items = [
            serialize_legacy_alert(rule, index, legacy_state)
            for index, rule in enumerate(legacy_state.get("rules") or [])
            if isinstance(rule, dict)
        ]

    daily = (legacy_state.get("subscription") or {}).get("daily")
    daily_time = str(daily) if daily is not None else None
    valid_daily_time = bool(
        daily_time
        and re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", daily_time)
    )
    subscription_status = legacy_result.status
    if legacy_result.status == "ok" and daily_time is not None and not valid_daily_time:
        subscription_status = "partial"
        warnings.append("El horario de suscripción legacy tiene un formato no reconocido")

    active_count = sum(item["status"] in {"active", "waiting_rearm"} for item in legacy_items)
    paused_count = sum(item["status"] == "paused" for item in legacy_items)
    if drafts:
        warnings.append("Los borradores SQLite no envían alertas por Telegram")

    overall_status = "ok" if legacy_result.status == "ok" else "partial"
    return {
        "status": overall_status,
        "source": "alert_center",
        "freshness": legacy_result.freshness if legacy_result.status == "ok" else "unknown",
        "warnings": warnings,
        "data": {
            "summary": {
                "legacy_count": len(legacy_items),
                "active_count": active_count,
                "paused_count": paused_count,
                "draft_count": len(drafts),
                "subscription_active": valid_daily_time,
            },
            "legacy": {
                "status": legacy_result.status,
                "source": legacy_result.source,
                "read_only": True,
                "items": legacy_items,
                "global_pause": bool(legacy_state.get("global_pause")),
                "global_pause_until": _epoch_iso(legacy_state.get("global_pause_until")),
                "manage_command": "/alertas_menu",
            },
            "drafts": {
                "status": "ok",
                "source": "sqlite.alert_drafts",
                "items": drafts,
            },
            "subscription": {
                "status": subscription_status,
                "source": legacy_result.source,
                "enabled": valid_daily_time,
                "daily_time": daily_time if valid_daily_time else None,
                "timezone": "America/Argentina/Buenos_Aires",
                "channel": "telegram",
                "manage_command": "/subs",
            },
            # Backward-compatible alias for clients built before the legacy view.
            "items": drafts,
        },
    }


def create_alert(db: Session, payload: AlertCreate, user_id: str) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    alert = AlertRule(
        user_id=user_id,
        category=payload.category,
        asset=payload.asset,
        metric=payload.metric,
        condition_type=payload.condition_type,
        operator=payload.operator,
        target_value=payload.target_value,
        prealert_tolerance_pct=payload.prealert_tolerance_pct,
        cooldown_minutes=payload.cooldown_minutes,
        confirmation_reads=1,
        channel="miniapp_draft",
        status="draft",
        created_at=now,
        updated_at=now,
    )
    db.add(alert)
    db.commit()
    db.refresh(alert)
    return serialize_alert(alert)


def _owned_alert(db: Session, alert_id: int, user_id: str) -> AlertRule | None:
    return db.execute(
        select(AlertRule)
        .where(AlertRule.id == alert_id)
        .where(AlertRule.user_id == user_id)
    ).scalars().first()


def update_alert(
    db: Session,
    alert_id: int,
    payload: AlertUpdate,
    user_id: str,
) -> dict[str, Any] | None:
    alert = _owned_alert(db, alert_id, user_id)
    if alert is None:
        return None
    for key, value in payload.model_dump(exclude_unset=True).items():
        setattr(alert, key, value)
    alert.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(alert)
    return serialize_alert(alert)


def delete_alert(db: Session, alert_id: int, user_id: str) -> bool:
    alert = _owned_alert(db, alert_id, user_id)
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


def record_alert_check(
    db: Session,
    alert_id: int,
    current_value: float,
    user_id: str,
    source: str = "miniapp_manual_check",
) -> dict[str, Any] | None:
    alert = _owned_alert(db, alert_id, user_id)
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
