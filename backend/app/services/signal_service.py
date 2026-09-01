"""Initial signal engine."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from backend.app.models import Signal
from backend.app.services.market_service import get_market_pulse
from backend.app.services.market_signal_rules import (
    derive_market_signal_candidates as _derive_market_signal_candidates,
)


RETIRED_SYNTHETIC_SIGNAL_KEYS = {"system:api-online"}


def _serialize_dt(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat().replace("+00:00", "Z")


def serialize_signal(signal: Signal) -> dict[str, Any]:
    return {
        "id": signal.id,
        "dedup_key": signal.dedup_key,
        "type": signal.type,
        "title": signal.title,
        "description": signal.description,
        "severity": signal.severity,
        "score": signal.score,
        "asset": signal.asset,
        "metric": signal.metric,
        "current_value": signal.current_value,
        "previous_value": signal.previous_value,
        "variation": signal.variation,
        "source": signal.source,
        "status": signal.status,
        "timestamp": _serialize_dt(signal.created_at),
    }


def emit_signal(
    db: Session,
    *,
    dedup_key: str,
    title: str,
    description: str,
    score: int,
    severity: str = "info",
    type: str = "system",
    source: str = "backend",
    cooldown_minutes: int = 60,
    asset: str | None = None,
    metric: str | None = None,
    current_value: float | None = None,
    previous_value: float | None = None,
    variation: float | None = None,
) -> tuple[dict[str, Any], bool]:
    now = datetime.now(timezone.utc)
    recent = db.execute(
        select(Signal)
        .where(Signal.dedup_key == dedup_key)
        .where(Signal.created_at >= now - timedelta(minutes=cooldown_minutes))
        .order_by(desc(Signal.created_at))
    ).scalars().first()
    if recent is not None:
        return serialize_signal(recent), False
    signal = Signal(
        dedup_key=dedup_key,
        title=title,
        description=description,
        score=score,
        severity=severity,
        type=type,
        source=source,
        asset=asset,
        metric=metric,
        current_value=current_value,
        previous_value=previous_value,
        variation=variation,
        status="active",
        created_at=now,
    )
    db.add(signal)
    db.commit()
    db.refresh(signal)
    return serialize_signal(signal), True


def list_signals(db: Session) -> list[dict[str, Any]]:
    rows = db.execute(
        select(Signal)
        .where(~Signal.dedup_key.in_(RETIRED_SYNTHETIC_SIGNAL_KEYS))
        .order_by(desc(Signal.created_at))
    ).scalars().all()
    return [serialize_signal(row) for row in rows]


def derive_market_signal_candidates(pulse: dict[str, Any]) -> list[dict[str, Any]]:
    """Compatibility wrapper kept for existing imports and tests."""
    return _derive_market_signal_candidates(pulse)


async def refresh_market_signals(db: Session) -> dict[str, Any]:
    pulse = await get_market_pulse()
    if pulse.get("status") in {"not_available", "error"}:
        return {
            "status": "not_available",
            "freshness": "unknown",
            "created": 0,
            "warnings": list(pulse.get("warnings") or ["Market Pulse no disponible"]),
        }

    created = 0
    for candidate in derive_market_signal_candidates(pulse):
        _, was_created = emit_signal(db, **candidate)
        created += int(was_created)
    return {
        "status": "ok",
        "freshness": pulse.get("freshness", "unknown"),
        "created": created,
        "warnings": [
            "Senales derivadas por reglas transparentes sobre datos reales; no son recomendaciones de inversion."
        ],
    }
