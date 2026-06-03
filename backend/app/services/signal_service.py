"""Initial signal engine."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from backend.app.config import get_settings
from backend.app.models import Signal


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
        status="active",
        created_at=now,
    )
    db.add(signal)
    db.commit()
    db.refresh(signal)
    return serialize_signal(signal), True


def ensure_system_signal(db: Session) -> None:
    settings = get_settings()
    emit_signal(
        db,
        dedup_key="system:api-online",
        title="API backend operativa",
        description="Se inicializo el backend modular del Observatorio Economico.",
        score=max(settings.signal_min_score, 70),
        severity="info",
        type="system",
        source="backend",
        cooldown_minutes=24 * 60,
    )


def list_signals(db: Session) -> list[dict[str, Any]]:
    ensure_system_signal(db)
    rows = db.execute(select(Signal).order_by(desc(Signal.created_at))).scalars().all()
    return [serialize_signal(row) for row in rows]

