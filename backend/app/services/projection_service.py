"""Projection persistence service."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from backend.app.models import Projection, Validation
from backend.app.schemas.projections import ProjectionCreate


def _serialize_dt(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat().replace("+00:00", "Z")


def _target_date(base_date: datetime, horizon: str) -> datetime:
    days = 90 if horizon == "3m" else 180
    return base_date + timedelta(days=days)


def serialize_projection(projection: Projection) -> dict[str, Any]:
    return {
        "id": projection.id,
        "symbol": projection.symbol,
        "horizon": projection.horizon,
        "base_price": projection.base_price,
        "scenarios": {
            "pessimistic": projection.pessimistic,
            "base": projection.base_case,
            "optimistic": projection.optimistic,
        },
        "confidence": projection.confidence,
        "base_date": _serialize_dt(projection.base_date),
        "target_date": _serialize_dt(projection.target_date),
        "validation_status": projection.validation_status,
        "source": projection.source,
        "created_at": _serialize_dt(projection.created_at),
    }


def create_projection(db: Session, payload: ProjectionCreate) -> dict[str, Any]:
    base_date = payload.base_date or datetime.now(timezone.utc)
    target_date = payload.target_date or _target_date(base_date, payload.horizon)
    projection = Projection(
        symbol=payload.symbol,
        horizon=payload.horizon,
        base_price=payload.base_price,
        pessimistic=payload.pessimistic,
        base_case=payload.base_case,
        optimistic=payload.optimistic,
        confidence=payload.confidence,
        base_date=base_date,
        target_date=target_date,
        validation_status="pending",
        source=payload.source,
    )
    db.add(projection)
    db.commit()
    db.refresh(projection)
    return serialize_projection(projection)


def list_projections(db: Session) -> list[dict[str, Any]]:
    rows = db.execute(select(Projection).order_by(desc(Projection.created_at))).scalars().all()
    return [serialize_projection(row) for row in rows]


def serialize_validation(validation: Validation) -> dict[str, Any]:
    return {
        "id": validation.id,
        "projection_id": validation.projection_id,
        "symbol": validation.symbol,
        "actual_return": validation.actual_return,
        "absolute_error": validation.absolute_error,
        "relative_error": validation.relative_error,
        "direction_hit": validation.direction_hit,
        "magnitude_hit": validation.magnitude_hit,
        "status": validation.status,
        "created_at": _serialize_dt(validation.created_at),
    }


def list_validations(db: Session) -> list[dict[str, Any]]:
    rows = db.execute(select(Validation).order_by(desc(Validation.created_at))).scalars().all()
    return [serialize_validation(row) for row in rows]

