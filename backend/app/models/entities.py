"""SQLAlchemy entities for the new API layer."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AlertRule(Base):
    __tablename__ = "alert_rules"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[str] = mapped_column(String(128), index=True, default="local-dev")
    category: Mapped[str] = mapped_column(String(64), default="market")
    asset: Mapped[str] = mapped_column(String(64), index=True)
    metric: Mapped[str] = mapped_column(String(64), default="price")
    condition_type: Mapped[str] = mapped_column(String(32), default="exact")
    operator: Mapped[str] = mapped_column(String(8), default=">=")
    target_value: Mapped[float] = mapped_column(Float)
    prealert_tolerance_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    cooldown_minutes: Mapped[int] = mapped_column(Integer, default=15)
    confirmation_reads: Mapped[int] = mapped_column(Integer, default=1)
    channel: Mapped[str] = mapped_column(String(32), default="telegram")
    status: Mapped[str] = mapped_column(String(32), default="active")
    last_triggered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_checked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    trigger_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    triggers: Mapped[list["AlertTrigger"]] = relationship(
        "AlertTrigger", back_populates="alert", cascade="all, delete-orphan"
    )


class AlertTrigger(Base):
    __tablename__ = "alert_triggers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    alert_id: Mapped[int] = mapped_column(ForeignKey("alert_rules.id"), index=True)
    value: Mapped[float] = mapped_column(Float)
    reason: Mapped[str] = mapped_column(String(128))
    source: Mapped[str] = mapped_column(String(128), default="api")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    alert: Mapped[AlertRule] = relationship("AlertRule", back_populates="triggers")


class Signal(Base):
    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    dedup_key: Mapped[str] = mapped_column(String(160), index=True)
    type: Mapped[str] = mapped_column(String(64), default="system")
    title: Mapped[str] = mapped_column(String(160))
    description: Mapped[str] = mapped_column(Text, default="")
    severity: Mapped[str] = mapped_column(String(32), default="info")
    score: Mapped[int] = mapped_column(Integer, default=0)
    asset: Mapped[str | None] = mapped_column(String(64), nullable=True)
    metric: Mapped[str | None] = mapped_column(String(64), nullable=True)
    current_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    previous_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    variation: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String(128), default="backend")
    status: Mapped[str] = mapped_column(String(32), default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Projection(Base):
    __tablename__ = "projections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    symbol: Mapped[str] = mapped_column(String(64), index=True)
    horizon: Mapped[str] = mapped_column(String(16), default="3m")
    base_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    pessimistic: Mapped[float | None] = mapped_column(Float, nullable=True)
    base_case: Mapped[float | None] = mapped_column(Float, nullable=True)
    optimistic: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    base_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    target_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    validation_status: Mapped[str] = mapped_column(String(32), default="pending")
    source: Mapped[str] = mapped_column(String(128), default="api")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Validation(Base):
    __tablename__ = "validations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    projection_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    symbol: Mapped[str] = mapped_column(String(64), index=True)
    actual_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    absolute_error: Mapped[float | None] = mapped_column(Float, nullable=True)
    relative_error: Mapped[float | None] = mapped_column(Float, nullable=True)
    direction_hit: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    magnitude_hit: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AppPreference(Base):
    __tablename__ = "app_preferences"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[str] = mapped_column(String(128), index=True, default="local-dev")
    base_currency: Mapped[str] = mapped_column(String(16), default="ARS")
    benchmark: Mapped[str] = mapped_column(String(32), default="dolar")
    signal_min_score: Mapped[int] = mapped_column(Integer, default=70)
    theme: Mapped[str] = mapped_column(String(32), default="dark")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

