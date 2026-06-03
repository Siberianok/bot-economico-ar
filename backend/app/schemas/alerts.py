"""Alert API schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AlertCreate(BaseModel):
    user_id: str = "local-dev"
    category: str = "market"
    asset: str
    metric: str = "price"
    condition_type: str = Field(default="exact", pattern="^(exact|approximation)$")
    operator: str = Field(default=">=", pattern="^(>=|<=|>|<|==)$")
    target_value: float
    prealert_tolerance_pct: float | None = None
    cooldown_minutes: int = Field(default=15, ge=0)
    confirmation_reads: int = Field(default=1, ge=1)
    channel: str = "telegram"
    status: str = Field(default="active", pattern="^(active|paused)$")


class AlertUpdate(BaseModel):
    category: str | None = None
    asset: str | None = None
    metric: str | None = None
    condition_type: str | None = Field(default=None, pattern="^(exact|approximation)$")
    operator: str | None = Field(default=None, pattern="^(>=|<=|>|<|==)$")
    target_value: float | None = None
    prealert_tolerance_pct: float | None = None
    cooldown_minutes: int | None = Field(default=None, ge=0)
    confirmation_reads: int | None = Field(default=None, ge=1)
    channel: str | None = None
    status: str | None = Field(default=None, pattern="^(active|paused)$")


class AlertEvaluationInput(BaseModel):
    current_value: float
    source: str = "api"

