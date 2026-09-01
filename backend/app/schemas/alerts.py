"""Alert API schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AlertCreate(BaseModel):
    category: str = "market"
    asset: str = Field(min_length=1, max_length=80)
    metric: str = "price"
    condition_type: str = Field(default="exact", pattern="^(exact|approximation)$")
    operator: str = Field(default=">=", pattern="^(>=|<=|>|<|==)$")
    target_value: float = Field(allow_inf_nan=False)
    prealert_tolerance_pct: float | None = Field(
        default=None,
        ge=0,
        le=100,
        allow_inf_nan=False,
    )
    cooldown_minutes: int = Field(default=15, ge=0)


class AlertUpdate(BaseModel):
    category: str | None = None
    asset: str | None = Field(default=None, min_length=1, max_length=80)
    metric: str | None = None
    condition_type: str | None = Field(default=None, pattern="^(exact|approximation)$")
    operator: str | None = Field(default=None, pattern="^(>=|<=|>|<|==)$")
    target_value: float | None = Field(default=None, allow_inf_nan=False)
    prealert_tolerance_pct: float | None = Field(
        default=None,
        ge=0,
        le=100,
        allow_inf_nan=False,
    )
    cooldown_minutes: int | None = Field(default=None, ge=0)
    status: str | None = Field(default=None, pattern="^(draft|paused)$")


class AlertEvaluationInput(BaseModel):
    current_value: float = Field(allow_inf_nan=False)
