"""Projection API schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ProjectionCreate(BaseModel):
    symbol: str
    horizon: str = Field(default="3m", pattern="^(3m|6m)$")
    base_price: float | None = None
    pessimistic: float | None = None
    base_case: float | None = None
    optimistic: float | None = None
    confidence: float | None = Field(default=None, ge=0, le=100)
    base_date: datetime | None = None
    target_date: datetime | None = None
    source: str = "api"

