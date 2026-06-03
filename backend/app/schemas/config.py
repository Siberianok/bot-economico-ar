"""Configuration API schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ConfigUpdate(BaseModel):
    base_currency: str | None = Field(default=None, pattern="^(ARS|USD)$")
    benchmark: str | None = None
    signal_min_score: int | None = Field(default=None, ge=0, le=100)
    theme: str | None = Field(default=None, pattern="^(dark|light)$")

