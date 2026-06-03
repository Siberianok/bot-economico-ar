"""Shared response envelope schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


Status = Literal["ok", "partial", "not_available", "error"]
Freshness = Literal["current", "stale", "fallback", "unknown"]


class ApiEnvelope(BaseModel):
    status: Status
    timestamp: str
    source: str
    freshness: Freshness
    data: Any = None
    warnings: list[str] = Field(default_factory=list)


def envelope(
    *,
    status: Status,
    timestamp: str,
    source: str,
    freshness: Freshness,
    data: Any = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "timestamp": timestamp,
        "source": source,
        "freshness": freshness,
        "data": data,
        "warnings": warnings or [],
    }

