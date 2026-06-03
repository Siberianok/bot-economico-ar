"""Screener service for acciones and CEDEARs."""

from __future__ import annotations

from typing import Any

from backend.app.schemas.envelope import envelope
from backend.app.services.legacy_adapter import legacy_adapter
from backend.app.utils.errors import ApiError
from backend.app.utils.time import utc_now_iso


def get_screener(kind: str) -> dict[str, Any]:
    normalized = kind.strip().lower()
    if normalized not in {"acciones", "cedears"}:
        raise ApiError("kind debe ser 'acciones' o 'cedears'", status_code=400, source="screener_service")
    result = legacy_adapter.list_symbols(normalized)
    if result.status != "ok" or not result.data:
        return envelope(
            status="not_available",
            timestamp=utc_now_iso(),
            source=result.source,
            freshness="unknown",
            data=None,
            warnings=result.warnings or ["Funcion pendiente de adaptar"],
        )
    items = [
        {
            "symbol": symbol,
            "status": "not_available",
            "price": None,
            "returns": {"1m": None, "3m": None, "6m": None},
            "score": None,
            "momentum": None,
            "risk": None,
            "liquidity": None,
            "projection": {"3m": None, "6m": None},
            "warnings": ["Metricas live pendientes de adaptar de forma segura"],
        }
        for symbol in result.data
    ]
    return envelope(
        status="partial",
        timestamp=utc_now_iso(),
        source=result.source,
        freshness="unknown",
        data={"kind": normalized, "items": items},
        warnings=["Listado conectado a legacy; metricas y rankings live pendientes de adaptar"],
    )

