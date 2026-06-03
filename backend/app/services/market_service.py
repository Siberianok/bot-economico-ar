"""Market pulse service."""

from __future__ import annotations

from typing import Any

from backend.app.schemas.envelope import envelope
from backend.app.services.legacy_adapter import legacy_adapter
from backend.app.utils.time import utc_now_iso


def _empty_market_blocks() -> dict[str, Any]:
    blocks = [
        "dolar_oficial",
        "dolar_blue",
        "dolar_mep",
        "dolar_ccl",
        "dolar_cripto",
        "reservas",
        "inflacion",
        "riesgo_pais",
        "bandas_cambiarias",
        "brechas",
        "signals",
        "news",
        "calendar",
    ]
    return {name: {"status": "not_available", "data": None} for name in blocks}


async def get_market_pulse() -> dict[str, Any]:
    result = await legacy_adapter.fetch_market_pulse()
    if result.status == "not_available" or not result.data:
        return envelope(
            status="not_available",
            timestamp=utc_now_iso(),
            source=result.source,
            freshness="unknown",
            data=_empty_market_blocks(),
            warnings=result.warnings or ["Funcion pendiente de adaptar"],
        )

    data = _empty_market_blocks()
    fx = result.data.get("fx") if isinstance(result.data, dict) else None
    if isinstance(fx, dict):
        mapping = {
            "oficial": "dolar_oficial",
            "blue": "dolar_blue",
            "mep": "dolar_mep",
            "ccl": "dolar_ccl",
            "cripto": "dolar_cripto",
        }
        for legacy_key, block_key in mapping.items():
            if legacy_key in fx:
                data[block_key] = {"status": "ok", "data": fx.get(legacy_key)}
    for legacy_key, block_key in (
        ("reservas", "reservas"),
        ("inflacion", "inflacion"),
        ("riesgo_pais", "riesgo_pais"),
        ("bandas", "bandas_cambiarias"),
    ):
        value = result.data.get(legacy_key)
        if value is not None:
            data[block_key] = {"status": "ok", "data": value}

    return envelope(
        status="partial" if result.warnings else "ok",
        timestamp=utc_now_iso(),
        source=result.source,
        freshness=result.freshness if result.freshness in {"current", "stale", "fallback", "unknown"} else "unknown",
        data=data,
        warnings=result.warnings,
    )

