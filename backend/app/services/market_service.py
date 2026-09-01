"""Market pulse service."""

from __future__ import annotations

from typing import Any

from backend.app.schemas.envelope import envelope
from backend.app.services.legacy_adapter import legacy_adapter
from backend.app.services.market_signal_rules import derive_market_signal_candidates
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


def _number(value: Any) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def _normalize_macro_value(name: str, value: Any) -> dict[str, Any] | None:
    """Translate legacy tuples into a stable API object contract."""
    if isinstance(value, dict):
        return value
    if not isinstance(value, (list, tuple)) or not value:
        numeric = _number(value)
        return {"value": numeric} if numeric is not None else None

    numeric = _number(value[0])
    if numeric is None:
        return None
    as_of = str(value[1]) if len(value) > 1 and value[1] is not None else None
    if name == "reservas":
        return {
            "value": numeric,
            "as_of": as_of,
            "previous_value": _number(value[2]) if len(value) > 2 else None,
            "from_cache": bool(value[3]) if len(value) > 3 else False,
        }
    return {
        "value": numeric,
        "as_of": as_of,
        "variation_pct": _number(value[2]) if len(value) > 2 else None,
        "from_cache": bool(value[3]) if name == "riesgo_pais" and len(value) > 3 else False,
    }


def _fx_sale(fx: dict[str, Any], key: str) -> float | None:
    row = fx.get(key)
    if not isinstance(row, dict):
        return None
    value = _number(row.get("venta"))
    return float(value) if value is not None else None


def _fx_gaps(fx: dict[str, Any]) -> dict[str, float] | None:
    official = _fx_sale(fx, "oficial")
    if official is None or official <= 0:
        return None
    gaps: dict[str, float] = {}
    for key in ("blue", "mep", "ccl", "cripto"):
        sale = _fx_sale(fx, key)
        if sale is not None:
            gaps[f"{key}_pct"] = round((sale / official - 1.0) * 100.0, 2)
    return gaps or None


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
        gaps = _fx_gaps(fx)
        if gaps:
            data["brechas"] = {
                "status": "ok",
                "data": gaps,
            }
    normalized_warnings = list(result.warnings)
    used_cache = False
    for legacy_key, block_key in (
        ("reservas", "reservas"),
        ("inflacion", "inflacion"),
        ("riesgo_pais", "riesgo_pais"),
        ("bandas", "bandas_cambiarias"),
    ):
        value = result.data.get(legacy_key)
        if value is not None:
            normalized = _normalize_macro_value(block_key, value)
            if normalized is None:
                normalized_warnings.append(
                    f"{legacy_key} devolvio un formato legacy no reconocido"
                )
                continue
            block_freshness = "stale" if normalized.get("from_cache") else "current"
            used_cache = used_cache or block_freshness == "stale"
            data[block_key] = {
                "status": "ok",
                "freshness": block_freshness,
                "data": normalized,
            }

    normalized_warnings.append(
        "Noticias se consultan por el endpoint dedicado; calendario economico pendiente."
    )
    pulse = envelope(
        status="partial",
        timestamp=utc_now_iso(),
        source=result.source,
        freshness="stale" if used_cache else "current",
        data=data,
        warnings=normalized_warnings,
    )
    readings = derive_market_signal_candidates(pulse)
    data["signals"] = {
        "status": "ok" if readings else "not_available",
        "data": readings,
    }
    return pulse
