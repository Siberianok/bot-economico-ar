from __future__ import annotations

from datetime import datetime

from .http import get_http_client
from .models import Quote

CRYPTOYA_DOLAR_URL = "https://criptoya.com/api/dolar"
DOLARAPI_BASE = "https://dolarapi.com/v1"


def _try_float(value: float | int | str | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _parse_crypto_item(name: str, payload: dict) -> Quote:
    buy = payload.get("bid") or payload.get("compra")
    sell = payload.get("ask") or payload.get("venta")
    ts = payload.get("time") or payload.get("last_update") or payload.get("timestamp")
    timestamp = datetime.fromtimestamp(ts) if isinstance(ts, (int, float)) else None
    return Quote(name=name, buy=_try_float(buy), sell=_try_float(sell), last_update=timestamp)


async def fetch_dolar_quotes() -> list[Quote]:
    http = await get_http_client()
    payload = await http.fetch_json(CRYPTOYA_DOLAR_URL)
    results: list[Quote] = []
    for name, raw in payload.items():
        if not isinstance(raw, dict):
            continue
        results.append(_parse_crypto_item(name, raw))
    return results


async def fetch_oficial_blue() -> list[Quote]:
    http = await get_http_client()
    names = ("oficial", "blue", "mep", "ccb", "ccl")
    results: list[Quote] = []
    for name in names:
        data = await http.fetch_json(f"{DOLARAPI_BASE}/cotizaciones/{name}")
        ts = data.get("fechaActualizacion")
        timestamp = datetime.fromisoformat(ts) if isinstance(ts, str) else None
        results.append(
            Quote(
                name=name,
                buy=_try_float(data.get("compra")),
                sell=_try_float(data.get("venta")),
                last_update=timestamp,
            )
        )
    return results
