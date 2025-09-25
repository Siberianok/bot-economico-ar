from __future__ import annotations

from datetime import datetime

from aiohttp import ClientResponseError

from .http import get_http_client
from .models import Quote

CRYPTOYA_DOLAR_URL = "https://criptoya.com/api/dolar"
DOLARAPI_BASE = "https://dolarapi.com/v1"

# Endpoints published by dolarapi.com for the main exchange rates we display in the bot.
# The API renamed several resources (e.g. ``mep`` -> ``bolsa`` and ``ccl`` ->
# ``contadoconliqui``).  Requesting the legacy slugs now returns ``404`` and the
# old code kept retrying until failing loudly during the prewarm task.  Sticking
# to the official slugs avoids the error altogether while preserving the
# original categories shown in the Telegram summary.
DOLARAPI_SLUGS = ("oficial", "blue", "bolsa", "contadoconliqui", "cripto")


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
    results: list[Quote] = []
    for slug in DOLARAPI_SLUGS:
        try:
            data = await http.fetch_json(f"{DOLARAPI_BASE}/dolares/{slug}")
        except ClientResponseError as exc:
            if exc.status == 404:
                # Skip gracefully if the provider removes a rate. We rely on the
                # API's official slugs so this should only trigger when they
                # deprecate one of them.
                continue
            raise
        ts = data.get("fechaActualizacion")
        timestamp = datetime.fromisoformat(ts) if isinstance(ts, str) else None
        results.append(
            Quote(
                name=str(data.get("nombre") or slug),
                buy=_try_float(data.get("compra")),
                sell=_try_float(data.get("venta")),
                last_update=timestamp,
            )
        )
    return results
