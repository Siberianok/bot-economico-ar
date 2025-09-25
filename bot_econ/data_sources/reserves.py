from __future__ import annotations

from datetime import datetime

from .http import get_http_client
from .models import ReserveStatus

LAMACRO_RESERVAS_URL = "https://www.lamacro.ar/variables/1"


async def fetch_reserves() -> ReserveStatus:
    http = await get_http_client()
    payload = await http.fetch_json(LAMACRO_RESERVAS_URL)
    latest = payload.get("data", [{}])[-1]
    total = _try_float(latest.get("valor"))
    variation = _try_float(latest.get("variacion_diaria"))
    date_str = latest.get("fecha")
    date = datetime.fromisoformat(date_str) if isinstance(date_str, str) else None
    return ReserveStatus(total=total, variation=variation, date=date)


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
