"""Screener service for acciones and CEDEARs."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from backend.app.schemas.envelope import envelope
from backend.app.services.legacy_adapter import legacy_adapter
from backend.app.utils.errors import ApiError
from backend.app.utils.time import utc_now_iso


def _empty_item(symbol: str, warning: str) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "status": "not_available",
        "price": None,
        "returns": {"1m": None, "3m": None, "6m": None},
        "score": None,
        "momentum": None,
        "risk": None,
        "liquidity": None,
        "projection": {"3m": None, "6m": None},
        "warnings": [warning],
    }


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _relative_scores(metrics: dict[str, Any]) -> dict[str, float]:
    ranked: list[tuple[str, float]] = []
    for symbol, values in metrics.items():
        if not isinstance(values, dict):
            continue
        performance = _number(values.get("6m_ars"))
        if performance is None:
            performance = _number(values.get("6m"))
        if performance is not None:
            ranked.append((str(symbol), performance))
    ranked.sort(key=lambda row: row[1])
    if not ranked:
        return {}
    if len(ranked) == 1:
        return {ranked[0][0]: 100.0}
    denominator = len(ranked) - 1
    return {
        symbol: round(index / denominator * 100.0, 1)
        for index, (symbol, _) in enumerate(ranked)
    }


def _freshness(last_timestamp: Any) -> str:
    timestamp = _number(last_timestamp)
    if timestamp is None:
        return "unknown"
    age_seconds = datetime.now(timezone.utc).timestamp() - timestamp
    return "current" if age_seconds <= 7 * 24 * 3600 else "stale"


async def get_screener(kind: str) -> dict[str, Any]:
    normalized = kind.strip().lower()
    if normalized not in {"acciones", "cedears"}:
        raise ApiError("kind debe ser 'acciones' o 'cedears'", status_code=400, source="screener_service")

    symbols_result = legacy_adapter.list_symbols(normalized)
    if symbols_result.status != "ok" or not symbols_result.data:
        return envelope(
            status="not_available",
            timestamp=utc_now_iso(),
            source=symbols_result.source,
            freshness="unknown",
            data=None,
            warnings=symbols_result.warnings or ["Funcion pendiente de adaptar"],
        )

    metrics_result = await legacy_adapter.fetch_screener_metrics(normalized)
    if metrics_result.status != "ok" or not metrics_result.data:
        warning = "Metricas live pendientes de adaptar de forma segura"
        items = [_empty_item(symbol, warning) for symbol in symbols_result.data]
        return envelope(
            status="partial",
            timestamp=utc_now_iso(),
            source=symbols_result.source,
            freshness="unknown",
            data={"kind": normalized, "items": items},
            warnings=metrics_result.warnings or [warning],
        )

    raw_metrics = metrics_result.data.get("metrics", {})
    if not isinstance(raw_metrics, dict):
        raw_metrics = {}
    scores = _relative_scores(raw_metrics)
    items: list[dict[str, Any]] = []
    unavailable_count = 0
    for symbol in symbols_result.data:
        values = raw_metrics.get(symbol)
        if not isinstance(values, dict):
            items.append(_empty_item(symbol, "Sin metricas reales para este instrumento"))
            unavailable_count += 1
            continue

        returns = {
            "1m": _number(values.get("1m")),
            "3m": _number(values.get("3m")),
            "6m": _number(values.get("6m")),
        }
        if normalized == "cedears":
            for horizon in ("1m", "3m", "6m"):
                returns[f"{horizon}_ars"] = _number(values.get(f"{horizon}_ars"))
                returns[f"{horizon}_usd"] = _number(values.get(f"{horizon}_usd"))
                returns[f"{horizon}_fx"] = _number(values.get(f"{horizon}_fx"))

        has_real_data = _number(values.get("last_px")) is not None or any(
            value is not None for value in returns.values()
        )
        if not has_real_data:
            unavailable_count += 1
        volatility = _number(values.get("vol_ann"))
        projections = legacy_adapter.calculate_projection_centers(values)
        items.append(
            {
                "symbol": symbol,
                "status": "ok" if has_real_data else "not_available",
                "price": _number(values.get("last_px")),
                "returns": returns,
                "score": scores.get(symbol),
                "momentum": returns.get("1m_ars") or returns.get("1m"),
                "risk": f"{volatility:.1f}% anual" if volatility is not None else None,
                "liquidity": None,
                "projection": projections,
                "warnings": [] if has_real_data else ["Sin metricas reales para este instrumento"],
            }
        )

    items.sort(
        key=lambda item: item["score"] if item["score"] is not None else -1.0,
        reverse=True,
    )
    warnings = [
        "Score relativo: percentil del rendimiento real a 6 meses dentro del universo consultado.",
        "Liquidez pendiente de una fuente real consistente.",
        "Proyecciones calculadas por el modelo legacy; no constituyen recomendacion de inversion.",
    ]
    if unavailable_count:
        warnings.append(f"{unavailable_count} instrumentos no devolvieron metricas reales.")

    freshness = _freshness(metrics_result.data.get("last_timestamp"))
    return envelope(
        status="partial",
        timestamp=utc_now_iso(),
        source=metrics_result.source,
        freshness=freshness,
        data={"kind": normalized, "items": items},
        warnings=warnings,
    )
