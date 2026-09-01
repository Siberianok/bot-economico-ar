"""Portfolio summary service."""

from __future__ import annotations

import math
from time import time
from typing import Any

from backend.app.schemas.envelope import envelope
from backend.app.services.legacy_adapter import legacy_adapter
from backend.app.utils.time import utc_now_iso


def _empty_blocks() -> dict[str, Any]:
    return {
        "configuration": {"status": "not_available", "data": None},
        "budget": {"status": "not_available", "data": None},
        "invested_value": {"status": "not_available", "data": None},
        "current_value": {"status": "not_available", "data": None},
        "result_absolute": {"status": "not_available", "data": None},
        "total_return": {"status": "not_available", "data": None},
        "twr": {"status": "not_available", "data": None},
        "mwr": {"status": "not_available", "data": None},
        "benchmark": {"status": "not_available", "data": None},
        "drawdown": {"status": "not_available", "data": None},
        "volatility": {"status": "not_available", "data": None},
        "composition": {"status": "not_available", "data": []},
        "currency_exposure": {"status": "not_available", "data": []},
        "contributors": {"status": "not_available", "data": []},
        "detractors": {"status": "not_available", "data": []},
        "projection": {"status": "not_available", "data": None},
        "rebalance": {"status": "not_available", "data": None},
        "valuation": {"status": "not_available", "data": None},
    }


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _last_price(metrics: dict[str, Any]) -> float | None:
    current = _number(metrics.get("last_px"))
    if current is not None:
        return current
    previous = _number(metrics.get("prev_px"))
    change = _number(metrics.get("last_chg"))
    if previous is not None and change is not None:
        return previous * (1.0 + change / 100.0)
    return previous


def _native_currency(item: dict[str, Any]) -> str:
    configured = item.get("moneda_nativa")
    if configured:
        return str(configured).upper()
    symbol = str(item.get("simbolo") or "").upper()
    kind = str(item.get("tipo") or "").lower()
    return "USD" if symbol.endswith("-USD") or kind == "cripto" else "ARS"


def _to_base(value: float | None, native: str, base: str, exchange_rate: float | None) -> float | None:
    if value is None:
        return None
    if native == base:
        return value
    if exchange_rate is None or exchange_rate <= 0:
        return None
    if native == "USD" and base == "ARS":
        return value * exchange_rate
    if native == "ARS" and base == "USD":
        return value / exchange_rate
    return None


def _basic_mwr(
    cash_flows: list[tuple[int, float]],
    end_timestamp: int,
    ending_value: float,
) -> float | None:
    flows = [(timestamp, amount) for timestamp, amount in cash_flows if amount > 0]
    if not flows or ending_value < 0:
        return None

    def npv(rate: float) -> float:
        if rate <= -0.999999:
            return float("inf")
        result = ending_value
        for timestamp, amount in flows:
            years = max(0.0, (end_timestamp - timestamp) / (365.25 * 24 * 3600))
            result -= amount / ((1.0 + rate) ** years)
        return result

    lower, upper = -0.999, 10.0
    lower_value, upper_value = npv(lower), npv(upper)
    if math.isfinite(lower_value) and math.isfinite(upper_value) and lower_value * upper_value < 0:
        for _ in range(80):
            middle = (lower + upper) / 2.0
            middle_value = npv(middle)
            if not math.isfinite(middle_value):
                break
            if abs(middle_value) < 1e-9:
                return middle
            if lower_value * middle_value <= 0:
                upper = middle
            else:
                lower = middle
                lower_value = middle_value
        return (lower + upper) / 2.0

    invested = sum(amount for _, amount in flows)
    return ending_value / invested - 1.0 if invested > 0 else None


def _benchmark_details(
    benchmark: Any,
    base_currency: str,
    metrics: dict[str, Any],
) -> dict[str, Any] | None:
    normalized = str(benchmark or "").lower()
    if normalized == "merval":
        label, symbol = "Merval", "^MERV"
    elif normalized in {"usd", "dolar", "dólar"}:
        label = "Dólar"
        symbol = "USDARS=X" if base_currency == "ARS" else "ARSUSD=X"
    else:
        return None
    values = metrics.get(symbol)
    if not isinstance(values, dict):
        return {"label": label, "symbol": symbol, "last_value": None, "daily_change_pct": None}
    return {
        "label": label,
        "symbol": symbol,
        "last_value": _last_price(values),
        "daily_change_pct": _number(values.get("last_chg")),
    }


async def get_portfolio_summary(user_id: str) -> dict[str, Any]:
    result = await legacy_adapter.fetch_portfolio_snapshot(user_id)
    blocks = _empty_blocks()
    if result.status == "not_available" or not result.data:
        return envelope(
            status="not_available",
            timestamp=utc_now_iso(),
            source=result.source,
            freshness="unknown",
            data=blocks,
            warnings=result.warnings or ["Portfolio legacy no disponible"],
        )

    portfolio = result.data.get("portfolio")
    metrics_by_symbol = result.data.get("metrics")
    if not isinstance(portfolio, dict):
        portfolio = {}
    if not isinstance(metrics_by_symbol, dict):
        metrics_by_symbol = {}
    items = portfolio.get("items")
    if not isinstance(items, list):
        items = []
    base_config = portfolio.get("base")
    if not isinstance(base_config, dict):
        base_config = {}
    base_currency = str(base_config.get("moneda") or "ARS").upper()
    exchange_rate = _number(base_config.get("tc_valor"))
    exchange_rate_timestamp = _number(base_config.get("tc_timestamp"))
    blocks["configuration"] = {
        "status": "ok",
        "data": {
            "base_currency": base_currency,
            "fx_reference": base_config.get("tc"),
            "fx_value": exchange_rate,
            "fx_timestamp": exchange_rate_timestamp,
            "benchmark": base_config.get("benchmark"),
        },
    }
    budget = _number(portfolio.get("monto"))
    if budget is not None:
        blocks["budget"] = {
            "status": "ok",
            "data": {"value": budget, "currency": base_currency},
        }

    total_invested = 0.0
    valued_invested = 0.0
    total_current = 0.0
    valued_items: list[dict[str, Any]] = []
    currency_totals: dict[str, float] = {}
    weighted_volatility = 0.0
    volatility_weight = 0.0
    unvalued = 0

    for raw_item in items:
        if not isinstance(raw_item, dict):
            unvalued += 1
            continue
        symbol = str(raw_item.get("simbolo") or "")
        invested = _number(raw_item.get("importe")) or 0.0
        total_invested += invested
        quantity = _number(raw_item.get("cantidad"))
        if quantity is None:
            purchase_price = _number(raw_item.get("precio_compra_base"))
            if purchase_price is not None and purchase_price > 0 and invested > 0:
                quantity = invested / purchase_price
        metrics = metrics_by_symbol.get(symbol)
        if not isinstance(metrics, dict):
            metrics = {}
        native_currency = _native_currency(raw_item)
        native_price = _last_price(metrics)
        base_price = _to_base(native_price, native_currency, base_currency, exchange_rate)
        if quantity is None or base_price is None:
            unvalued += 1
            continue

        current_value = quantity * base_price
        total_current += current_value
        valued_invested += invested
        currency_totals[native_currency] = currency_totals.get(native_currency, 0.0) + current_value
        result_value = current_value - invested
        result_pct = result_value / invested * 100.0 if invested > 0 else None
        volatility = _number(metrics.get("vol_ann"))
        added_timestamp = _number(raw_item.get("added_ts"))
        projections = legacy_adapter.calculate_projection_centers(metrics)
        valued_items.append(
            {
                "symbol": symbol,
                "type": raw_item.get("tipo"),
                "currency": native_currency,
                "quantity": quantity,
                "invested_value": invested,
                "current_value": current_value,
                "result": result_value,
                "result_pct": result_pct,
                "weight": 0.0,
                "price_native": native_price,
                "price_base": base_price,
                "daily_change_pct": _number(metrics.get("last_chg")),
                "volatility_pct": volatility,
                "added_timestamp": int(added_timestamp) if added_timestamp is not None else None,
                "valuation_mode": "observed_price",
                "projection": projections,
            }
        )

    for item in valued_items:
        item["weight"] = item["current_value"] / total_current if total_current > 0 else 0.0
        if item["volatility_pct"] is not None:
            weighted_volatility += item["weight"] * item["volatility_pct"]
            volatility_weight += item["weight"]

    blocks["invested_value"] = {
        "status": "ok",
        "data": {"value": total_invested, "currency": base_currency},
    }
    if valued_items:
        coverage = len(valued_items) / len(items) * 100.0 if items else 100.0
        current_status = "ok" if not unvalued else "partial"
        blocks["current_value"] = {
            "status": current_status,
            "data": {
                "value": total_current,
                "currency": base_currency,
                "coverage_pct": round(coverage, 2),
                "valued_items": len(valued_items),
                "total_items": len(items),
            },
        }
        blocks["composition"] = {"status": current_status, "data": valued_items}
        blocks["currency_exposure"] = {
            "status": current_status,
            "data": [
                {
                    "currency": currency,
                    "value": value,
                    "weight": value / total_current if total_current > 0 else 0.0,
                }
                for currency, value in sorted(currency_totals.items())
            ],
        }
        if valued_invested > 0:
            return_pct = (total_current / valued_invested - 1.0) * 100.0
            result_absolute = total_current - valued_invested
            blocks["result_absolute"] = {
                "status": current_status,
                "data": {"value": result_absolute, "currency": base_currency},
            }
            blocks["total_return"] = {
                "status": current_status,
                "data": {
                    "value_pct": return_pct,
                    "basis": "valued_items",
                    "invested_value": valued_invested,
                },
            }
            blocks["twr"] = {
                "status": "partial",
                "data": {
                    "value_pct": return_pct,
                    "method": "current_snapshot_approximation",
                },
            }

        cash_flows = [
            (item["added_timestamp"], item["invested_value"])
            for item in valued_items
            if item["added_timestamp"] is not None and item["invested_value"] > 0
        ]
        end_timestamp_raw = _number(result.data.get("last_timestamp"))
        end_timestamp = int(end_timestamp_raw) if end_timestamp_raw is not None else int(time())
        mwr = _basic_mwr(cash_flows, end_timestamp, total_current)
        if mwr is not None:
            blocks["mwr"] = {
                "status": "partial",
                "data": {
                    "value_pct": mwr * 100.0,
                    "method": "cash_flow_snapshot_approximation",
                    "cash_flow_count": len(cash_flows),
                },
            }

        if volatility_weight > 0:
            blocks["volatility"] = {
                "status": "partial",
                "data": {
                    "value_pct": weighted_volatility / volatility_weight,
                    "method": "weighted_component_volatility",
                },
            }

        ranked = sorted(valued_items, key=lambda item: item["result"], reverse=True)
        blocks["contributors"] = {"status": current_status, "data": ranked[:5]}
        blocks["detractors"] = {"status": current_status, "data": list(reversed(ranked[-5:]))}
        projection_rows = [
            item for item in valued_items if item["projection"].get("3m") is not None
        ]
        if projection_rows:
            weight_3m = sum(item["weight"] for item in projection_rows)
            weighted_3m = sum(
                item["weight"] * float(item["projection"]["3m"])
                for item in projection_rows
            ) / weight_3m if weight_3m > 0 else None
            projection_rows_6m = [
                item for item in projection_rows if item["projection"].get("6m") is not None
            ]
            weight_6m = sum(item["weight"] for item in projection_rows_6m)
            weighted_6m = sum(
                item["weight"] * float(item["projection"]["6m"])
                for item in projection_rows_6m
            ) / weight_6m if weight_6m > 0 else None
            blocks["projection"] = {
                "status": current_status,
                "data": {"3m": weighted_3m, "6m": weighted_6m, "method": "current_value_weighted"},
            }

        benchmark = _benchmark_details(
            base_config.get("benchmark"),
            base_currency,
            metrics_by_symbol,
        )
        if benchmark:
            benchmark_available = benchmark["daily_change_pct"] is not None
            blocks["benchmark"] = {
                "status": "partial" if benchmark_available else "not_available",
                "data": benchmark,
            }

        blocks["valuation"] = {
            "status": current_status,
            "data": {
                "as_of": result.data.get("last_timestamp"),
                "coverage_pct": round(coverage, 2),
                "valued_items": len(valued_items),
                "total_items": len(items),
                "method": "observed_prices_read_only",
            },
        }

    warnings = list(result.warnings)
    warnings.append("Portfolio legacy adaptado en modo lectura; PF no fue migrado ni modificado.")
    warnings.append("TWR/MWR canónicos, drawdown y rebalanceo requieren historial validado.")
    if blocks["twr"]["status"] == "partial" or blocks["mwr"]["status"] == "partial":
        warnings.append("TWR/MWR son aproximaciones de la foto actual y se identifican como parciales.")
    if unvalued:
        warnings.append(f"{unvalued} instrumentos sin valuacion real completa.")

    return envelope(
        status="partial",
        timestamp=utc_now_iso(),
        source=result.source,
        freshness=result.freshness,
        data=blocks,
        warnings=warnings,
    )


def get_unavailable_portfolio_summary() -> dict[str, Any]:
    return envelope(
        status="not_available",
        timestamp=utc_now_iso(),
        source="portfolio_service",
        freshness="unknown",
        data=_empty_blocks(),
        warnings=["PF legacy no se migra en esta fase; adaptacion segura pendiente"],
    )
