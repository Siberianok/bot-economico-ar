"""Transparent signal rules shared by Market Pulse and the signal engine."""

from __future__ import annotations

from typing import Any


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def derive_market_signal_candidates(pulse: dict[str, Any]) -> list[dict[str, Any]]:
    data = pulse.get("data") if isinstance(pulse, dict) else None
    if not isinstance(data, dict):
        return []

    candidates: list[dict[str, Any]] = []
    gaps_block = data.get("brechas")
    gaps = gaps_block.get("data") if isinstance(gaps_block, dict) else None
    if isinstance(gaps, dict):
        for market, threshold in (("ccl", 3.0), ("blue", 5.0)):
            gap = _number(gaps.get(f"{market}_pct"))
            if gap is None or abs(gap) < threshold:
                continue
            magnitude = abs(gap)
            severity = "warning" if magnitude >= 10.0 else "info"
            candidates.append(
                {
                    "dedup_key": f"market:fx-gap:{market}",
                    "title": f"Brecha {market.upper()} {'amplia' if magnitude >= 10.0 else 'moderada'}",
                    "description": (
                        f"El {market.upper()} se ubica {gap:.2f}% respecto del dolar oficial. "
                        "Lectura automatica basada en cotizaciones observadas."
                    ),
                    "score": min(100, round(50 + magnitude * 5)),
                    "severity": severity,
                    "type": "fx_gap",
                    "source": pulse.get("source", "legacy_adapter"),
                    "cooldown_minutes": 360,
                    "asset": market.upper(),
                    "metric": "gap_vs_official_pct",
                    "current_value": gap,
                    "variation": gap,
                }
            )

    for block_name, asset in (
        ("dolar_blue", "BLUE"),
        ("dolar_mep", "MEP"),
        ("dolar_ccl", "CCL"),
        ("dolar_cripto", "CRIPTO"),
    ):
        block = data.get(block_name)
        quote = block.get("data") if isinstance(block, dict) else None
        variation = _number(quote.get("variation")) if isinstance(quote, dict) else None
        if variation is None or abs(variation) < 2.0:
            continue
        candidates.append(
            {
                "dedup_key": f"market:fx-daily-move:{asset.lower()}",
                "title": f"Movimiento diario relevante en {asset}",
                "description": f"La variacion diaria observada es {variation:.2f}%.",
                "score": min(100, round(60 + abs(variation) * 6)),
                "severity": "warning" if abs(variation) >= 4.0 else "info",
                "type": "fx_move",
                "source": pulse.get("source", "legacy_adapter"),
                "cooldown_minutes": 360,
                "asset": asset,
                "metric": "daily_variation_pct",
                "current_value": variation,
                "variation": variation,
            }
        )

    risk_block = data.get("riesgo_pais")
    risk_data = risk_block.get("data") if isinstance(risk_block, dict) else None
    risk_value = _number(risk_data.get("value")) if isinstance(risk_data, dict) else None
    if risk_value is not None and risk_value >= 600:
        candidates.append(
            {
                "dedup_key": "market:country-risk:elevated",
                "title": "Riesgo pais elevado",
                "description": f"El ultimo valor observado es {risk_value:.0f} puntos basicos.",
                "score": min(100, round(55 + (risk_value - 600) / 15)),
                "severity": "warning" if risk_value >= 800 else "info",
                "type": "macro_risk",
                "source": pulse.get("source", "legacy_adapter"),
                "cooldown_minutes": 720,
                "asset": "ARGENTINA",
                "metric": "country_risk_bps",
                "current_value": risk_value,
            }
        )

    inflation_block = data.get("inflacion")
    inflation_data = inflation_block.get("data") if isinstance(inflation_block, dict) else None
    inflation = _number(inflation_data.get("value")) if isinstance(inflation_data, dict) else None
    if inflation is not None and inflation >= 4.0:
        candidates.append(
            {
                "dedup_key": "market:inflation:monthly-high",
                "title": "Inflacion mensual elevada",
                "description": f"El ultimo dato mensual observado es {inflation:.2f}%.",
                "score": min(100, round(55 + inflation * 5)),
                "severity": "warning" if inflation >= 6.0 else "info",
                "type": "macro_inflation",
                "source": pulse.get("source", "legacy_adapter"),
                "cooldown_minutes": 24 * 60,
                "asset": "ARGENTINA",
                "metric": "monthly_inflation_pct",
                "current_value": inflation,
            }
        )
    return candidates
