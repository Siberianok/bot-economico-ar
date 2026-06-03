"""Money formatting helpers."""

from __future__ import annotations


def format_number(value: float | int | None, decimals: int = 2) -> str:
    if value is None:
        return "-"
    formatted = f"{float(value):,.{decimals}f}"
    return formatted.replace(",", "X").replace(".", ",").replace("X", ".")


def format_money(value: float | int | None, currency: str = "ARS", decimals: int = 2) -> str:
    if value is None:
        return "$ -" if currency.upper() == "ARS" else "US$ -"
    prefix = "$" if currency.upper() == "ARS" else "US$"
    return f"{prefix} {format_number(value, decimals)}"

