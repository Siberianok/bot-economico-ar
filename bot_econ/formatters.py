from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from zoneinfo import ZoneInfo


@dataclass(slots=True)
class TimeFormatter:
    tz: ZoneInfo

    def humanize(self, ts: datetime, fmt: str = "%d/%m %H:%M") -> str:
        return ts.astimezone(self.tz).strftime(fmt)


def fmt_number(value: float | None, ndigits: int = 2) -> str:
    if value is None:
        return "—"
    try:
        formatted = f"{value:,.{ndigits}f}"
    except (TypeError, ValueError):
        return str(value)
    return formatted.replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_money_ars(value: float | None, ndigits: int = 2) -> str:
    if value is None:
        return "$ —"
    try:
        return f"$ {fmt_number(float(value), ndigits)}"
    except (TypeError, ValueError):
        return f"$ {value}"


def fmt_money_usd(value: float | None, ndigits: int = 2) -> str:
    if value is None:
        return "US$ —"
    try:
        return f"US$ {fmt_number(float(value), ndigits)}"
    except (TypeError, ValueError):
        return f"US$ {value}"


def fmt_pct(value: float | None, ndigits: int = 2) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):+.{ndigits}f}%".replace(".", ",")
    except (TypeError, ValueError):
        return str(value)
