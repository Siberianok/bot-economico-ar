from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True)
class Quote:
    name: str
    buy: float | None
    sell: float | None
    last_update: datetime | None = None


@dataclass(slots=True)
class MarketMetric:
    symbol: str
    price: float | None
    change_pct: float | None
    currency: str
    timestamp: datetime | None = None


@dataclass(slots=True)
class ReserveStatus:
    total: float | None
    variation: float | None
    date: datetime | None
