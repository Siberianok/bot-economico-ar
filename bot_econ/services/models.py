from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True)
class AlertRule:
    type: str
    symbol: str
    threshold: float
    direction: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    extra: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class PortfolioPosition:
    symbol: str
    quantity: float
    price: float | None = None


@dataclass(slots=True)
class Portfolio:
    chat_id: int
    positions: list[PortfolioPosition]
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(slots=True)
class Subscription:
    chat_id: int
    hhmm: str
    timezone: str
    paused: bool = False
    updated_at: datetime = field(default_factory=datetime.utcnow)
