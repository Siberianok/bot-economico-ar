"""Portfolio summary service."""

from __future__ import annotations

from typing import Any

from backend.app.schemas.envelope import envelope
from backend.app.utils.time import utc_now_iso


def get_portfolio_summary() -> dict[str, Any]:
    blocks: dict[str, Any] = {
        "invested_value": {"status": "not_available", "data": None},
        "current_value": {"status": "not_available", "data": None},
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
    }
    return envelope(
        status="not_available",
        timestamp=utc_now_iso(),
        source="portfolio_service",
        freshness="unknown",
        data=blocks,
        warnings=["PF legacy no se migra en esta fase; adaptacion segura pendiente"],
    )

