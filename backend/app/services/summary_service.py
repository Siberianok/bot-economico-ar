"""Daily overview composed from real market and news services."""

from __future__ import annotations

import asyncio
from typing import Any

from backend.app.schemas.envelope import envelope
from backend.app.services.market_service import get_market_pulse
from backend.app.services.news_service import get_news
from backend.app.utils.time import utc_now_iso


async def get_daily_summary() -> dict[str, Any]:
    market, news = await asyncio.gather(get_market_pulse(), get_news(limit=4))
    statuses = {market.get("status"), news.get("status")}
    if statuses == {"not_available"}:
        status = "not_available"
    elif statuses <= {"ok"}:
        status = "ok"
    else:
        status = "partial"
    freshness = (
        "current"
        if market.get("freshness") == "current" or news.get("freshness") == "current"
        else "unknown"
    )
    return envelope(
        status=status,
        timestamp=utc_now_iso(),
        source="summary_service",
        freshness=freshness,
        data={
            "market": market.get("data"),
            "news": news.get("data"),
        },
        warnings=[*(market.get("warnings") or []), *(news.get("warnings") or [])],
    )
