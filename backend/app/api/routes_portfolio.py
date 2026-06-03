"""Portfolio routes."""

from __future__ import annotations

from fastapi import APIRouter

from backend.app.services.portfolio_service import get_portfolio_summary


router = APIRouter(prefix="/portfolio", tags=["portfolio"])


@router.get("/summary")
def portfolio_summary() -> dict:
    return get_portfolio_summary()

