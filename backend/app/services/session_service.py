"""Authenticated Mini App session diagnostics."""

from __future__ import annotations

from typing import Any

from backend.app.bot.telegram_bot import telegram_runtime
from backend.app.schemas.envelope import envelope
from backend.app.services.legacy_adapter import legacy_adapter
from backend.app.utils.telegram_auth import TelegramIdentity
from backend.app.utils.time import utc_now_iso


def get_session_status(identity: TelegramIdentity) -> dict[str, Any]:
    portfolio = legacy_adapter.inspect_portfolio_state(identity.user_id)
    portfolio_data = portfolio.data if isinstance(portfolio.data, dict) else {}
    alert_state = legacy_adapter.inspect_alert_state(identity.user_id)
    alert_data = alert_state.data if isinstance(alert_state.data, dict) else {}
    alert_rules = alert_data.get("rules") if isinstance(alert_data.get("rules"), list) else []
    subscription = (
        alert_data.get("subscription")
        if isinstance(alert_data.get("subscription"), dict)
        else {}
    )
    authenticated = not identity.is_development
    warnings = list(dict.fromkeys([*portfolio.warnings, *alert_state.warnings]))
    if identity.is_development:
        warnings.insert(0, "Sesión de desarrollo: no representa una identidad Telegram firmada")

    return envelope(
        status="ok" if authenticated else "partial",
        timestamp=utc_now_iso(),
        source="telegram_session",
        freshness="current",
        data={
            "authentication": {
                "mode": "telegram" if authenticated else "development",
                "validated": authenticated,
                "validation_method": "telegram_init_data" if authenticated else "development_fallback",
            },
            "user": {
                "id": identity.user_id,
                "username": identity.username,
                "first_name": identity.first_name,
            },
            "portfolio": {
                "status": portfolio.status,
                "source": portfolio.source,
                "item_count": portfolio_data.get("item_count", 0),
            },
            "alerts": {
                "status": alert_state.status,
                "source": alert_state.source,
                "rule_count": len(alert_rules),
            },
            "subscription": {
                "status": alert_state.status,
                "source": alert_state.source,
                "daily_time": subscription.get("daily"),
                "enabled": bool(subscription.get("daily")),
            },
            "telegram_runtime": telegram_runtime.status(),
        },
        warnings=warnings,
    )
