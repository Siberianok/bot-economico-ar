from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.legacy_adapter import LegacyResult
from backend.app.utils.telegram_auth import TelegramIdentity, get_telegram_identity


def test_alerts_endpoint_exposes_legacy_and_subscription_read_only(monkeypatch):
    monkeypatch.setattr(
        "backend.app.services.alert_service.legacy_adapter.inspect_alert_state",
        lambda user_id: LegacyResult(
            status="ok",
            data={
                "rules": [
                    {
                        "kind": "ticker",
                        "symbol": "GGAL.BA",
                        "op": ">",
                        "value": 2500,
                        "armed": True,
                    }
                ],
                "subscription": {"daily": "10:00"},
                "global_pause": False,
                "global_pause_until": None,
                "defaults": {"rearm_cooldown_seconds": 900},
            },
            warnings=[],
            source="legacy.ALERTS_SUBS.read_only",
            freshness="current",
        ),
    )
    app.dependency_overrides[get_telegram_identity] = lambda: TelegramIdentity(
        user_id="778899",
        username="ariel",
    )
    try:
        with TestClient(app) as client:
            response = client.get("/api/v1/alerts")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["data"]["summary"]["legacy_count"] == 1
    assert payload["data"]["legacy"]["items"][0]["asset"] == "GGAL.BA"
    assert payload["data"]["legacy"]["read_only"] is True
    assert payload["data"]["subscription"]["enabled"] is True
    assert payload["data"]["subscription"]["manage_command"] == "/subs"
