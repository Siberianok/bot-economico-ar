from fastapi.testclient import TestClient

from backend.app.bot.telegram_bot import telegram_runtime
from backend.app.main import app


def test_api_import_keeps_telegram_legacy_lazy_by_default():
    assert telegram_runtime.state == "disabled"
    assert telegram_runtime.legacy is None
    assert telegram_runtime.application is None


def test_health_reports_disabled_telegram_runtime():
    with TestClient(app) as client:
        payload = client.get("/api/v1/health").json()
    assert payload["data"]["telegram"]["enabled"] is False
    assert payload["data"]["telegram"]["state"] == "disabled"


def test_telegram_webhook_rejects_unknown_secret_without_importing_legacy():
    with TestClient(app) as client:
        response = client.post(
            "/telegram/webhook",
            json={"update_id": 1},
            headers={"X-Telegram-Bot-Api-Secret-Token": "invalid"},
        )
    assert response.status_code == 404
    assert response.json()["status"] == "error"
    assert telegram_runtime.legacy is None
    assert telegram_runtime.application is None


def test_telegram_webhook_returns_unavailable_when_secret_is_valid_but_runtime_is_off(monkeypatch):
    from dataclasses import replace

    monkeypatch.setattr(
        telegram_runtime,
        "settings",
        replace(telegram_runtime.settings, webhook_secret="valid_secret"),
    )
    with TestClient(app) as client:
        response = client.post(
            "/telegram/webhook",
            json={"update_id": 1},
            headers={"X-Telegram-Bot-Api-Secret-Token": "valid_secret"},
        )
    assert response.status_code == 503
    assert response.json()["status"] == "error"
