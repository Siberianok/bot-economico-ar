import hashlib
import hmac
import json
from time import time
from urllib.parse import urlencode

from fastapi.testclient import TestClient

from backend.app.config import get_settings
from backend.app.main import app
from backend.app.services.legacy_adapter import LegacyResult
from backend.app.utils.telegram_auth import TelegramIdentity, get_telegram_identity


def _signed_header(token: str, user_id: int) -> str:
    fields = {
        "auth_date": str(int(time())),
        "query_id": "AAE-session-test",
        "user": json.dumps(
            {"id": user_id, "first_name": "Ariel", "username": "ariel"},
            separators=(",", ":"),
        ),
    }
    data_check_string = "\n".join(f"{key}={fields[key]}" for key in sorted(fields))
    secret = hmac.new(b"WebAppData", token.encode(), hashlib.sha256).digest()
    fields["hash"] = hmac.new(
        secret,
        data_check_string.encode(),
        hashlib.sha256,
    ).hexdigest()
    return urlencode(fields)


def test_session_reports_server_validated_telegram_identity(monkeypatch):
    monkeypatch.setattr(
        "backend.app.services.session_service.legacy_adapter.inspect_portfolio_state",
        lambda user_id: LegacyResult(
            status="ok",
            data={"portfolio": {}, "item_count": 3},
            warnings=[],
            source="legacy.PF.read_only",
            freshness="current",
        ),
    )
    monkeypatch.setattr(
        "backend.app.services.session_service.legacy_adapter.inspect_alert_state",
        lambda user_id: LegacyResult(
            status="ok",
            data={"rules": [{"kind": "ticker"}], "subscription": {"daily": "09:00"}},
            warnings=[],
            source="legacy.ALERTS_SUBS.read_only",
            freshness="current",
        ),
    )
    app.dependency_overrides[get_telegram_identity] = lambda: TelegramIdentity(
        user_id="1234",
        username="ariel",
        first_name="Ariel",
    )
    try:
        with TestClient(app) as client:
            response = client.get("/api/v1/session")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["data"]["authentication"] == {
        "mode": "telegram",
        "validated": True,
        "validation_method": "telegram_init_data",
    }
    assert payload["data"]["user"]["id"] == "1234"
    assert payload["data"]["portfolio"]["item_count"] == 3
    assert payload["data"]["alerts"]["rule_count"] == 1
    assert payload["data"]["subscription"]["daily_time"] == "09:00"


def test_session_marks_development_fallback_as_unverified():
    app.dependency_overrides[get_telegram_identity] = lambda: TelegramIdentity(
        user_id="local-dev",
        username="local-dev",
        is_development=True,
    )
    try:
        with TestClient(app) as client:
            response = client.get("/api/v1/session")
    finally:
        app.dependency_overrides.clear()

    payload = response.json()
    assert payload["status"] == "partial"
    assert payload["data"]["authentication"]["validated"] is False
    assert payload["data"]["portfolio"]["status"] == "not_available"


def test_signed_header_drives_portfolio_user_id(monkeypatch):
    captured: list[str] = []

    async def fake_summary(user_id: str):
        captured.append(user_id)
        return {
            "status": "not_available",
            "timestamp": "2026-08-31T00:00:00Z",
            "source": "test",
            "freshness": "unknown",
            "data": None,
            "warnings": [],
        }

    monkeypatch.setenv("TELEGRAM_AUTH_REQUIRED", "1")
    monkeypatch.setenv("TELEGRAM_TOKEN", "signed-test-token")
    monkeypatch.setattr(
        "backend.app.api.routes_portfolio.get_portfolio_summary",
        fake_summary,
    )
    get_settings.cache_clear()
    try:
        with TestClient(app) as client:
            response = client.get(
                "/api/v1/portfolio/summary",
                headers={
                    "X-Telegram-Init-Data": _signed_header("signed-test-token", 778899),
                },
            )
    finally:
        get_settings.cache_clear()

    assert response.status_code == 200
    assert captured == ["778899"]
