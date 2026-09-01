import copy
from datetime import datetime, timezone
from types import SimpleNamespace

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.database import Base
from backend.app.services import alert_service
from backend.app.services.alert_service import get_alert_center, serialize_legacy_alert
from backend.app.services.legacy_adapter import LegacyAdapter, LegacyResult


def _db():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine, future=True)()


def test_alert_state_inspection_is_scoped_and_deep_copied():
    rules = [
        {
            "kind": "ticker",
            "symbol": "GGAL.BA",
            "op": ">",
            "value": 2500.0,
            "armed": True,
        }
    ]
    subscriptions = {123: {"daily": "09:00"}}
    original_rules = copy.deepcopy(rules)
    adapter = LegacyAdapter()
    adapter._module = SimpleNamespace(
        ALERTS={123: rules},
        SUBS=subscriptions,
        ALERTS_PAUSED={123},
        ALERTS_SILENT_UNTIL={},
        ALERT_REARM_COOLDOWN_SECS=900,
        ALERT_COMPOSITE_COOLDOWN_DEFAULT_SECS=1800,
        ALERT_PRICE_TOLERANCE_PCT=0.002,
    )

    found = adapter.inspect_alert_state("123")
    missing = adapter.inspect_alert_state("999")
    found.data["rules"][0]["value"] = 1.0
    found.data["subscription"]["daily"] = None

    assert found.status == "ok"
    assert found.data["global_pause"] is True
    assert missing.status == "ok"
    assert missing.data["rules"] == []
    assert rules == original_rules
    assert subscriptions[123]["daily"] == "09:00"


def test_legacy_alert_serialization_does_not_clean_expired_pause():
    expired = 1_700_000_000
    rule = {
        "kind": "crypto",
        "symbol": "BTCUSDT",
        "quote": "USDT",
        "op": ">",
        "value": 100_000,
        "pause_until": expired,
        "armed": False,
        "last_trigger_ts": 1_700_000_100,
        "last_trigger_price": 99_900,
    }
    original = copy.deepcopy(rule)

    serialized = serialize_legacy_alert(
        rule,
        0,
        {"defaults": {"rearm_cooldown_seconds": 900}},
        now=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    assert serialized["status"] == "waiting_rearm"
    assert serialized["asset"] == "BTCUSDT"
    assert serialized["cooldown_seconds"] == 900
    assert rule == original


def test_alert_center_keeps_operational_rules_separate_from_drafts(monkeypatch):
    state = {
        "rules": [
            {
                "kind": "fx",
                "type": "mep",
                "side": "venta",
                "op": ">",
                "value": 1600,
                "armed": True,
            },
            {
                "kind": "metric",
                "type": "riesgo",
                "op": "<",
                "value": 500,
                "pause_indef": True,
            },
        ],
        "subscription": {"daily": "08:00"},
        "global_pause": False,
        "global_pause_until": None,
        "defaults": {"rearm_cooldown_seconds": 900, "price_tolerance_pct": 0.002},
    }
    monkeypatch.setattr(
        alert_service.legacy_adapter,
        "inspect_alert_state",
        lambda _user_id: LegacyResult(
            status="ok",
            data=state,
            warnings=[],
            source="legacy.ALERTS_SUBS.read_only",
            freshness="current",
        ),
    )

    payload = get_alert_center(_db(), "123")

    assert payload["status"] == "ok"
    assert payload["data"]["summary"] == {
        "legacy_count": 2,
        "active_count": 1,
        "paused_count": 1,
        "draft_count": 0,
        "subscription_active": True,
    }
    assert payload["data"]["legacy"]["read_only"] is True
    assert payload["data"]["subscription"]["daily_time"] == "08:00"
    assert payload["data"]["items"] == payload["data"]["drafts"]["items"]


def test_alert_center_marks_legacy_unavailable_without_faking_rules(monkeypatch):
    monkeypatch.setattr(
        alert_service.legacy_adapter,
        "inspect_alert_state",
        lambda _user_id: LegacyResult(
            status="not_available",
            data=None,
            warnings=["Identidad Telegram requerida"],
        ),
    )

    payload = get_alert_center(_db(), "local-dev")

    assert payload["status"] == "partial"
    assert payload["data"]["legacy"]["items"] == []
    assert payload["data"]["subscription"]["enabled"] is False
    assert payload["warnings"] == ["Identidad Telegram requerida"]
