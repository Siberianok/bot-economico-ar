from datetime import datetime, timedelta, timezone

from backend.app.models import AlertRule
from backend.app.services.alert_service import evaluate_alert


def test_exact_alert_triggers_on_threshold():
    alert = AlertRule(
        asset="dolar_cripto",
        condition_type="exact",
        operator=">=",
        target_value=1420,
        cooldown_minutes=15,
        status="active",
    )
    result = evaluate_alert(alert, 1421)
    assert result["triggered"] is True
    assert result["reason"] == "exact"


def test_approximation_alert_triggers_inside_tolerance():
    alert = AlertRule(
        asset="dolar_cripto",
        condition_type="approximation",
        operator=">=",
        target_value=1420,
        prealert_tolerance_pct=0.5,
        cooldown_minutes=15,
        status="active",
    )
    result = evaluate_alert(alert, 1414)
    assert result["triggered"] is True
    assert result["reason"] == "approximation"


def test_cooldown_blocks_repeated_trigger():
    now = datetime.now(timezone.utc)
    alert = AlertRule(
        asset="dolar_cripto",
        condition_type="exact",
        operator=">=",
        target_value=1420,
        cooldown_minutes=15,
        status="active",
        last_triggered_at=now - timedelta(minutes=5),
    )
    result = evaluate_alert(alert, 1500, now=now)
    assert result["triggered"] is False
    assert result["reason"] == "cooldown"

