import json

import pytest
from bot_econ.config import AppConfig
from bot_econ.services.state_service import StateService


@pytest.mark.asyncio
async def test_migrate_from_json(monkeypatch, tmp_path):
    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps(
            {
                "alerts": {"1": [{"symbol": "GGAL", "threshold": 100}]},
                "subs": {"1": {"hhmm": "13:00", "tz": "America/Argentina/Buenos_Aires"}},
                "pf": {"1": {"positions": []}},
            }
        ),
        encoding="utf-8",
    )

    calls = {"alerts": 0, "subs": 0, "pf": 0}

    async def fake_alerts_add(chat_id, rule):
        calls["alerts"] += 1

    async def fake_subs_set(chat_id, hhmm, tz, paused):
        calls["subs"] += 1

    async def fake_pf_set(chat_id, payload):
        calls["pf"] += 1

    target = "bot_econ.services.state_service.storage_adapter"
    monkeypatch.setattr(f"{target}.alerts_add", fake_alerts_add)
    monkeypatch.setattr(f"{target}.subs_set", fake_subs_set)
    monkeypatch.setattr(f"{target}.pf_set", fake_pf_set)

    config = AppConfig(
        telegram_token="dummy",
        webhook_secret="secret",
        port=8080,
        base_url="http://localhost",
        redis_url="redis://localhost",
        redis_prefix="test",
        state_path=str(state_file),
    )
    service = StateService(config)
    await service.migrate_from_json()

    assert calls == {"alerts": 1, "subs": 1, "pf": 1}
