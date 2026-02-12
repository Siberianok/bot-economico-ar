import asyncio
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
os.environ.setdefault("TELEGRAM_TOKEN", "test-token")

import bot_econ_full_plus_rank_alerts as bot


class _DummyStore:
    def __init__(self, payload):
        self.payload = payload

    async def load(self):
        return self.payload


def test_load_state_assigns_projection_globals(monkeypatch):
    monkeypatch.setattr(bot, "STATE_STORE", _DummyStore({
        "alerts": {},
        "subs": {},
        "pf": {},
        "pf_history": {},
        "alert_usage": {},
        "projection_records": [{"symbol": "GGAL", "horizon": 63, "base_price": 100.0, "projection": 120.0, "created_at": 1700000000.0}],
        "projection_batches": [{"batch_id": "b-1", "horizon": 63, "created_at": 1700000000.0, "symbols": ["GGAL"], "predictions": {"GGAL": 120.0}, "base_prices": {"GGAL": 100.0}, "evaluated": True}],
    }))
    monkeypatch.setattr(bot, "FALLBACK_STATE_STORE", None)

    bot.PROJECTION_RECORDS = []
    bot.PROJECTION_BATCHES = []

    asyncio.run(bot.load_state())

    assert bot.PROJECTION_RECORDS == [{"symbol": "GGAL", "horizon": 63, "base_price": 100.0, "projection": 120.0, "created_at": 1700000000.0}]
    assert bot.PROJECTION_BATCHES == [{"batch_id": "b-1", "horizon": 63, "created_at": 1700000000.0, "symbols": ["GGAL"], "predictions": {"GGAL": 120.0}, "base_prices": {"GGAL": 100.0}, "evaluated": True}]


def test_load_state_invalid_projection_payload_resets_to_empty_lists(monkeypatch):
    monkeypatch.setattr(bot, "STATE_STORE", _DummyStore({
        "alerts": {},
        "subs": {},
        "pf": {},
        "pf_history": {},
        "alert_usage": {},
        "projection_records": {"bad": "payload"},
        "projection_batches": "invalid",
    }))
    monkeypatch.setattr(bot, "FALLBACK_STATE_STORE", None)

    bot.PROJECTION_RECORDS = []
    bot.PROJECTION_BATCHES = []

    asyncio.run(bot.load_state())

    assert bot.PROJECTION_RECORDS == []
    assert bot.PROJECTION_BATCHES == []
