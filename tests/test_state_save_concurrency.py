import asyncio
import copy
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
os.environ.setdefault("TELEGRAM_TOKEN", "test-token")

import bot_econ_full_plus_rank_alerts as bot


class _FakeStore:
    def __init__(self, delay: float = 0.0):
        self.delay = delay
        self.saved_payloads = []

    async def save(self, payload):
        if self.delay:
            await asyncio.sleep(self.delay)
        self.saved_payloads.append(copy.deepcopy(payload))
        return True


def test_save_state_serializes_parallel_calls(monkeypatch):
    async def _run():
        fake_store = _FakeStore(delay=0.02)
        monkeypatch.setattr(bot, "STATE_STORE", fake_store)
        monkeypatch.setattr(bot, "FALLBACK_STATE_STORE", None)
        monkeypatch.setattr(bot, "_prune_news_history", lambda: None)
        monkeypatch.setattr(bot, "_prune_proj_history", lambda: None)

        bot.ALERTS = {1: [{"kind": "metric", "value": 1}]}

        first = asyncio.create_task(bot.save_state())
        await asyncio.sleep(0.005)

        bot.ALERTS = {1: [{"kind": "metric", "value": 2}]}
        second = asyncio.create_task(bot.save_state())

        await asyncio.gather(first, second)

        first_alerts = fake_store.saved_payloads[0]["alerts"]
        second_alerts = fake_store.saved_payloads[1]["alerts"]
        assert len(fake_store.saved_payloads) == 2
        assert (first_alerts.get("1") or first_alerts.get(1))[0]["value"] == 1
        assert (second_alerts.get("1") or second_alerts.get(1))[0]["value"] == 2

    asyncio.run(_run())


def test_schedule_state_save_debounces_burst(monkeypatch):
    async def _run():
        call_count = 0

        async def _fake_save_state():
            nonlocal call_count
            call_count += 1

        monkeypatch.setattr(bot, "save_state", _fake_save_state)
        monkeypatch.setattr(bot, "STATE_SAVE_DEBOUNCE_SECONDS", 0.01)
        monkeypatch.setattr(bot, "_STATE_SAVE_TASK", None)
        monkeypatch.setattr(bot, "_STATE_SAVE_DIRTY", False)

        for _ in range(15):
            bot.schedule_state_save()

        await asyncio.sleep(0.06)

        assert call_count == 1
        assert bot._STATE_SAVE_TASK is None

    asyncio.run(_run())
