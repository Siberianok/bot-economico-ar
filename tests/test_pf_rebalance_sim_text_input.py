import asyncio
import os
import pathlib
import sys
from types import SimpleNamespace

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
os.environ.setdefault("TELEGRAM_TOKEN", "test-token")

import bot_econ_full_plus_rank_alerts as bot


class DummyMessage:
    def __init__(self, text: str):
        self.text = text
        self.replies = []

    async def reply_text(self, text, **kwargs):
        self.replies.append((text, kwargs))


@pytest.fixture(autouse=True)
def _reset_pf():
    bot.PF.clear()


def _build_snapshot():
    return [
        {
            "label": "AL30",
            "symbol": "AL30",
            "peso": 0.5,
            "valor_actual": 500.0,
            "raw": {"objetivo_pct": 50.0},
        },
        {
            "label": "GGAL",
            "symbol": "GGAL",
            "peso": 0.5,
            "valor_actual": 500.0,
            "raw": {"objetivo_pct": 50.0},
        },
    ]


def test_pf_rebal_sim_one_with_valid_numeric_input(monkeypatch):
    async def _fake_market_snapshot(_pf):
        return _build_snapshot(), 0, 1000.0, 1000.0, None, None, None

    monkeypatch.setattr(bot, "pf_market_snapshot", _fake_market_snapshot)

    async def _fake_refresh_menu(_context, _chat_id, **_kwargs):
        return None

    monkeypatch.setattr(bot, "pf_refresh_menu", _fake_refresh_menu)

    chat_id = 101
    bot.pf_get(chat_id)
    message = DummyMessage("100")
    update = SimpleNamespace(effective_chat=SimpleNamespace(id=chat_id), message=message)
    context = SimpleNamespace(user_data={"pf_mode": "pf_rebal_sim_one"})

    asyncio.run(bot.pf_text_input(update, context))

    assert message.replies
    response_text, kwargs = message.replies[-1]
    assert "Simulación de aporte único" in response_text
    assert "Sugerencias por instrumento" in response_text
    assert kwargs.get("parse_mode") == bot.ParseMode.HTML


def test_pf_rebal_sim_month_with_valid_numeric_input(monkeypatch):
    async def _fake_market_snapshot(_pf):
        return _build_snapshot(), 0, 1000.0, 1000.0, None, None, None

    monkeypatch.setattr(bot, "pf_market_snapshot", _fake_market_snapshot)

    async def _fake_refresh_menu(_context, _chat_id, **_kwargs):
        return None

    monkeypatch.setattr(bot, "pf_refresh_menu", _fake_refresh_menu)

    chat_id = 202
    bot.pf_get(chat_id)
    message = DummyMessage("250")
    update = SimpleNamespace(effective_chat=SimpleNamespace(id=chat_id), message=message)
    context = SimpleNamespace(user_data={"pf_mode": "pf_rebal_sim_month"})

    asyncio.run(bot.pf_text_input(update, context))

    assert message.replies
    response_text, kwargs = message.replies[-1]
    assert "Simulación de aporte mensual" in response_text
    assert "Sugerencias por instrumento" in response_text
    assert kwargs.get("parse_mode") == bot.ParseMode.HTML
