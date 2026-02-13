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


class DummyQuery:
    def __init__(self, data, chat_id=10):
        self.data = data
        self.message = SimpleNamespace(message_id=1, chat_id=chat_id)
        self.edits = []
        self._chat_id = chat_id

    async def answer(self):
        return None

    async def edit_message_text(self, text, **kwargs):
        self.edits.append((text, kwargs))


class DummyMessage:
    def __init__(self, text):
        self.text = text
        self.replies = []

    async def reply_text(self, text, **kwargs):
        self.replies.append((text, kwargs))


@pytest.fixture(autouse=True)
def _reset_pf(monkeypatch):
    bot.PF.clear()

    async def _fake_save_state():
        return None

    async def _fake_market_snapshot(_pf):
        return ([], 0.0, 0.0, 0.0, None, None, None)

    monkeypatch.setattr(bot, "save_state", _fake_save_state)
    monkeypatch.setattr(bot, "pf_market_snapshot", _fake_market_snapshot)


def _make_update_with_query(data, chat_id=10):
    q = DummyQuery(data=data, chat_id=chat_id)
    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=chat_id),
        callback_query=q,
    )
    return update, q


def test_kb_pf_main_budget_button_label():
    kb = bot.kb_pf_main()
    assert kb.inline_keyboard[1][1].text == "Presupuesto"


def test_parse_budget_value_accepts_presets_and_manual_text():
    assert bot.parse_budget_value("100.000") == pytest.approx(100000.0)
    assert bot.parse_budget_value("1,5") == pytest.approx(1.5)
    assert bot.parse_budget_value("abc") is None


def test_pf_setmonto_opens_budget_screen():
    update, q = _make_update_with_query("PF:SETMONTO")
    context = SimpleNamespace(user_data={})

    asyncio.run(bot.pf_menu_cb(update, context))

    assert q.edits
    text, kwargs = q.edits[-1]
    assert "PF:BUDGET" in text
    kb = kwargs["reply_markup"].inline_keyboard
    assert kb[0][0].text.endswith("ARS")
    assert kb[-2][0].text == "Volver"


def test_kb_pf_budget_usd_contains_expected_presets_and_manual_option():
    kb = bot.kb_pf_budget("USD").inline_keyboard

    usd_preset_labels = [row[0].text for row in kb[1:8]]
    assert usd_preset_labels == [
        "1.000",
        "10.000",
        "100.000",
        "1.000.000",
        "10.000.000",
        "50.000.000",
        "100.000.000",
    ]
    assert kb[8][0].text == "Ingresar manual"


def test_pf_budget_preset_updates_amount_and_returns_main(monkeypatch):
    async def _fake_main_menu_text(_chat_id):
        return "menu actualizado"

    monkeypatch.setattr(bot, "pf_main_menu_text", _fake_main_menu_text)
    chat_id = 22
    bot.pf_get(chat_id)

    update, q = _make_update_with_query("PF:BUDGET:PRESET:ARS:100.000", chat_id=chat_id)
    context = SimpleNamespace(user_data={})

    asyncio.run(bot.pf_menu_cb(update, context))

    assert bot.PF[chat_id]["monto"] == pytest.approx(100000.0)
    assert q.edits[-1][0] == "menu actualizado"


def test_pf_budget_manual_mode_uses_numeric_parser(monkeypatch):
    calls = []

    async def _fake_refresh_menu(_context, _chat_id, *, force_new=False):
        calls.append(force_new)

    monkeypatch.setattr(bot, "pf_refresh_menu", _fake_refresh_menu)

    chat_id = 35
    bot.pf_get(chat_id)
    context = SimpleNamespace(user_data={"pf_mode": "set_monto_manual"})
    message = DummyMessage("500.000")
    update = SimpleNamespace(effective_chat=SimpleNamespace(id=chat_id), message=message)

    asyncio.run(bot.pf_text_input(update, context))

    assert bot.PF[chat_id]["monto"] == pytest.approx(500000.0)
    assert calls == [True]
    assert context.user_data["pf_mode"] is None
