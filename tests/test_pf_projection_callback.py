import asyncio

import bot_econ_full_plus_rank_alerts as bot


class _DummyMessage:
    chat_id = 123
    message_id = 456


class _DummyCallbackQuery:
    def __init__(self, data: str):
        self.data = data
        self.message = _DummyMessage()

    async def answer(self):
        return None


class _DummyUpdate:
    def __init__(self, data: str):
        self.callback_query = _DummyCallbackQuery(data)


class _DummyContext:
    def __init__(self):
        self.user_data = {}


def test_pf_proj_h_12_callback_routes_to_projection(monkeypatch):
    called = {}

    async def _fake_show_projection(context, chat_id, horizon):
        called["chat_id"] = chat_id
        called["horizon"] = horizon
        called["context"] = context

    monkeypatch.setattr(bot, "pf_show_projection_below", _fake_show_projection)

    update = _DummyUpdate("PF:PROJ:H:12")
    context = _DummyContext()

    asyncio.run(bot.pf_menu_cb(update, context))

    assert called == {"chat_id": 123, "horizon": 12, "context": context}
    assert context.user_data["pf_menu_msg_id"] == 456


def test_pf_show_projection_below_supports_12m_horizon(monkeypatch):
    sent_messages = []

    monkeypatch.setattr(
        bot,
        "pf_get",
        lambda _chat_id: {"items": [{"symbol": "GGAL"}], "base": {"moneda": "ARS", "tc": "mep"}},
    )

    async def _fake_market_snapshot(_pf):
        snapshot = [
            {
                "symbol": "GGAL",
                "label": "Grupo Galicia",
                "peso": 1.0,
                "invertido": 100000.0,
                "valor_actual": 110000.0,
                "added_ts": None,
                "metrics": {
                    "1m": 3.0,
                    "3m": 9.0,
                    "6m": 18.0,
                    "vol_ann": 25.0,
                    "slope50": 2.0,
                    "trend_flag": 1.0,
                    "hi52": -5.0,
                    "dd6m": 8.0,
                },
            }
        ]
        return snapshot, 1700000000.0, 100000.0, 110000.0, None, None, None

    monkeypatch.setattr(bot, "pf_market_snapshot", _fake_market_snapshot)

    async def _fake_send_below_menu(_context, _chat_id, text=None, **_kwargs):
        if text:
            sent_messages.append(text)

    monkeypatch.setattr(bot, "_send_below_menu", _fake_send_below_menu)
    monkeypatch.setattr(bot, "_projection_bar_image", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(bot, "_projection_by_instrument_single_image", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(bot, "register_projection_history", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(bot, "schedule_state_save", lambda: None)

    async def _fake_refresh_menu(_context, _chat_id, force_new=False):
        return None

    monkeypatch.setattr(bot, "pf_refresh_menu", _fake_refresh_menu)

    context = _DummyContext()
    asyncio.run(bot.pf_show_projection_below(context, 123, 12))

    assert context.user_data["pf_proj_horizon"] == 12
    assert any("Horizonte: 12M (~252 ruedas)." in msg for msg in sent_messages)
