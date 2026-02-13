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
