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

    async def edit_message_text(self, *args, **kwargs):
        return None


class _DummyUpdate:
    def __init__(self, data: str):
        self.callback_query = _DummyCallbackQuery(data)


class _DummyBot:
    def __init__(self):
        self.messages = []

    async def send_message(self, chat_id, text, **kwargs):
        self.messages.append((chat_id, text, kwargs))


class _DummyContext:
    def __init__(self):
        self.user_data = {}
        self.bot = _DummyBot()


def test_pf_export_filter_history_entries_by_ts_and_invalids():
    entries = [
        {"ts": 1_700_000_000, "snapshot": [{"symbol": "GGAL.BA"}]},
        {"ts": 1_710_000_000, "snapshot": [{"symbol": "YPFD.BA"}]},
        {"ts": "invalid", "snapshot": [{"symbol": "PAMP.BA"}]},
        {"snapshot": [{"symbol": "BMA.BA"}]},
    ]

    filtered, invalid_count = bot._pf_export_filter_history_entries(
        entries,
        from_ts=1_705_000_000,
        to_ts=1_715_000_000,
    )

    assert len(filtered) == 1
    assert filtered[0]["snapshot"][0]["symbol"] == "YPFD.BA"
    assert invalid_count == 2


def test_pf_export_history_rejects_inverted_range(monkeypatch):
    chat_id = 123
    bot.PF_HISTORY[chat_id] = [{"ts": 1_710_000_000, "snapshot": [{"symbol": "GGAL.BA"}]}]
    context = _DummyContext()
    context.user_data[bot.PF_EXPORT_RANGE_FROM_KEY] = 2_000
    context.user_data[bot.PF_EXPORT_RANGE_TO_KEY] = 1_000
    context.user_data[bot.PF_EXPORT_RANGE_LABEL_KEY] = "invertido"

    asyncio.run(bot.pf_export_history(context, chat_id))

    assert context.bot.messages
    assert "Rango inválido" in context.bot.messages[0][1]


def test_pf_export_range_preset_callback_sets_user_data_and_exports(monkeypatch):
    called = {}

    async def _fake_export_history(context, chat_id):
        called["export_chat_id"] = chat_id
        called["label"] = context.user_data.get(bot.PF_EXPORT_RANGE_LABEL_KEY)

    async def _fake_refresh_menu(context, chat_id, force_new=False):
        called["refresh_chat_id"] = chat_id
        called["force_new"] = force_new

    monkeypatch.setattr(bot, "pf_export_history", _fake_export_history)
    monkeypatch.setattr(bot, "pf_refresh_menu", _fake_refresh_menu)

    update = _DummyUpdate("PF:EXPORT:HISTORY:RANGE:PRESET:30d")
    context = _DummyContext()

    asyncio.run(bot.pf_menu_cb(update, context))

    assert called["export_chat_id"] == 123
    assert called["refresh_chat_id"] == 123
    assert called["force_new"] is True
    assert context.user_data[bot.PF_EXPORT_RANGE_FROM_KEY] is not None
    assert context.user_data[bot.PF_EXPORT_RANGE_TO_KEY] is not None
    assert "a" in context.user_data[bot.PF_EXPORT_RANGE_LABEL_KEY]
