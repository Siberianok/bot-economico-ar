import asyncio

import bot_econ_full_plus_rank_alerts as bot


class _DummyChat:
    id = 101


class _DummyUser:
    id = 202


class _DummyMessage:
    def __init__(self):
        self.replies = []

    async def reply_text(self, text, **_kwargs):
        self.replies.append(text)


class _DummyCallbackQuery:
    def __init__(self, data: str):
        self.data = data

    async def answer(self):
        return None


class _DummyUpdate:
    def __init__(self, data: str, message: _DummyMessage):
        self.callback_query = _DummyCallbackQuery(data)
        self.effective_chat = _DummyChat()
        self.effective_user = _DummyUser()
        self.effective_message = message


class _DummyContext:
    user_data = {}


def test_acc_top3_then_top5_callbacks_do_not_apply_wait_throttling(monkeypatch):
    used_throttle_keys = []
    message = _DummyMessage()

    async def _fake_rank_top3(update, _symbols, _title, throttle_key=None):
        used_throttle_keys.append(("ACC:TOP3", throttle_key))
        await update.effective_message.reply_text("top3 ok")

    async def _fake_rank_top5(update, _symbols, _title, throttle_key=None):
        used_throttle_keys.append(("ACC:TOP5", throttle_key))
        await update.effective_message.reply_text("top5 ok")

    async def _fake_dec(*_args, **_kwargs):
        return None

    monkeypatch.setattr(bot, "_rank_top3", _fake_rank_top3)
    monkeypatch.setattr(bot, "_rank_proj5", _fake_rank_top5)
    monkeypatch.setattr(bot, "dec_and_maybe_show", _fake_dec)

    context = _DummyContext()
    asyncio.run(bot.acc_ced_cb(_DummyUpdate("ACC:TOP3", message), context))
    asyncio.run(bot.acc_ced_cb(_DummyUpdate("ACC:TOP5", message), context))

    assert used_throttle_keys == [
        ("ACC:TOP3", None),
        ("ACC:TOP5", None),
    ]
    assert all("⏳ Consultá de nuevo en unos segundos." not in text for text in message.replies)


def test_ced_top3_and_top5_callbacks_keep_same_no_wait_throttling_policy(monkeypatch):
    used_throttle_keys = []
    message = _DummyMessage()

    async def _fake_rank_top3(update, _symbols, _title, throttle_key=None):
        used_throttle_keys.append(("CED:TOP3", throttle_key))
        await update.effective_message.reply_text("ced top3 ok")

    async def _fake_rank_top5(update, _symbols, _title, throttle_key=None):
        used_throttle_keys.append(("CED:TOP5", throttle_key))
        await update.effective_message.reply_text("ced top5 ok")

    async def _fake_dec(*_args, **_kwargs):
        return None

    monkeypatch.setattr(bot, "_rank_top3", _fake_rank_top3)
    monkeypatch.setattr(bot, "_rank_proj5", _fake_rank_top5)
    monkeypatch.setattr(bot, "dec_and_maybe_show", _fake_dec)

    context = _DummyContext()
    asyncio.run(bot.acc_ced_cb(_DummyUpdate("CED:TOP3", message), context))
    asyncio.run(bot.acc_ced_cb(_DummyUpdate("CED:TOP5", message), context))

    assert used_throttle_keys == [
        ("CED:TOP3", None),
        ("CED:TOP5", None),
    ]
    assert all("⏳ Consultá de nuevo en unos segundos." not in text for text in message.replies)
