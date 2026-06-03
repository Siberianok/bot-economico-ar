import asyncio
from types import SimpleNamespace

import bot_econ_full_plus_rank_alerts as bot


class DummyMessage:
    def __init__(self):
        self.replies = []

    async def reply_text(self, text, **kwargs):
        self.replies.append((text, kwargs))


def _run(coro):
    return asyncio.run(coro)


def test_dashboard_without_miniapp_url_returns_exact_message(monkeypatch):
    monkeypatch.setattr(bot, "MINIAPP_URL", "")
    message = DummyMessage()
    update = SimpleNamespace(effective_message=message)
    context = SimpleNamespace()

    _run(bot.cmd_dashboard(update, context))

    assert message.replies
    text, kwargs = message.replies[-1]
    assert text == "Dashboard no configurado. Definí MINIAPP_URL."
    assert "reply_markup" not in kwargs


def test_dashboard_with_miniapp_url_uses_webapp_button(monkeypatch):
    url = "https://observatorio-economico-miniapp.onrender.com"
    monkeypatch.setattr(bot, "MINIAPP_URL", url)
    message = DummyMessage()
    update = SimpleNamespace(effective_message=message)
    context = SimpleNamespace()

    _run(bot.cmd_dashboard(update, context))

    text, kwargs = message.replies[-1]
    assert "Observatorio Económico" in text
    assert "Abrí el dashboard interactivo desde Telegram." in text
    button = kwargs["reply_markup"].inline_keyboard[0][0]
    assert button.text == "Abrir Dashboard"
    assert button.web_app.url == url


def test_start_adds_dashboard_button_without_removing_existing_buttons(monkeypatch):
    url = "https://observatorio-economico-miniapp.onrender.com"
    monkeypatch.setattr(bot, "MINIAPP_URL", url)
    message = DummyMessage()
    update = SimpleNamespace(effective_message=message)
    context = SimpleNamespace(user_data={})

    _run(bot.cmd_start(update, context))

    _, kwargs = message.replies[-1]
    rows = kwargs["reply_markup"].inline_keyboard
    button_texts = [button.text for row in rows for button in row]
    assert "Abrir Dashboard" in button_texts
    assert any(button.callback_data == "ECO:DOLAR" for row in rows for button in row)
    assert any(button.callback_data == "PF:MENU" for row in rows for button in row)
    dashboard_button = next(button for row in rows for button in row if button.text == "Abrir Dashboard")
    assert dashboard_button.web_app.url == url


def test_bot_commands_keep_legacy_commands_and_add_dashboard():
    command_names = {command.command for command in bot.BOT_COMMANDS}
    assert {
        "economia",
        "acciones",
        "cedears",
        "alertas_menu",
        "portafolio",
        "subs",
        "performance",
        "dashboard",
    }.issubset(command_names)
