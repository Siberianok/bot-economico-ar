import asyncio
from dataclasses import replace
from types import SimpleNamespace

from backend.app.bot import telegram_bot
from backend.app.bot.telegram_bot import TelegramRuntime
from backend.app.config import get_settings


class FakeBot:
    def __init__(self, events):
        self.events = events
        self.webhook_kwargs = None

    async def set_my_commands(self, _commands):
        self.events.append("commands")

    async def set_webhook(self, **kwargs):
        self.webhook_kwargs = kwargs
        self.events.append("webhook")


class FakeApplication:
    def __init__(self, events):
        self.events = events
        self.bot = FakeBot(events)
        self.update_queue = asyncio.Queue()

    async def initialize(self):
        self.events.append("initialize")

    async def start(self):
        self.events.append("start")

    async def stop(self):
        self.events.append("stop")

    async def shutdown(self):
        self.events.append("shutdown")


async def _wait_forever(*_args):
    await asyncio.Event().wait()


def test_runtime_preserves_updates_and_persists_after_stop(monkeypatch):
    events = []
    application = FakeApplication(events)

    async def load_state():
        events.append("load")

    async def save_state():
        events.append("save")

    legacy = SimpleNamespace(
        load_state=load_state,
        save_state=save_state,
        build_application=lambda: application,
        _schedule_all_subs=lambda _app: events.append("subs"),
        _schedule_projection_calibration=lambda _app: events.append("calibration"),
        _schedule_projection_performance=lambda _app: events.append("performance"),
        BOT_COMMANDS=[],
        alerts_loop=_wait_forever,
        instrument_catalog_loop=_wait_forever,
        keepalive_loop=_wait_forever,
    )
    monkeypatch.setattr(telegram_bot.importlib, "import_module", lambda _name: legacy)
    settings = replace(
        get_settings(),
        telegram_runtime_enabled=True,
        telegram_token_configured=True,
        webhook_secret="valid_secret",
        public_base_url="https://example.test",
        keepalive_enabled=False,
    )
    runtime = TelegramRuntime(settings)

    async def scenario():
        await runtime.start()
        assert runtime.state == "running"
        assert application.bot.webhook_kwargs == {
            "url": "https://example.test/telegram/webhook",
            "drop_pending_updates": False,
            "secret_token": "valid_secret",
        }
        await runtime.stop()

    asyncio.run(scenario())

    assert events.index("stop") < events.index("save") < events.index("shutdown")


def test_runtime_cleans_up_after_partial_initialization_failure(monkeypatch):
    events = []

    class FailingApplication(FakeApplication):
        async def initialize(self):
            self.events.append("initialize")
            raise RuntimeError("initialization failed")

    application = FailingApplication(events)

    async def load_state():
        events.append("load")

    async def save_state():
        events.append("save")

    legacy = SimpleNamespace(
        load_state=load_state,
        save_state=save_state,
        build_application=lambda: application,
        _schedule_all_subs=lambda _app: events.append("subs"),
        _schedule_projection_calibration=lambda _app: events.append("calibration"),
        _schedule_projection_performance=lambda _app: events.append("performance"),
        BOT_COMMANDS=[],
        alerts_loop=_wait_forever,
        instrument_catalog_loop=_wait_forever,
        keepalive_loop=_wait_forever,
    )
    monkeypatch.setattr(telegram_bot.importlib, "import_module", lambda _name: legacy)
    settings = replace(
        get_settings(),
        telegram_runtime_enabled=True,
        telegram_token_configured=True,
        webhook_secret="valid_secret",
        public_base_url="https://example.test",
        keepalive_enabled=False,
    )
    runtime = TelegramRuntime(settings)

    asyncio.run(runtime.start())

    assert runtime.state == "error"
    assert runtime.application is None
    assert runtime.legacy is None
    assert events[-3:] == ["stop", "save", "shutdown"]
