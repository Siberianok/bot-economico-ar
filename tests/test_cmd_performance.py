import asyncio
from types import SimpleNamespace

import bot_econ_full_plus_rank_alerts as bot


class DummyMessage:
    def __init__(self):
        self.replies = []

    async def reply_text(self, text, **kwargs):
        self.replies.append((text, kwargs))


class FrozenDateTime:
    @classmethod
    def now(cls, _tz):
        from datetime import datetime

        return datetime(2024, 4, 15, 12, 0, 0)

    @classmethod
    def strptime(cls, value, fmt):
        from datetime import datetime

        return datetime.strptime(value, fmt)

    @classmethod
    def fromtimestamp(cls, ts, tz=None):
        from datetime import datetime

        return datetime.fromtimestamp(ts, tz)


def _run_cmd_performance():
    message = DummyMessage()
    update = SimpleNamespace(effective_message=message)
    context = SimpleNamespace()
    asyncio.run(bot.cmd_performance(update, context))
    assert message.replies
    return message.replies[-1]


def test_cmd_performance_without_evaluated_shows_operational_status(monkeypatch):
    monkeypatch.setattr(bot, "datetime", FrozenDateTime)
    monkeypatch.setattr(
        bot,
        "PROJECTION_BATCHES",
        [
            {
                "batch_id": "b-63",
                "horizon": 63,
                "created_date": "2024-03-01",
                "created_at": 1_709_251_200.0,
                "evaluated": False,
            },
            {
                "batch_id": "b-126",
                "horizon": 126,
                "created_date": "2024-03-15",
                "created_at": 1_710_460_800.0,
                "evaluated": False,
            },
        ],
    )
    monkeypatch.setattr(
        bot,
        "_trading_days_between",
        lambda start, end: 10 if str(start) == "2024-03-01" else 5,
    )

    text, kwargs = _run_cmd_performance()

    assert "<b>Precisión histórica</b>" in text
    assert "Aún sin batches evaluados" in text
    assert "<b>Estado actual del motor de proyecciones</b>" in text
    assert "Total de batches registrados: 2" in text
    assert "Pendientes por horizonte: 126r: 1 · 63r: 1" in text
    assert "Último batch creado:" in text
    assert "Progreso estimado: 63r: 10/63" in text
    assert "126r: 5/126" in text
    assert kwargs.get("parse_mode") == bot.ParseMode.HTML


def test_cmd_performance_with_evaluated_keeps_historical_block(monkeypatch):
    monkeypatch.setattr(bot, "datetime", FrozenDateTime)
    monkeypatch.setattr(
        bot,
        "PROJECTION_BATCHES",
        [
            {
                "batch_id": "b-63-eval",
                "horizon": 63,
                "created_date": "2024-01-02",
                "created_at": 1_704_153_600.0,
                "evaluated": True,
                "mae": 2.5,
                "hit_rate": 0.5,
                "spearman": 0.7,
                "count": 4,
            },
            {
                "batch_id": "b-126-pending",
                "horizon": 126,
                "created_date": "2024-03-01",
                "created_at": 1_709_251_200.0,
                "evaluated": False,
            },
        ],
    )
    monkeypatch.setattr(bot, "_trading_days_between", lambda *_args: 10)

    text, _kwargs = _run_cmd_performance()

    assert "<b>Precisión histórica</b>" in text
    assert "3M (63 ruedas): MAE" in text
    assert "Spearman" in text
    assert "<b>Estado actual del motor de proyecciones</b>" in text
    assert "Total de batches registrados: 2" in text
    assert "Pendientes por horizonte: 126r: 1" in text
