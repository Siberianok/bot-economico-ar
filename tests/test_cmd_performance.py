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


def _run_cmd_performance(args=None):
    message = DummyMessage()
    update = SimpleNamespace(effective_message=message)
    context = SimpleNamespace(args=args or [])
    asyncio.run(bot.cmd_performance(update, context))
    assert message.replies
    return message.replies[-1]


def _run_cmd_performance_detalle():
    message = DummyMessage()
    update = SimpleNamespace(effective_message=message)
    context = SimpleNamespace(args=[])
    asyncio.run(bot.cmd_performance_detalle(update, context))
    assert message.replies
    return message.replies[-1]


def test_cmd_performance_default_returns_short_executive_summary(monkeypatch):
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
                "evaluated_at": 1_710_000_000.0,
                "mae": 2.5,
                "hit_rate": 0.6,
                "hit_count": 3,
                "spearman": 0.7,
                "count": 5,
            },
            {
                "batch_id": "b-126-pending",
                "horizon": 126,
                "created_date": "2024-03-15",
                "created_at": 1_710_460_800.0,
                "evaluated": False,
            },
        ],
    )

    text, kwargs = _run_cmd_performance()

    assert "Estado general:" in text
    assert "umbrales: Hit ≥" in text
    assert "3M (63 ruedas): MAE" in text
    assert "6M (126 ruedas): sin datos evaluados." in text
    assert "Batches pendientes: 1" in text
    assert "Última evaluación:" in text
    assert "<b>Últimos batches</b>" not in text
    assert "Progreso estimado:" not in text
    assert kwargs.get("parse_mode") == bot.ParseMode.HTML


def test_cmd_performance_full_and_detail_command_return_extended_output(monkeypatch):
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
                "evaluated_at": 1_710_000_000.0,
                "mae": 3.1,
                "hit_rate": 0.52,
                "hit_count": 6,
                "spearman": 0.42,
                "count": 12,
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

    text_full, _ = _run_cmd_performance(args=["full"])
    text_detail, _ = _run_cmd_performance_detalle()

    for text in (text_full, text_detail):
        assert "Estado general:" in text
        assert "<b>Últimos batches</b>" in text
        assert "<b>Estado actual del motor de proyecciones</b>" in text
        assert "Pendientes por horizonte: 126r: 1" in text
        assert "Progreso estimado: 126r: 10/126" in text
