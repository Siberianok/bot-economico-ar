import asyncio

import pytest

import bot_econ_full_plus_rank_alerts as bot


@pytest.fixture
def projection_state(monkeypatch):
    batches = [
        {
            "batch_id": "batch-63",
            "horizon": 63,
            "created_date": "2024-01-01",
            "created_at": 1_704_067_200.0,
            "symbols": ["GGAL.BA", "BMA.BA"],
            "predictions": {"GGAL.BA": 12.0, "BMA.BA": -4.0},
            "base_prices": {"GGAL.BA": 100.0, "BMA.BA": 200.0},
            "evaluated": False,
        }
    ]
    records = [
        {
            "batch_id": "batch-63",
            "symbol": "GGAL.BA",
            "horizon": 63,
            "base_price": 100.0,
            "projection": 12.0,
            "created_at": 1_704_067_200.0,
        },
        {
            "batch_id": "batch-63",
            "symbol": "BMA.BA",
            "horizon": 63,
            "base_price": 200.0,
            "projection": -4.0,
            "created_at": 1_704_067_200.0,
        },
    ]

    monkeypatch.setattr(bot, "PROJECTION_BATCHES", batches)
    monkeypatch.setattr(bot, "PROJECTION_RECORDS", records)
    return batches, records


def test_calibrate_projection_uses_coefficients(monkeypatch):
    monkeypatch.setattr(bot, "PROJ_CALIBRATION", {"3m": {"a": 1.5, "b": 0.5}})

    assert bot.calibrate_projection(10.0, "3m") == pytest.approx(6.5)
    assert bot.calibrate_projection(10.0, "6m") == pytest.approx(10.0)


def test_evaluate_projection_batches_marks_matured_and_updates_records(monkeypatch, projection_state):
    batches, records = projection_state

    class _FrozenDateTime:
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

    async def _fake_metrics(_session, symbols):
        assert set(symbols) == {"GGAL.BA", "BMA.BA"}
        return (
            {
                "GGAL.BA": {"last_px": 110.0},
                "BMA.BA": {"last_px": 210.0},
            },
            1_713_182_400,
        )

    monkeypatch.setattr(bot, "datetime", _FrozenDateTime)
    monkeypatch.setattr(bot, "metrics_for_symbols", _fake_metrics)
    monkeypatch.setattr(bot, "metric_last_price", lambda m: m.get("last_px"))

    updated = asyncio.run(bot._evaluate_projection_batches())

    assert updated == 1
    batch = batches[0]
    assert batch["evaluated"] is True
    assert batch["count"] == 2
    assert batch["hit_count"] == 1
    assert batch["hit_rate"] == pytest.approx(0.5)
    assert batch["mae"] == pytest.approx(5.5)

    rec_g = next(r for r in records if r["symbol"] == "GGAL.BA")
    rec_b = next(r for r in records if r["symbol"] == "BMA.BA")
    assert rec_g["evaluated"] is True
    assert rec_g["actual_return"] == pytest.approx(10.0)
    assert rec_b["actual_return"] == pytest.approx(5.0)
