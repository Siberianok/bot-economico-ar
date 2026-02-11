import asyncio
import os
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
os.environ.setdefault("TELEGRAM_TOKEN", "test-token")

import bot_econ_full_plus_rank_alerts as bot


@pytest.fixture(autouse=True)
def _snapshot_stubs(monkeypatch):
    async def _fake_metrics(_session, symbols):
        return ({sym: {} for sym in symbols}, 1700000100)

    async def _fake_save_state():
        return None

    monkeypatch.setattr(bot, "metrics_for_symbols", _fake_metrics)
    monkeypatch.setattr(bot, "save_state", _fake_save_state)
    monkeypatch.setattr(bot, "get_tc_value", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(bot, "instrument_currency", lambda *_args, **_kwargs: "ARS")
    monkeypatch.setattr(bot, "metric_last_price", lambda met: met.get("last_px") if met else None)


def test_pf_market_snapshot_with_price_exact(monkeypatch):
    async def _metrics(_session, _symbols):
        return ({"AAA": {"last_px": 120.0, "last_chg": 1.5, "last_ts": 1700000000}}, 1700000100)

    monkeypatch.setattr(bot, "metrics_for_symbols", _metrics)

    pf = {
        "base": {"moneda": "ARS"},
        "items": [{"tipo": "accion", "simbolo": "AAA", "cantidad": 10.0, "importe": 1000.0}],
    }

    snapshot, *_ = asyncio.run(bot.pf_market_snapshot(pf))

    assert snapshot[0]["valor_actual"] == pytest.approx(1200.0)
    assert snapshot[0]["valuation_mode"] == "price"
    assert pf["items"][0]["last_valued_base"] == pytest.approx(1200.0)
    assert isinstance(pf["items"][0]["last_valued_ts"], int)


def test_pf_market_snapshot_without_price_uses_daily_change_estimate(monkeypatch):
    async def _metrics(_session, _symbols):
        return ({"BBB": {"last_chg": 5.0, "last_ts": 1700000000}}, 1700000100)

    monkeypatch.setattr(bot, "metrics_for_symbols", _metrics)

    pf = {
        "base": {"moneda": "ARS"},
        "items": [
            {
                "tipo": "accion",
                "simbolo": "BBB",
                "cantidad": 10.0,
                "importe": 1000.0,
                "last_valued_base": 1100.0,
                "last_valued_ts": 1699990000,
            }
        ],
    }

    snapshot, *_ = asyncio.run(bot.pf_market_snapshot(pf))

    assert snapshot[0]["precio_base"] is None
    assert snapshot[0]["valuation_mode"] == "estimated_from_daily_change"
    assert snapshot[0]["valor_actual"] == pytest.approx(1155.0)


def test_pf_market_snapshot_without_price_or_daily_change_keeps_last_known(monkeypatch):
    async def _metrics(_session, _symbols):
        return ({"CCC": {"last_ts": 1700000000}}, 1700000100)

    monkeypatch.setattr(bot, "metrics_for_symbols", _metrics)

    pf = {
        "base": {"moneda": "ARS"},
        "items": [
            {
                "tipo": "accion",
                "simbolo": "CCC",
                "cantidad": 4.0,
                "importe": 1000.0,
                "last_valued_base": 980.0,
                "last_valued_ts": 1699990000,
            }
        ],
    }

    snapshot, *_ = asyncio.run(bot.pf_market_snapshot(pf))

    assert snapshot[0]["valuation_mode"] == "last_known"
    assert snapshot[0]["valor_actual"] == pytest.approx(980.0)
