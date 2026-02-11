import asyncio
import os
import sys
from pathlib import Path
from time import time

os.environ.setdefault("TELEGRAM_TOKEN", "test-token")
sys.path.append(str(Path(__file__).resolve().parents[1]))

import bot_econ_full_plus_rank_alerts as app


def _portfolio_with_fx_item(*, fx_rate: float, fx_ts: int, tc_val: float, tc_ts: int):
    return {
        "base": {
            "moneda": "ARS",
            "tc": "mep",
            "tc_valor": tc_val,
            "tc_timestamp": tc_ts,
        },
        "items": [
            {
                "simbolo": "BTC-USD",
                "tipo": "cripto",
                "cantidad": 1,
                "importe": 1000,
                "fx_rate": fx_rate,
                "fx_ts": fx_ts,
            }
        ],
    }


def _mock_metrics(symbol: str, px: float, ts: int):
    return {
        symbol: {
            "last_px": px,
            "currency": "USD",
            "last_ts": ts,
        }
    }


def test_pf_market_snapshot_uses_fresh_item_fx(monkeypatch):
    now = int(time())
    pf = _portfolio_with_fx_item(fx_rate=900.0, fx_ts=now - 1800, tc_val=1000.0, tc_ts=now - 120)

    async def fake_metrics_for_symbols(_session, _symbols):
        return _mock_metrics("BTC-USD", 10.0, now), now

    async def fake_get_tc_value(_session, _tc_name):
        return 1234.0

    async def fake_save_state():
        return None

    monkeypatch.setattr(app, "metrics_for_symbols", fake_metrics_for_symbols)
    monkeypatch.setattr(app, "get_tc_value", fake_get_tc_value)
    monkeypatch.setattr(app, "save_state", fake_save_state)

    snapshot, *_ = asyncio.run(app.pf_market_snapshot(pf))
    entry = snapshot[0]

    assert entry["fx_rate"] == 900.0
    assert entry["fx_ts"] == now - 1800
    assert entry["fx_rate_used"] == 900.0
    assert entry["fx_ts_used"] == now - 1800
    assert entry["valor_actual"] == 9000.0
    assert pf["items"][0]["fx_rate_used"] == 900.0
    assert pf["items"][0]["fx_ts_used"] == now - 1800


def test_pf_market_snapshot_uses_base_fx_when_item_stale(monkeypatch):
    now = int(time())
    base_tc = 1000.0
    base_ts = now - 300
    pf = _portfolio_with_fx_item(fx_rate=900.0, fx_ts=now - (30 * 60 * 60), tc_val=base_tc, tc_ts=base_ts)

    async def fake_metrics_for_symbols(_session, _symbols):
        return _mock_metrics("BTC-USD", 10.0, now), now

    async def fake_get_tc_value(_session, _tc_name):
        return 1234.0

    async def fake_save_state():
        return None

    monkeypatch.setattr(app, "metrics_for_symbols", fake_metrics_for_symbols)
    monkeypatch.setattr(app, "get_tc_value", fake_get_tc_value)
    monkeypatch.setattr(app, "save_state", fake_save_state)

    snapshot, *_ = asyncio.run(app.pf_market_snapshot(pf))
    entry = snapshot[0]

    assert entry["fx_rate"] == base_tc
    assert entry["fx_ts"] == base_ts
    assert entry["fx_rate_used"] == base_tc
    assert entry["fx_ts_used"] == base_ts
    assert entry["valor_actual"] == 10000.0
    assert pf["items"][0]["fx_rate_used"] == base_tc
    assert pf["items"][0]["fx_ts_used"] == base_ts
