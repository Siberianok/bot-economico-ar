import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
os.environ.setdefault("TELEGRAM_TOKEN", "test-token")

import copy

from bot_econ_full_plus_rank_alerts import _evaluate_composite_rule


def _snapshot(cur_ticker=110.0, cur_fx=100.0, ret_ticker=5.0, ret_fx=1.0):
    return {
        "ticker:GGAL.BA": {"current": cur_ticker, "ret": {"1d": ret_ticker, "3d": 7.0, "7d": 9.0}},
        "fx:ccl:venta": {"current": cur_fx, "ret": {"1d": ret_fx, "3d": 2.0, "7d": 3.0}},
    }


def test_composite_and_with_variation_and_cross_triggers():
    rule = {
        "kind": "composite",
        "logic": "AND",
        "cooldown_secs": 1800,
        "conditions": [
            {
                "type": "variation",
                "indicator": {"kind": "ticker", "symbol": "GGAL.BA"},
                "window": "3d",
                "op": ">",
                "value": 4.0,
            },
            {
                "type": "cross",
                "left": {"kind": "ticker", "symbol": "GGAL.BA"},
                "right": {"kind": "fx", "type": "ccl", "side": "venta"},
                "direction": "up",
            },
        ],
        "last_snapshot": _snapshot(cur_ticker=90.0, cur_fx=100.0),
    }

    ok, changed, reasons = _evaluate_composite_rule(rule, _snapshot(cur_ticker=110.0, cur_fx=100.0), now_ts=10_000)
    assert ok is True
    assert changed is True
    assert len(reasons) == 2


def test_composite_or_triggers_when_one_condition_true():
    rule = {
        "kind": "composite",
        "logic": "OR",
        "conditions": [
            {
                "type": "variation",
                "indicator": {"kind": "ticker", "symbol": "GGAL.BA"},
                "window": "1d",
                "op": ">",
                "value": 10.0,
            },
            {
                "type": "variation",
                "indicator": {"kind": "fx", "type": "ccl", "side": "venta"},
                "window": "1d",
                "op": ">",
                "value": 0.5,
            },
        ],
        "last_snapshot": _snapshot(),
    }

    ok, changed, reasons = _evaluate_composite_rule(rule, _snapshot(ret_ticker=1.0, ret_fx=0.8), now_ts=5_000)
    assert ok is True
    assert changed is True
    assert len(reasons) == 1


def test_composite_respects_cooldown():
    rule = {
        "kind": "composite",
        "logic": "OR",
        "cooldown_secs": 3600,
        "last_trigger_ts": 9_000,
        "conditions": [
            {
                "type": "variation",
                "indicator": {"kind": "fx", "type": "ccl", "side": "venta"},
                "window": "1d",
                "op": ">",
                "value": 0.1,
            }
        ],
        "last_snapshot": _snapshot(),
    }

    ok, changed, reasons = _evaluate_composite_rule(rule, _snapshot(ret_fx=2.0), now_ts=10_000)
    assert ok is False
    assert changed is False
    assert reasons == []


def test_composite_missing_data_is_safe_and_false():
    rule = {
        "kind": "composite",
        "logic": "AND",
        "conditions": [
            {
                "type": "variation",
                "indicator": {"kind": "metric", "type": "riesgo"},
                "window": "7d",
                "op": ">",
                "value": 1.0,
            }
        ],
    }

    snap = {"metric:riesgo": {"current": 700.0, "ret": {"1d": None, "3d": None, "7d": None}}}
    ok, changed, reasons = _evaluate_composite_rule(rule, copy.deepcopy(snap), now_ts=500)
    assert ok is False
    assert changed is True
    assert reasons == []
