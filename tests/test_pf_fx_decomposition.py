import math
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
os.environ.setdefault("TELEGRAM_TOKEN", "test-token")

from bot_econ_full_plus_rank_alerts import _pf_fx_decomposition


def test_pf_fx_decomposition_uses_metrics_last_px_for_usd_instrument_base_ars():
    snapshot = [
        {
            "cantidad": 10.0,
            "cantidad_derivada": False,
            "invertido": 10000.0,
            "inst_currency": "USD",
            "fx_rate_item": 100.0,
            "fx_rate": 120.0,
            "metrics": {"last_px": 12.0},
        }
    ]

    fx_price_effect, fx_fx_effect, fx_has_data = _pf_fx_decomposition(snapshot, "ARS")

    assert fx_has_data is True
    assert math.isclose(fx_price_effect, 2000.0, rel_tol=1e-9)
    assert math.isclose(fx_fx_effect, 2400.0, rel_tol=1e-9)
