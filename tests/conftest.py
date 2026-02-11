import os
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
os.environ.setdefault("TELEGRAM_TOKEN", "test-token")


@pytest.fixture
def dirty_state_payload():
    return {
        "alerts": {
            "100": [
                {
                    "kind": "ticker",
                    "symbol": "GGAL.BA",
                    "op": ">",
                    "value": "1000",
                    "last_trigger_ts": "1711111111",
                    "last_trigger_price": "999.9",
                    "armed": 0,
                },
                {"kind": "invalid", "value": 1},
            ],
            "bad-chat": [{"kind": "ticker", "symbol": "BMA.BA", "op": "<", "value": 10}],
        },
        "subs": {"100": {"daily": 930, "foo": "bar"}},
        "pf": {
            "100": {
                "base": {"moneda": "ARS"},
                "monto": 1000,
                "items": [{"simbolo": "GGAL.BA"}, "ignore-me"],
            }
        },
        "projection_records": [
            {
                "symbol": "GGAL.BA",
                "horizon": 63,
                "base_price": "100",
                "projection": "10",
                "created_at": "1700000000",
                "created_date": "2024-01-01",
                "batch_id": "batch-1",
            },
            {"symbol": "BAD", "horizon": 10, "base_price": 1, "projection": 1, "created_at": 1},
        ],
        "projection_batches": [
            {
                "batch_id": "batch-1",
                "horizon": 63,
                "created_at": "1700000000",
                "symbols": ["GGAL.BA"],
                "predictions": {"GGAL.BA": "8.0", "": 1},
                "base_prices": {"GGAL.BA": "100.0", "X": "bad"},
                "created_date": "2024-01-01",
            },
            {"batch_id": "broken", "horizon": 63, "created_at": 1, "symbols": "nope", "predictions": {}, "base_prices": {}},
        ],
    }
