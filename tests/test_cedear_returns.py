from datetime import datetime, timedelta
from bisect import bisect_left
from pathlib import Path
import os
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("TELEGRAM_TOKEN", "test-token")

from bot_econ_full_plus_rank_alerts import _metrics_from_chart


def _trading_days(start: datetime, count: int) -> list[datetime]:
    days: list[datetime] = []
    cur = start
    while len(days) < count:
        if cur.weekday() < 5:
            days.append(cur)
        cur += timedelta(days=1)
    return days


def _closest_index(ts_vals: list[int], target: int) -> int:
    if not ts_vals:
        return 0
    pos = bisect_left(ts_vals, target)
    if pos <= 0:
        return 0
    if pos >= len(ts_vals):
        return len(ts_vals) - 1
    before = pos - 1
    after = pos
    if abs(ts_vals[after] - target) < abs(ts_vals[before] - target):
        return after
    return before


def test_amd_3m_positive_uses_trading_window() -> None:
    """AMD: 3M positivo usando el mismo rango (63 ruedas) que el backtest."""
    start = datetime(2024, 1, 2)
    dates = _trading_days(start, 220)
    ts = [int(d.timestamp()) for d in dates]

    closes = [105.0 for _ in ts]
    closes[-1] = 110.0
    closes[-63] = 100.0
    closes[-10] = 50.0

    target = int((dates[-1] - timedelta(days=90)).timestamp())
    idx_calendar = _closest_index(ts, target)
    assert idx_calendar != len(closes) - 63
    if idx_calendar != len(closes) - 63:
        closes[idx_calendar] = 150.0

    res = {"timestamp": ts, "indicators": {"adjclose": [{"adjclose": closes}]}}
    metrics = _metrics_from_chart(res)

    assert metrics is not None
    ret3 = metrics.get("3m")
    assert ret3 is not None
    assert ret3 > 0
    assert 9.5 < ret3 < 10.5
