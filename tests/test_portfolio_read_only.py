import asyncio
import copy
from types import SimpleNamespace

from backend.app.services import portfolio_service
from backend.app.services.legacy_adapter import LegacyAdapter, LegacyResult


def test_portfolio_state_inspection_is_user_scoped_deep_copy():
    original = {
        "base": {"moneda": "ARS"},
        "items": [{"simbolo": "GGAL.BA", "importe": 1000.0}],
    }
    adapter = LegacyAdapter()
    adapter._module = SimpleNamespace(PF={123: original})

    found = adapter.inspect_portfolio_state("123")
    missing = adapter.inspect_portfolio_state("999")
    found.data["portfolio"]["items"][0]["importe"] = 1.0

    assert found.status == "ok"
    assert found.data["item_count"] == 1
    assert missing.status == "not_available"
    assert original["items"][0]["importe"] == 1000.0


def test_portfolio_summary_values_real_snapshot_without_mutating_pf(monkeypatch):
    legacy_portfolio = {
        "base": {
            "moneda": "ARS",
            "tc": "mep",
            "tc_valor": 1500.0,
            "benchmark": "merval",
        },
        "items": [
            {
                "simbolo": "AAA.BA",
                "tipo": "accion",
                "cantidad": 10.0,
                "importe": 1000.0,
                "moneda_nativa": "ARS",
                "added_ts": 1_700_000_000,
            },
            {
                "simbolo": "BTC-USD",
                "tipo": "cripto",
                "cantidad": 0.01,
                "importe": 1200.0,
                "moneda_nativa": "USD",
                "added_ts": 1_700_100_000,
            },
        ],
    }
    original = copy.deepcopy(legacy_portfolio)

    async def fake_snapshot(_user_id):
        return LegacyResult(
            status="ok",
            data={
                "portfolio": legacy_portfolio,
                "metrics": {
                    "AAA.BA": {"last_px": 120.0, "1m": 2.0, "vol_ann": 20.0},
                    "BTC-USD": {"last_px": 100.0, "1m": 3.0, "vol_ann": 60.0},
                    "^MERV": {"last_px": 2_000_000.0, "last_chg": 1.2},
                },
                "last_timestamp": 1_787_900_000,
            },
            warnings=[],
            source="legacy.PF.read_only",
            freshness="current",
        )

    monkeypatch.setattr(portfolio_service.legacy_adapter, "fetch_portfolio_snapshot", fake_snapshot)
    monkeypatch.setattr(
        portfolio_service.legacy_adapter,
        "calculate_projection_centers",
        lambda values: {"3m": values["1m"] * 3, "6m": values["1m"] * 6},
    )

    payload = asyncio.run(portfolio_service.get_portfolio_summary("123"))

    assert legacy_portfolio == original
    assert payload["status"] == "partial"
    assert payload["source"] == "legacy.PF.read_only"
    assert payload["data"]["invested_value"]["data"]["value"] == 2200.0
    assert payload["data"]["current_value"]["data"]["value"] == 2700.0
    assert payload["data"]["current_value"]["data"]["coverage_pct"] == 100.0
    assert payload["data"]["total_return"]["data"]["value_pct"] > 22.7
    assert payload["data"]["result_absolute"]["data"]["value"] == 500.0
    assert payload["data"]["twr"]["status"] == "partial"
    assert payload["data"]["mwr"]["status"] == "partial"
    assert payload["data"]["benchmark"]["data"]["daily_change_pct"] == 1.2
    assert payload["data"]["volatility"]["data"]["value_pct"] > 37.0
    assert payload["data"]["valuation"]["data"]["coverage_pct"] == 100.0


def test_portfolio_summary_keeps_unknown_user_not_available(monkeypatch):
    async def fake_snapshot(_user_id):
        return LegacyResult(
            status="not_available",
            data=None,
            warnings=["El usuario no tiene un portfolio legacy cargado"],
        )

    monkeypatch.setattr(portfolio_service.legacy_adapter, "fetch_portfolio_snapshot", fake_snapshot)
    payload = asyncio.run(portfolio_service.get_portfolio_summary("999"))

    assert payload["status"] == "not_available"
    assert payload["data"]["current_value"]["data"] is None
