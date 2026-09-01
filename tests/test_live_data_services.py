import asyncio
from pathlib import Path

from backend.app.config import get_settings
from backend.app.services import market_service, screener_service
from backend.app.services.legacy_adapter import LegacyResult


def test_market_pulse_derives_real_fx_gaps(monkeypatch):
    async def fake_fetch():
        return LegacyResult(
            status="ok",
            data={
                "fx": {
                    "oficial": {"venta": 1000.0},
                    "blue": {"venta": 1200.0},
                    "mep": {"venta": 1100.0},
                },
                "reservas": (50_000.0, "2026-08-28", None, False),
                "inflacion": (2.0, "2026-07-31", 0.0),
                "riesgo_pais": (500, "2026-08-28", 0.0, False),
                "bandas": {"banda_inferior": 800.0, "banda_superior": 1500.0},
            },
            warnings=[],
            freshness="current",
        )

    monkeypatch.setattr(market_service.legacy_adapter, "fetch_market_pulse", fake_fetch)
    payload = asyncio.run(market_service.get_market_pulse())

    assert payload["status"] == "partial"
    assert payload["freshness"] == "current"
    assert payload["data"]["brechas"]["data"]["blue_pct"] == 20.0
    assert payload["data"]["brechas"]["data"]["mep_pct"] == 10.0


def test_screener_maps_real_metrics_and_keeps_unknown_fields_unavailable(monkeypatch):
    monkeypatch.setattr(
        screener_service.legacy_adapter,
        "list_symbols",
        lambda _kind: LegacyResult("ok", ["AAA.BA", "BBB.BA"], [], source="legacy.ACCIONES_BA"),
    )

    async def fake_metrics(_kind):
        return LegacyResult(
            status="ok",
            data={
                "metrics": {
                    "AAA.BA": {"last_px": 120.0, "1m": 4.0, "3m": 8.0, "6m": 20.0, "vol_ann": 30.0},
                    "BBB.BA": {"last_px": 90.0, "1m": -2.0, "3m": 1.0, "6m": 5.0, "vol_ann": 45.0},
                },
                "last_timestamp": 1_787_900_000,
            },
            warnings=[],
            source="legacy.metrics_for_symbols_cached",
            freshness="current",
        )

    monkeypatch.setattr(screener_service.legacy_adapter, "fetch_screener_metrics", fake_metrics)
    monkeypatch.setattr(
        screener_service.legacy_adapter,
        "calculate_projection_centers",
        lambda values: {"3m": values["1m"] * 3, "6m": values["1m"] * 6},
    )

    payload = asyncio.run(screener_service.get_screener("acciones"))
    first = payload["data"]["items"][0]

    assert payload["status"] == "partial"
    assert payload["source"] == "legacy.metrics_for_symbols_cached"
    assert first["symbol"] == "AAA.BA"
    assert first["price"] == 120.0
    assert first["score"] == 100.0
    assert first["risk"] == "30.0% anual"
    assert first["projection"]["3m"] == 12.0
    assert first["liquidity"] is None


def test_api_only_legacy_state_defaults_to_isolated_project_path(monkeypatch):
    monkeypatch.delenv("LEGACY_STATE_PATH", raising=False)
    get_settings.cache_clear()
    try:
        state_path = Path(get_settings().legacy_state_path).resolve()
        expected_path = (Path.cwd() / ".codex" / ".tmp" / "legacy-adapter-state.json").resolve()
        assert state_path == expected_path
    finally:
        get_settings.cache_clear()
