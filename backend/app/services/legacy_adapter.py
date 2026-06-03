"""Lazy adapter for the existing Telegram bot module.

The adapter never starts polling, webhooks, schedulers or Telegram message sending.
It imports the legacy module only when an endpoint explicitly requests legacy data.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from types import ModuleType
from typing import Any

from aiohttp import ClientSession

from backend.app.config import get_settings


@dataclass
class LegacyResult:
    status: str
    data: Any
    warnings: list[str]
    source: str = "legacy_adapter"
    freshness: str = "unknown"


class LegacyAdapter:
    def __init__(self) -> None:
        self._module: ModuleType | None = None
        self._import_error: str | None = None

    @property
    def imported(self) -> bool:
        return self._module is not None

    def import_error(self) -> str | None:
        return self._import_error

    def _safe_import(self) -> ModuleType:
        if self._module is not None:
            return self._module
        settings = get_settings()
        os.environ.setdefault("TELEGRAM_TOKEN", settings.legacy_import_token)
        os.environ.setdefault("BOT_TOKEN", settings.legacy_import_token)
        os.environ.setdefault("STATE_PATH", settings.legacy_state_path)
        try:
            self._module = importlib.import_module("bot_econ_full_plus_rank_alerts")
        except Exception as exc:  # pragma: no cover - exercised through service fallback
            self._import_error = str(exc)
            raise
        return self._module

    def list_symbols(self, kind: str) -> LegacyResult:
        try:
            module = self._safe_import()
        except Exception as exc:
            return LegacyResult(
                status="not_available",
                data=None,
                warnings=[f"No se pudo importar legacy sin riesgo: {exc}"],
            )
        attr = "ACCIONES_BA" if kind == "acciones" else "CEDEARS_BA"
        symbols = getattr(module, attr, None)
        if not isinstance(symbols, list):
            return LegacyResult(
                status="not_available",
                data=None,
                warnings=[f"Listado legacy {attr} no disponible"],
            )
        return LegacyResult(
            status="ok",
            data=[str(symbol) for symbol in symbols],
            warnings=[],
            source=f"legacy.{attr}",
        )

    async def fetch_market_pulse(self) -> LegacyResult:
        settings = get_settings()
        if not settings.legacy_live_fetch_enabled:
            return LegacyResult(
                status="not_available",
                data=None,
                warnings=[
                    "Fetch live legacy desactivado. Activar LEGACY_LIVE_FETCH_ENABLED=1 para consultar fuentes reales.",
                ],
            )
        try:
            module = self._safe_import()
        except Exception as exc:
            return LegacyResult(
                status="not_available",
                data=None,
                warnings=[f"No se pudo importar legacy para market pulse: {exc}"],
            )
        async with ClientSession() as session:
            payload: dict[str, Any] = {}
            warnings: list[str] = []
            for name, func_name in (
                ("fx", "get_dolares"),
                ("reservas", "get_reservas_con_variacion"),
                ("inflacion", "get_inflacion_mensual"),
                ("riesgo_pais", "get_riesgo_pais"),
                ("bandas", "get_bandas_cambiarias"),
            ):
                func = getattr(module, func_name, None)
                if func is None:
                    payload[name] = None
                    warnings.append(f"Funcion legacy {func_name} no disponible")
                    continue
                try:
                    payload[name] = await func(session)
                except Exception as exc:
                    payload[name] = None
                    warnings.append(f"{func_name} fallo: {exc}")
        status = "partial" if warnings else "ok"
        return LegacyResult(status=status, data=payload, warnings=warnings, freshness="fallback")


legacy_adapter = LegacyAdapter()

