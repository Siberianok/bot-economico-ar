"""Lazy adapter for the existing Telegram bot module.

The adapter never starts polling, webhooks, schedulers or Telegram message sending.
It imports the legacy module only when an endpoint explicitly requests legacy data.
"""

from __future__ import annotations

import asyncio
import copy
import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from aiohttp import ClientSession, ClientTimeout

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
        state_path = Path(settings.legacy_state_path)
        state_path.parent.mkdir(parents=True, exist_ok=True)
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

    def calculate_projection_centers(self, metrics: dict[str, Any]) -> dict[str, float | None]:
        try:
            module = self._safe_import()
        except Exception:
            return {"3m": None, "6m": None}

        projections: dict[str, float | None] = {}
        for horizon, function_name in (("3m", "projection_3m"), ("6m", "projection_6m")):
            function = getattr(module, function_name, None)
            if function is None:
                projections[horizon] = None
                continue
            try:
                result = function(metrics)
                center = result[0] if isinstance(result, (list, tuple)) and result else None
                projections[horizon] = (
                    float(center) if isinstance(center, (int, float)) and not isinstance(center, bool) else None
                )
            except Exception:
                projections[horizon] = None
        return projections

    def list_news_feeds(self) -> LegacyResult:
        """Copy the legacy RSS catalog without running its stateful news pipeline."""
        try:
            module = self._safe_import()
        except Exception as exc:
            return LegacyResult(
                status="not_available",
                data=None,
                warnings=[f"No se pudo leer el catalogo RSS legacy: {exc}"],
            )
        primary = getattr(module, "RSS_FEEDS", None)
        extended = getattr(module, "RSS_FEEDS_EXTENDED", None)
        if not isinstance(primary, list):
            return LegacyResult(
                status="not_available",
                data=None,
                warnings=["Catalogo RSS legacy no disponible"],
            )
        feeds = [str(url) for url in [*primary, *(extended or [])] if str(url).startswith("http")]
        return LegacyResult(
            status="ok",
            data=feeds,
            warnings=[],
            source="legacy.RSS_FEEDS.read_only",
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
        calls = (
            ("fx", "get_dolares"),
            ("reservas", "get_reservas_con_variacion"),
            ("inflacion", "get_inflacion_mensual"),
            ("riesgo_pais", "get_riesgo_pais"),
            ("bandas", "get_bandas_cambiarias"),
        )
        timeout_seconds = settings.legacy_market_fetch_timeout_seconds
        timeout = ClientTimeout(total=timeout_seconds)

        async with ClientSession(timeout=timeout) as session:
            async def fetch_one(name: str, func_name: str) -> tuple[str, Any, str | None]:
                func = getattr(module, func_name, None)
                if func is None:
                    return name, None, f"Funcion legacy {func_name} no disponible"
                try:
                    value = await asyncio.wait_for(func(session), timeout=timeout_seconds)
                    return name, value, None
                except TimeoutError:
                    return name, None, f"{func_name} excedio {timeout_seconds:g}s"
                except Exception as exc:
                    return name, None, f"{func_name} fallo: {exc}"

            results = await asyncio.gather(
                *(fetch_one(name, func_name) for name, func_name in calls)
            )

        payload: dict[str, Any] = {}
        warnings: list[str] = []
        for name, value, warning in results:
            payload[name] = value
            if warning:
                warnings.append(warning)
        status = "partial" if warnings else "ok"
        return LegacyResult(status=status, data=payload, warnings=warnings, freshness="current")

    async def fetch_screener_metrics(self, kind: str) -> LegacyResult:
        settings = get_settings()
        if not settings.legacy_live_fetch_enabled:
            return LegacyResult(
                status="not_available",
                data=None,
                warnings=[
                    "Metricas live desactivadas. Activar LEGACY_LIVE_FETCH_ENABLED=1 para consultar fuentes reales.",
                ],
            )
        try:
            module = self._safe_import()
        except Exception as exc:
            return LegacyResult(
                status="not_available",
                data=None,
                warnings=[f"No se pudo importar legacy para screener: {exc}"],
            )

        attr = "ACCIONES_BA" if kind == "acciones" else "CEDEARS_BA"
        symbols = getattr(module, attr, None)
        metrics_func = getattr(module, "metrics_for_symbols_cached", None)
        if not isinstance(symbols, list) or metrics_func is None:
            return LegacyResult(
                status="not_available",
                data=None,
                warnings=["Catalogo o funcion de metricas legacy no disponible"],
            )

        timeout_seconds = settings.legacy_screener_fetch_timeout_seconds
        timeout = ClientTimeout(total=min(timeout_seconds, 30.0))
        try:
            async with ClientSession(timeout=timeout) as session:
                metrics, last_timestamp = await asyncio.wait_for(
                    metrics_func(session, list(symbols)),
                    timeout=timeout_seconds,
                )
        except TimeoutError:
            return LegacyResult(
                status="not_available",
                data=None,
                warnings=[f"Screener legacy excedio {timeout_seconds:g}s"],
            )
        except Exception as exc:
            return LegacyResult(
                status="not_available",
                data=None,
                warnings=[f"Screener legacy fallo: {exc}"],
            )

        if not isinstance(metrics, dict):
            return LegacyResult(
                status="not_available",
                data=None,
                warnings=["Screener legacy devolvio un formato no reconocido"],
            )
        return LegacyResult(
            status="ok",
            data={"metrics": metrics, "last_timestamp": last_timestamp},
            warnings=[],
            source="legacy.metrics_for_symbols_cached",
            freshness="current",
        )

    async def fetch_portfolio_snapshot(self, user_id: str) -> LegacyResult:
        connection = self.inspect_portfolio_state(user_id)
        if connection.status != "ok":
            return connection
        try:
            module = self._safe_import()
        except Exception as exc:
            return LegacyResult(
                status="not_available",
                data=None,
                warnings=[f"No se pudo importar legacy para portfolio: {exc}"],
            )

        portfolio_copy = connection.data["portfolio"]
        items = portfolio_copy.get("items")
        if not isinstance(items, list):
            items = []
            portfolio_copy["items"] = items
        symbols = sorted(
            {
                str(item.get("simbolo"))
                for item in items
                if isinstance(item, dict) and item.get("simbolo")
            }
        )
        base_config = portfolio_copy.get("base")
        if isinstance(base_config, dict):
            base_currency = str(base_config.get("moneda") or "ARS").upper()
            benchmark = str(base_config.get("benchmark") or "").lower()
            if benchmark == "merval":
                symbols.append("^MERV")
            elif benchmark in {"usd", "dolar", "dólar"}:
                symbols.append("USDARS=X" if base_currency == "ARS" else "ARSUSD=X")
            symbols = sorted(set(symbols))
        if not symbols:
            return LegacyResult(
                status="ok",
                data={"portfolio": portfolio_copy, "metrics": {}, "last_timestamp": None},
                warnings=["Portfolio legacy sin instrumentos"],
                source="legacy.PF.read_only",
                freshness="unknown",
            )

        settings = get_settings()
        if not settings.legacy_live_fetch_enabled:
            return LegacyResult(
                status="partial",
                data={"portfolio": portfolio_copy, "metrics": {}, "last_timestamp": None},
                warnings=["Precios live desactivados para portfolio"],
                source="legacy.PF.read_only",
                freshness="unknown",
            )

        metrics_func = getattr(module, "metrics_for_symbols_cached", None)
        if metrics_func is None:
            return LegacyResult(
                status="partial",
                data={"portfolio": portfolio_copy, "metrics": {}, "last_timestamp": None},
                warnings=["Funcion de precios legacy no disponible"],
                source="legacy.PF.read_only",
                freshness="unknown",
            )

        timeout_seconds = settings.legacy_screener_fetch_timeout_seconds
        timeout = ClientTimeout(total=min(timeout_seconds, 30.0))
        try:
            async with ClientSession(timeout=timeout) as session:
                metrics, last_timestamp = await asyncio.wait_for(
                    metrics_func(session, symbols),
                    timeout=timeout_seconds,
                )
        except TimeoutError:
            return LegacyResult(
                status="partial",
                data={"portfolio": portfolio_copy, "metrics": {}, "last_timestamp": None},
                warnings=[f"Consulta de portfolio excedio {timeout_seconds:g}s"],
                source="legacy.PF.read_only",
                freshness="unknown",
            )
        except Exception as exc:
            return LegacyResult(
                status="partial",
                data={"portfolio": portfolio_copy, "metrics": {}, "last_timestamp": None},
                warnings=[f"Consulta de precios de portfolio fallo: {exc}"],
                source="legacy.PF.read_only",
                freshness="unknown",
            )

        return LegacyResult(
            status="ok",
            data={
                "portfolio": portfolio_copy,
                "metrics": metrics if isinstance(metrics, dict) else {},
                "last_timestamp": last_timestamp,
            },
            warnings=[],
            source="legacy.PF.read_only",
            freshness="current",
        )

    def inspect_portfolio_state(self, user_id: str) -> LegacyResult:
        """Inspect one user's PF without loading, saving or mutating legacy state."""
        try:
            chat_id = int(user_id)
        except (TypeError, ValueError):
            return LegacyResult(
                status="not_available",
                data=None,
                warnings=["Portfolio legacy requiere una identidad Telegram numerica"],
            )
        if chat_id <= 0:
            return LegacyResult(
                status="not_available",
                data=None,
                warnings=["Identidad Telegram inválida para portfolio"],
            )
        try:
            module = self._safe_import()
        except Exception as exc:
            return LegacyResult(
                status="not_available",
                data=None,
                warnings=[f"No se pudo importar legacy para portfolio: {exc}"],
            )
        portfolios = getattr(module, "PF", None)
        if not isinstance(portfolios, dict):
            return LegacyResult(
                status="not_available",
                data=None,
                warnings=["Estado PF legacy no disponible"],
            )
        portfolio = portfolios.get(chat_id)
        if not isinstance(portfolio, dict):
            return LegacyResult(
                status="not_available",
                data=None,
                warnings=["El usuario no tiene un portfolio legacy cargado"],
                source="legacy.PF.read_only",
            )
        portfolio_copy = copy.deepcopy(portfolio)
        items = portfolio_copy.get("items")
        if not isinstance(items, list):
            items = []
            portfolio_copy["items"] = items
        return LegacyResult(
            status="ok",
            data={"portfolio": portfolio_copy, "item_count": len(items)},
            warnings=[],
            source="legacy.PF.read_only",
            freshness="current",
        )

    def inspect_alert_state(self, user_id: str) -> LegacyResult:
        """Copy one user's ALERTS and SUBS without touching legacy persistence."""
        try:
            chat_id = int(user_id)
        except (TypeError, ValueError):
            return LegacyResult(
                status="not_available",
                data=None,
                warnings=["Alertas legacy requieren una identidad Telegram numerica"],
            )
        if chat_id <= 0:
            return LegacyResult(
                status="not_available",
                data=None,
                warnings=["Identidad Telegram inválida para alertas"],
            )
        try:
            module = self._safe_import()
        except Exception as exc:
            return LegacyResult(
                status="not_available",
                data=None,
                warnings=[f"No se pudo importar legacy para alertas: {exc}"],
            )

        alerts = getattr(module, "ALERTS", None)
        subscriptions = getattr(module, "SUBS", None)
        if not isinstance(alerts, dict) or not isinstance(subscriptions, dict):
            return LegacyResult(
                status="not_available",
                data=None,
                warnings=["Estado ALERTS/SUBS legacy no disponible"],
            )

        rules = alerts.get(chat_id, []) or []
        subscription = subscriptions.get(chat_id, {}) or {}
        if not isinstance(rules, list):
            return LegacyResult(
                status="not_available",
                data=None,
                warnings=["Las alertas legacy del usuario tienen un formato no reconocido"],
                source="legacy.ALERTS_SUBS.read_only",
            )
        if not isinstance(subscription, dict):
            subscription = {}

        paused_chats = getattr(module, "ALERTS_PAUSED", set())
        silent_until = getattr(module, "ALERTS_SILENT_UNTIL", {})
        global_pause = isinstance(paused_chats, set) and chat_id in paused_chats
        global_pause_until = (
            silent_until.get(chat_id) if isinstance(silent_until, dict) else None
        )

        return LegacyResult(
            status="ok",
            data={
                "rules": copy.deepcopy(rules),
                "subscription": copy.deepcopy(subscription),
                "global_pause": global_pause,
                "global_pause_until": copy.deepcopy(global_pause_until),
                "defaults": {
                    "rearm_cooldown_seconds": getattr(
                        module, "ALERT_REARM_COOLDOWN_SECS", None
                    ),
                    "composite_cooldown_seconds": getattr(
                        module, "ALERT_COMPOSITE_COOLDOWN_DEFAULT_SECS", None
                    ),
                    "price_tolerance_pct": getattr(
                        module, "ALERT_PRICE_TOLERANCE_PCT", None
                    ),
                },
            },
            warnings=[],
            source="legacy.ALERTS_SUBS.read_only",
            freshness="current",
        )


legacy_adapter = LegacyAdapter()
