from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import asdict

from .. import storage_adapter
from ..config import AppConfig
from . import models

log = logging.getLogger(__name__)


class StateService:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._migrated = False

    async def migrate_from_json(self) -> None:
        if self._migrated:
            return
        path = self._config.state_path
        if not path or not os.path.exists(path):
            self._migrated = True
            return
        try:
            payload = await asyncio.to_thread(_read_legacy_state, path)
        except Exception as exc:
            log.warning("No se pudo migrar estado local", exc_info=exc)
            self._migrated = True
            return

        alerts: dict[str, list[dict]] = payload.get("alerts", {}) if payload else {}
        for chat_id, rules in alerts.items():
            for rule in rules:
                try:
                    await storage_adapter.alerts_add(int(chat_id), rule)
                except Exception:
                    log.exception("Error migrando alerta", extra={"chat_id": chat_id})

        subs: dict[str, dict] = payload.get("subs", {}) if payload else {}
        for chat_id, sub in subs.items():
            try:
                await storage_adapter.subs_set(
                    int(chat_id),
                    sub.get("hhmm", "13:00"),
                    sub.get("tz", self._config.timezone),
                    bool(sub.get("paused", False)),
                )
            except Exception:
                log.exception("Error migrando subscripcion", extra={"chat_id": chat_id})

        pf: dict[str, dict] = payload.get("pf", {}) if payload else {}
        for chat_id, raw in pf.items():
            try:
                await storage_adapter.pf_set(int(chat_id), raw)
            except Exception:
                log.exception("Error migrando portafolio", extra={"chat_id": chat_id})

        self._migrated = True

    async def add_alert(self, chat_id: int, rule: models.AlertRule) -> str:
        payload = asdict(rule)
        payload["created_at"] = int(rule.created_at.timestamp())
        return await storage_adapter.alerts_add(chat_id, payload)

    async def list_alerts(self, chat_id: int) -> list[dict]:
        return await storage_adapter.alerts_list(chat_id)

    async def delete_alert(self, chat_id: int, index: int) -> bool:
        return await storage_adapter.alerts_del_by_index(chat_id, index)

    async def set_portfolio(self, portfolio: models.Portfolio) -> None:
        payload = {
            "positions": [asdict(pos) for pos in portfolio.positions],
            "updated_at": portfolio.updated_at.isoformat(),
        }
        await storage_adapter.pf_set(portfolio.chat_id, payload)

    async def get_portfolio(self, chat_id: int) -> dict | None:
        return await storage_adapter.pf_get(chat_id)

    async def delete_portfolio(self, chat_id: int) -> None:
        await storage_adapter.pf_del(chat_id)

    async def set_subscription(self, sub: models.Subscription) -> None:
        await storage_adapter.subs_set(sub.chat_id, sub.hhmm, sub.timezone, sub.paused)

    async def get_subscription(self, chat_id: int) -> dict | None:
        return await storage_adapter.subs_get(chat_id)

    async def delete_subscription(self, chat_id: int) -> None:
        await storage_adapter.subs_del(chat_id)


def _read_legacy_state(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return None
