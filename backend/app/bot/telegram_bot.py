"""Lazy Telegram runtime hosted by the FastAPI process."""

from __future__ import annotations

import asyncio
import hmac
import importlib
import logging
import re
from typing import Any

from backend.app.config import Settings, get_settings


logger = logging.getLogger(__name__)


class TelegramRuntime:
    """Own the legacy PTB application without starting its aiohttp server."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.application: Any = None
        self.legacy: Any = None
        self.tasks: list[asyncio.Task[Any]] = []
        self.state = "disabled" if not self.settings.telegram_runtime_enabled else "stopped"
        self.error: str | None = None
        self.webhook_url: str | None = None

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.settings.telegram_runtime_enabled,
            "state": self.state,
            "token_configured": self.settings.telegram_token_configured,
            "webhook_configured": bool(self.settings.webhook_secret),
            "keepalive_enabled": self.settings.keepalive_enabled,
            "error": self.error,
        }

    def matches_secret(self, candidate: str) -> bool:
        expected = self.settings.webhook_secret
        return bool(expected) and hmac.compare_digest(candidate, expected)

    async def start(self) -> None:
        if not self.settings.telegram_runtime_enabled:
            self.state = "disabled"
            return
        if self.application is not None:
            return
        if not self.settings.telegram_token_configured:
            self.state = "configuration_error"
            self.error = "TELEGRAM_TOKEN no configurado"
            logger.error(self.error)
            return
        if not self.settings.webhook_secret:
            self.state = "configuration_error"
            self.error = "WEBHOOK_SECRET no configurado"
            logger.error(self.error)
            return
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,256}", self.settings.webhook_secret):
            self.state = "configuration_error"
            self.error = "WEBHOOK_SECRET contiene caracteres inválidos"
            logger.error(self.error)
            return

        self.state = "starting"
        try:
            legacy = importlib.import_module("bot_econ_full_plus_rank_alerts")
            await legacy.load_state()
            application = legacy.build_application()
            self.legacy = legacy
            self.application = application
            legacy._schedule_all_subs(application)
            legacy._schedule_projection_calibration(application)
            legacy._schedule_projection_performance(application)

            await application.initialize()
            await application.start()
            await application.bot.set_my_commands(legacy.BOT_COMMANDS)

            self.webhook_url = f"{self.settings.public_base_url}/telegram/webhook"
            await application.bot.set_webhook(
                url=self.webhook_url,
                drop_pending_updates=False,
                secret_token=self.settings.webhook_secret,
            )

            self.tasks = [
                asyncio.create_task(legacy.alerts_loop(application), name="telegram-alerts"),
                asyncio.create_task(legacy.instrument_catalog_loop(), name="telegram-catalog"),
            ]
            if self.settings.keepalive_enabled:
                self.tasks.append(
                    asyncio.create_task(legacy.keepalive_loop(), name="render-keepalive")
                )
            self.state = "running"
            self.error = None
            logger.info("Telegram runtime iniciado con webhook compartido")
        except Exception as exc:
            self.state = "error"
            self.error = f"{exc.__class__.__name__}: {exc}"
            logger.exception("No se pudo iniciar Telegram runtime")
            await self.stop()

    async def enqueue_update(self, payload: dict[str, Any]) -> None:
        if self.state != "running" or self.application is None:
            raise RuntimeError("Telegram runtime no disponible")
        from telegram import Update

        update = Update.de_json(payload, self.application.bot)
        await self.application.update_queue.put(update)

    async def stop(self) -> None:
        previous_state = self.state
        application = self.application
        legacy = self.legacy
        tasks = list(self.tasks)
        self.tasks = []

        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        if application is not None:
            try:
                await application.stop()
            except RuntimeError:
                logger.debug("Telegram Application no estaba iniciada o ya se detuvo")
            except Exception:
                logger.exception("No se pudo detener Telegram Application")

        if legacy is not None:
            try:
                await legacy.save_state()
            except Exception:
                logger.exception("No se pudo persistir el estado legacy al apagar")

        if application is not None:
            try:
                await application.shutdown()
            except RuntimeError:
                logger.debug("Telegram Application no estaba inicializada o ya se cerró")
            except Exception:
                logger.exception("No se pudo cerrar Telegram Application")

        self.application = None
        self.legacy = None
        self.webhook_url = None
        if previous_state in {"error", "configuration_error", "disabled"}:
            self.state = previous_state
        else:
            self.state = "stopped"


telegram_runtime = TelegramRuntime()
