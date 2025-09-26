# bot_econ/telegram/handlers.py
from __future__ import annotations

import json
import logging
from typing import Any, List

from telegram import BotCommand, Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackContext,
    CommandHandler,
    MessageHandler,
    filters,
)

from ..config import AppConfig
from ..services.metrics_pipeline import MetricsPipeline
from ..services.state_service import StateService

# Import tolerante: si no existe, hacemos no-op
try:
    from ..data_sources.http import close_http_client  # type: ignore
except Exception:
    async def close_http_client() -> None:  # type: ignore
        return

log = logging.getLogger(__name__)


class TelegramBot:
    """
    Arma la Application de python-telegram-bot (v20+), registra comandos y jobs.
    La ejecución webhook/polling la hace main.py.
    """

    def __init__(self, config: AppConfig, pipeline: MetricsPipeline, state: StateService) -> None:
        self._config = config
        self._pipeline = pipeline
        self._state = state

    # ========= Handlers =========

    async def _cmd_start(self, update: Update, context: CallbackContext) -> None:
        txt = (
            "Hola 👋 Soy el Bot Económico AR.\n\n"
            "Comandos:\n"
            "• /start – ayuda\n"
            "• /ping – prueba rápida\n"
            "• /resumen – resumen económico del día\n"
            "• /whinfo – diagnóstico del webhook\n"
        )
        await update.effective_message.reply_text(txt)

    async def _cmd_ping(self, update: Update, context: CallbackContext) -> None:
        await update.effective_message.reply_text("pong 🏓")

    async def _cmd_resumen(self, update: Update, context: CallbackContext) -> None:
        try:
            data = await self._pipeline.fetch_summary()
            lines: List[str] = ["<b>Resumen económico</b>"]
            if data.dolar:
                d = data.dolar

                def fmt(v: Any) -> str:
                    try:
                        return f"${float(v):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                    except Exception:
                        return str(v)

                parts = []
                if d.oficial and d.oficial.venta is not None:
                    parts.append(f"Oficial: {fmt(d.oficial.venta)}")
                if d.blue and d.blue.venta is not None:
                    parts.append(f"Blue: {fmt(d.blue.venta)}")
                if d.mep and d.mep.venta is not None:
                    parts.append(f"MEP: {fmt(d.mep.venta)}")
                if d.ccl and d.ccl.venta is not None:
                    parts.append(f"CCL: {fmt(d.ccl.venta)}")
                if parts:
                    lines.append("💵 " + " | ".join(parts))

            if data.riesgo_pais is not None:
                lines.append(f"📈 Riesgo País: <b>{int(data.riesgo_pais)} pb</b>")
            if data.inflacion_mensual is not None:
                lines.append(f"📉 Inflación (mensual): <b>{str(round(data.inflacion_mensual,1)).replace('.',',')}%</b>")
            if data.reservas_musd is not None:
                try:
                    val = f"{float(data.reservas_musd):,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
                except Exception:
                    val = str(data.reservas_musd)
                lines.append(f"🏦 Reservas: <b>{val} MUS$</b>")

            await update.effective_message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)
        except Exception as e:
            log.warning("cmd_resumen error: %s", e)
            await update.effective_message.reply_text("No pude armar el resumen ahora 😕. Probá de nuevo en un rato.")

    async def _cmd_whinfo(self, update: Update, context: CallbackContext) -> None:
        """Diagnóstico del webhook directamente desde el bot."""
        try:
            info = await context.bot.get_webhook_info()
            payload = {
                "url": info.url,
                "has_custom_certificate": info.has_custom_certificate,
                "pending_update_count": info.pending_update_count,
                "ip_address": info.ip_address,
                "last_error_date": info.last_error_date,
                "last_error_message": info.last_error_message,
                "max_connections": info.max_connections,
                "allowed_updates": info.allowed_updates,
            }
            txt = "<b>Webhook info</b>\n<pre>" + _pretty(payload) + "</pre>"
            await update.effective_message.reply_text(txt, parse_mode=ParseMode.HTML)
        except Exception as e:
            log.warning("whinfo error: %s", e)
            await update.effective_message.reply_text("No pude leer WebhookInfo.")

    async def _prewarm(self, context: CallbackContext) -> None:
        try:
            await self._pipeline.fetch_summary()
            log.info("Prewarm OK")
        except Exception as e:
            log.error("Prewarm failed", exc_info=e)

    # Catch-all: si llega cualquier cosa, contestamos (prueba rápida)
    async def _fallback_echo(self, update: Update, context: CallbackContext) -> None:
        try:
            text = update.effective_message.text or ""
            if text.startswith("/"):
                await update.effective_message.reply_text("Comando no reconocido. Probá /start.")
            else:
                await update.effective_message.reply_text("Te leo perfecto. Probá /resumen o /whinfo.")
        except Exception:
            pass

    # ========= Hooks =========

    async def _on_startup(self, app: Application) -> None:
        cmds = [
            BotCommand("start", "Ayuda"),
            BotCommand("ping", "Prueba rápida"),
            BotCommand("resumen", "Resumen económico del día"),
            BotCommand("whinfo", "Diagnóstico del webhook"),
        ]
        try:
            await app.bot.set_my_commands(cmds)
        except Exception as e:
            log.warning("set_my_commands falló: %s", e)

        app.job_queue.run_repeating(self._prewarm, interval=300, first=5, name="_prewarm")
        log.info("Startup hook completado")

    async def _on_shutdown(self, app: Application) -> None:
        try:
            await close_http_client()
        except Exception:
            pass
        log.info("Shutdown hook completado")

    # ========= Wiring =========

    def build_application(self) -> Application:
        app = (
            ApplicationBuilder()
            .token(self._config.telegram_token)
            .concurrent_updates(True)
            .build()
        )

        app.add_handler(CommandHandler("start", self._cmd_start))
        app.add_handler(CommandHandler("ping", self._cmd_ping))
        app.add_handler(CommandHandler("resumen", self._cmd_resumen))
        app.add_handler(CommandHandler("whinfo", self._cmd_whinfo))

        # catch-all al final
        app.add_handler(MessageHandler(filters.ALL, self._fallback_echo))

        app.post_init = self._on_startup
        app.post_shutdown = self._on_shutdown

        return app


def _pretty(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    except Exception:
        return str(obj)
