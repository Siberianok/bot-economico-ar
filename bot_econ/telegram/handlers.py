# bot_econ/telegram/handlers.py
from __future__ import annotations

import logging
from typing import Any, List

from telegram import BotCommand, Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackContext,
    CommandHandler,
)

from ..config import AppConfig
from ..services.metrics_pipeline import MetricsPipeline
from ..services.state_service import StateService

# Import tolerante: si no existe en http.py, definimos un no-op
try:
    from ..data_sources.http import close_http_client  # type: ignore
except Exception:
    async def close_http_client() -> None:  # type: ignore
        return

log = logging.getLogger(__name__)


class TelegramBot:
    """
    Arma la Application de python-telegram-bot (v20+), registra comandos y
    configura jobs (prewarm). La ejecución (polling/webhook) la hace main.py.
    """

    def __init__(self, config: AppConfig, pipeline: MetricsPipeline, state: StateService) -> None:
        self._config = config
        self._pipeline = pipeline
        self._state = state

    # ========= Handlers básicos =========

    async def _cmd_start(self, update: Update, context: CallbackContext) -> None:
        txt = (
            "Hola 👋 Soy el Bot Económico AR.\n\n"
            "Comandos:\n"
            "• /start – ayuda\n"
            "• /ping – prueba rápida\n"
            "• /resumen – resumen económico del día\n"
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
                # Mostramos algunos tipos si están disponibles
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

    # ========= Jobs =========

    async def _prewarm(self, context: CallbackContext) -> None:
        """
        Llamado periódicamente para calentar cachés de fuentes.
        No envía mensajes; solo dispara fetch en paralelo.
        """
        try:
            await self._pipeline.fetch_summary()
            log.info("Prewarm OK")
        except Exception as e:
            # No levantamos excepción: queremos que el job siga vivo
            log.error("Prewarm failed", exc_info=e)

    # ========= Hooks de ciclo de vida =========

    async def _on_startup(self, app: Application) -> None:
        # Seteamos comandos del bot (lista en la UI de Telegram)
        cmds = [
            BotCommand("start", "Ayuda"),
            BotCommand("ping", "Prueba rápida"),
            BotCommand("resumen", "Resumen económico del día"),
        ]
        try:
            await app.bot.set_my_commands(cmds)
        except Exception as e:
            log.warning("set_my_commands falló: %s", e)

        # Job de prewarm cada 5 minutos
        app.job_queue.run_repeating(self._prewarm, interval=300, first=5, name="_prewarm")
        log.info("Startup hook completado")

    async def _on_shutdown(self, app: Application) -> None:
        # Cerramos HTTP client si existe
        try:
            await close_http_client()
        except Exception:
            pass
        log.info("Shutdown hook completado")

    # ========= Wiring =========

    def build_application(self) -> Application:
        """
        Construye y devuelve la Application ya configurada con handlers y hooks.
        La ejecución (polling/webhook) la maneja main.py.
        """
        app = (
            ApplicationBuilder()
            .token(self._config.telegram_token)
            .concurrent_updates(True)
            .build()
        )

        # Handlers de comandos básicos
        app.add_handler(CommandHandler("start", self._cmd_start))
        app.add_handler(CommandHandler("ping", self._cmd_ping))
        app.add_handler(CommandHandler("resumen", self._cmd_resumen))

        # Hooks
        app.post_init = self._on_startup  # llamado al iniciar
        app.post_shutdown = self._on_shutdown  # llamado al apagar

        return app
