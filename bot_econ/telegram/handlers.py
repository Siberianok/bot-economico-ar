from __future__ import annotations

import logging

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from ..config import AppConfig
from ..formatters import fmt_money_ars, fmt_money_usd
from ..services.metrics_pipeline import MetricsPipeline
from ..services.state_service import StateService

log = logging.getLogger(__name__)


class TelegramBot:
    def __init__(self, config: AppConfig, pipeline: MetricsPipeline, state: StateService) -> None:
        self._config = config
        self._pipeline = pipeline
        self._state = state

    def build_application(self) -> Application:
        app = (
            Application.builder()
            .token(self._config.telegram_token)
            .post_init(self._on_startup)
            .post_shutdown(self._on_shutdown)
            .build()
        )
        app.add_handler(CommandHandler("start", self.start))
        app.add_handler(CommandHandler("resumen", self.summary))
        app.add_handler(CommandHandler("healthcheck", self.healthcheck))
        if app.job_queue:
            app.job_queue.run_repeating(self._prewarm, interval=300, first=5)
        return app

    async def _on_startup(self, app: Application) -> None:  # pragma: no cover - telegram lifecycle
        await self._state.migrate_from_json()

    async def _on_shutdown(self, app: Application) -> None:  # pragma: no cover - telegram lifecycle
        from ..data_sources.http import get_http_client

        client = await get_http_client()
        await client.close()

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.effective_chat is None:
            return
        await update.effective_chat.send_message(
            "👋 ¡Hola! Probá /resumen para ver las cotizaciones del día."
        )

    async def summary(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.effective_chat is None:
            return

        try:
            summary = await self._pipeline.fetch_summary()
        except Exception:  # pragma: no cover - defensive logging
            log.exception("Error generando resumen")
            await update.effective_chat.send_message("No pude obtener los datos en este momento.")
            return

        lines = ["📊 *Resumen de mercado*"]

        if summary.oficial_quotes:
            lines.append("\n💱 *Dólar oficial/blue*")
            for quote in summary.oficial_quotes:
                buy = fmt_money_ars(quote.buy)
                sell = fmt_money_ars(quote.sell)
                lines.append(f"• {quote.name.title()}: compra {buy}, venta {sell}")

        if summary.dolar_quotes:
            lines.append("\n🪙 *Cotizaciones digitales*")
            for quote in summary.dolar_quotes[:6]:
                buy = fmt_money_usd(quote.buy)
                sell = fmt_money_usd(quote.sell)
                lines.append(f"• {quote.name.title()}: compra {buy}, venta {sell}")

        if summary.reserves:
            total = fmt_money_usd(summary.reserves.total)
            variation = fmt_money_usd(summary.reserves.variation)
            date = summary.reserves.date.strftime("%d/%m/%Y") if summary.reserves.date else "—"
            lines.append("\n🏦 *Reservas BCRA*")
            lines.append(
                "Total: "
                f"{total} (variación diaria {variation})"
                f"\nÚltima actualización: {date}"
            )

        text = "\n".join(lines)
        await update.effective_chat.send_message(text, parse_mode="Markdown")

    async def healthcheck(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.effective_chat is None:
            return
        from .. import storage_adapter

        status = await storage_adapter.redis_ping()
        await update.effective_chat.send_message(status)

    async def _prewarm(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        try:
            await self._pipeline.fetch_summary()
        except Exception:  # pragma: no cover - background job resilience
            log.exception("Prewarm failed")
