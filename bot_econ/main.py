# bot_econ/main.py
from __future__ import annotations

import logging
import os

from .config import AppConfig, configure_logging
from .services.metrics_pipeline import MetricsPipeline
from .services.state_service import StateService
from .storage_adapter import configure_storage
from .telegram.handlers import TelegramBot


def _run_webhook() -> None:
    """
    Ejecuta el bot en modo Webhook (recomendado en Render).
    Usa:
      - BASE_URL
      - WEBHOOK_SECRET
      - PORT
    """
    configure_logging()
    config = AppConfig.load()

    if not config.telegram_token:
        raise RuntimeError("TELEGRAM_TOKEN es obligatorio para el bot.")
    if not config.base_url:
        raise RuntimeError("BASE_URL es obligatorio (Render lo inyecta).")
    if not config.webhook_secret:
        raise RuntimeError("WEBHOOK_SECRET es obligatorio.")
    if not config.port:
        raise RuntimeError("PORT es obligatorio.")

    if config.redis_url:
        configure_storage(config)

    state = StateService(config)
    pipeline = MetricsPipeline()
    bot = TelegramBot(config, pipeline, state)
    app = bot.build_application()

    webhook_path = f"/{config.webhook_secret}"
    logging.getLogger(__name__).info(
        "Iniciando Webhook | listen=%s | port=%s | path=%s | url=%s",
        "0.0.0.0",
        config.port,
        webhook_path,
        f"{config.base_url}{webhook_path}",
    )

    # run_webhook se ocupa de registrar setWebhook
    app.run_webhook(
        listen="0.0.0.0",
        port=int(config.port),
        url=f"{config.base_url}{webhook_path}",
        webhook_path=webhook_path,
        drop_pending_updates=True,   # limpia cola vieja
        allowed_updates=None,        # aceptar todo
    )


def main() -> None:
    try:
        _run_webhook()
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Bot detenido por el usuario")


if __name__ == "__main__":
    main()
