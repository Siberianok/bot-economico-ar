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
    Inicia el bot con servidor HTTP interno y registra el Webhook en Telegram.
    Requiere:
      - TELEGRAM_TOKEN (lo usa TelegramBot internamente)
      - BASE_URL (Render la completa desde render.yaml -> fromService:url)
      - WEBHOOK_SECRET (valor sin '/'; define la ruta del webhook)
      - PORT (Render expone este puerto; por defecto 10000)
    """
    configure_logging()
    config = AppConfig.load()

    # Si tenés Redis, lo configuramos. Si no, seguimos con storage en archivo/memoria.
    # (Antes era obligatorio; ahora solo avisamos para no bloquear el arranque.)
    if getattr(config, "redis_url", None):
        configure_storage(config)
    else:
        logging.getLogger(__name__).warning(
            "REDIS_URL no está configurado. Uso almacenamiento local (STATE_PATH)."
        )

    state = StateService(config)
    pipeline = MetricsPipeline()
    bot = TelegramBot(config, pipeline, state)
    app = bot.build_application()  # acá se registran handlers y jobs propios

    # === Variables para Webhook ===
    base_url = (os.getenv("BASE_URL") or os.getenv("RENDER_EXTERNAL_URL") or "").rstrip("/")
    if not base_url:
        raise RuntimeError(
            "Falta BASE_URL (Render la completa automáticamente en render.yaml)."
        )
    webhook_secret = os.getenv("WEBHOOK_SECRET", "tgwebhook-secret").strip().strip("/")
    port = int(os.getenv("PORT", "10000"))

    webhook_path = f"/{webhook_secret}"
    webhook_url = f"{base_url}{webhook_path}"

    log = logging.getLogger(__name__)
    log.info(
        "Iniciando Webhook | listen=0.0.0.0 | port=%s | path=%s | url=%s",
        port,
        webhook_path,
        webhook_url,
    )

    # Levanta el servidor HTTP interno y registra el webhook en Telegram.
    app.run_webhook(
        listen="0.0.0.0",
        port=port,
        url_path=webhook_secret,   # la ruta interna expuesta (sin slash inicial)
        webhook_url=webhook_url,   # URL pública completa para Telegram
        drop_pending_updates=True, # evita duplicados al reiniciar
    )


def main() -> None:
    try:
        _run_webhook()
    except KeyboardInterrupt:  # pragma: no cover - CLI exit
        logging.getLogger(__name__).info("Bot detenido por el usuario")


if __name__ == "__main__":
    main()
