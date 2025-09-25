from __future__ import annotations

import logging

from .config import AppConfig, configure_logging
from .services.metrics_pipeline import MetricsPipeline
from .services.state_service import StateService
from .storage_adapter import configure_storage
from .telegram.handlers import TelegramBot


def _run_polling() -> None:
    configure_logging()
    config = AppConfig.load()
    if not config.redis_url:
        raise RuntimeError("REDIS_URL es obligatorio para el bot.")

    configure_storage(config)

    state = StateService(config)
    pipeline = MetricsPipeline()
    bot = TelegramBot(config, pipeline, state)
    app = bot.build_application()
    app.run_polling()


def main() -> None:
    try:
        _run_polling()
    except KeyboardInterrupt:  # pragma: no cover - CLI exit
        logging.getLogger(__name__).info("Bot detenido por el usuario")


if __name__ == "__main__":
    main()
