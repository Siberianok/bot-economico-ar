"""Punto de entrada del bot con logging JSON estructurado y métricas básicas."""

import asyncio
import json
import logging
import logging.config
from datetime import datetime, timezone


class JsonLogFormatter(logging.Formatter):
    """Formateador de logs en JSON con campos estándar."""

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "module": record.module,
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key.startswith("_") or key in payload:
                continue
            if key in {"args", "msg", "exc_info", "exc_text", "stack_info", "stacklevel"}:
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except Exception:
                payload[key] = repr(value)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging() -> None:
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {
                    "()": JsonLogFormatter,
                }
            },
            "handlers": {
                "stdout": {
                    "class": "logging.StreamHandler",
                    "formatter": "json",
                    "level": "INFO",
                }
            },
            "root": {
                "handlers": ["stdout"],
                "level": "INFO",
            },
        }
    )


async def _run_bot() -> None:
    from bot_econ_full_plus_rank_alerts import main as bot_main

    await bot_main()


def main() -> None:
    configure_logging()
    asyncio.run(_run_bot())


if __name__ == "__main__":
    main()
