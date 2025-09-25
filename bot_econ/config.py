from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass


@dataclass(slots=True)
class AppConfig:
    telegram_token: str
    webhook_secret: str
    port: int
    base_url: str
    redis_url: str
    redis_prefix: str
    state_path: str
    timezone: str = "America/Argentina/Buenos_Aires"

    @classmethod
    def load(cls) -> AppConfig:
        token = (os.getenv("TELEGRAM_TOKEN") or os.getenv("BOT_TOKEN") or "").strip()
        if not token:
            raise RuntimeError("TELEGRAM_TOKEN/BOT_TOKEN no configurado.")

        base_url = os.getenv(
            "BASE_URL", os.getenv("RENDER_EXTERNAL_URL", "http://localhost")
        ).rstrip("/")
        webhook_secret = os.getenv("WEBHOOK_SECRET", "tgwebhook").strip().strip("/")
        redis_url = os.getenv("REDIS_URL", "").strip()
        redis_prefix = os.getenv("REDIS_PREFIX", "bot-ar").strip() or "bot-ar"
        state_path = os.getenv("STATE_PATH", "state.json")

        return cls(
            telegram_token=token,
            webhook_secret=webhook_secret,
            port=int(os.getenv("PORT", "10000")),
            base_url=base_url,
            redis_url=redis_url,
            redis_prefix=redis_prefix,
            state_path=state_path,
        )

    @property
    def webhook_path(self) -> str:
        return f"/{self.webhook_secret}"

    @property
    def webhook_url(self) -> str:
        return f"{self.base_url}{self.webhook_path}"


class JsonLogFormatter(logging.Formatter):
    """Structured JSON formatter with sane defaults."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        payload: dict[str, object] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        for key in ("extra", "user_id", "chat_id"):
            value: object | None = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonLogFormatter())
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)
