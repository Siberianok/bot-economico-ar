"""Configuración centralizada del bot."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class UpstashConfig:
    """Configuración para Upstash."""

    rest_url: str
    rest_token: str
    redis_url: str
    state_key: str = "bot-econ-state"

    @property
    def use_rest(self) -> bool:
        return bool(self.rest_url and self.rest_token)

    @property
    def use_redis(self) -> bool:
        return bool(self.redis_url)


@dataclass(frozen=True)
class BotConfig:
    """Configuración principal del bot."""

    telegram_token: str
    webhook_secret: str
    port: int
    base_url: str
    state_path: Path
    upstash: UpstashConfig


@dataclass(frozen=True)
class BotDependencies:
    """Dependencias derivadas de la configuración del bot."""

    telegram_token: str
    webhook_secret: str
    webhook_path: str
    webhook_url: str
    port: int
    base_url: str
    state_path: Path
    upstash: UpstashConfig


def _get_first(keys: Iterable[str], default: str = "") -> str:
    for key in keys:
        val = os.getenv(key)
        if val is not None:
            return val.strip()
    return default


def _get_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None:
        return default
    raw = raw.strip()
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"Valor inválido para {key}: {raw}") from exc


def load_config() -> BotConfig:
    telegram_token = _get_first(["TELEGRAM_TOKEN", "BOT_TOKEN"])
    if not telegram_token:
        raise RuntimeError("Debe configurar TELEGRAM_TOKEN o BOT_TOKEN con el token de Telegram.")

    webhook_secret = _get_first(["WEBHOOK_SECRET"], "tgwebhook").strip("/")
    base_url = _get_first(["BASE_URL", "RENDER_EXTERNAL_URL"], "http://localhost").rstrip("/")
    port = _get_int("PORT", 10000)
    state_path = Path(_get_first(["STATE_PATH"], "state.json"))

    upstash_rest_url = _get_first(["UPSTASH_REDIS_REST_URL", "UPSTASH_URL"])
    upstash_rest_token = _get_first(["UPSTASH_REDIS_REST_TOKEN", "UPSTASH_TOKEN"])
    if upstash_rest_url and not upstash_rest_token:
        raise RuntimeError(
            "UPSTASH_REDIS_REST_TOKEN/UPSTASH_TOKEN es requerido cuando se configura "
            "UPSTASH_REDIS_REST_URL/UPSTASH_URL."
        )
    if upstash_rest_token and not upstash_rest_url:
        raise RuntimeError(
            "UPSTASH_REDIS_REST_URL/UPSTASH_URL es requerido cuando se configura "
            "UPSTASH_REDIS_REST_TOKEN/UPSTASH_TOKEN."
        )

    upstash_redis_url = _get_first(["UPSTASH_REDIS_URL", "REDIS_URL", "redis-url"])
    upstash_state_key = _get_first(["UPSTASH_STATE_KEY"], "bot-econ-state")
    upstash = UpstashConfig(
        rest_url=upstash_rest_url,
        rest_token=upstash_rest_token,
        redis_url=upstash_redis_url,
        state_key=upstash_state_key,
    )

    return BotConfig(
        telegram_token=telegram_token,
        webhook_secret=webhook_secret,
        port=port,
        base_url=base_url,
        state_path=state_path,
        upstash=upstash,
    )


config = load_config()


def build_dependencies(cfg: BotConfig | None = None) -> BotDependencies:
    """Crea las dependencias derivadas de la configuración base."""

    cfg = cfg or config
    webhook_path = f"/{cfg.webhook_secret}"
    webhook_url = f"{cfg.base_url}{webhook_path}"

    return BotDependencies(
        telegram_token=cfg.telegram_token,
        webhook_secret=cfg.webhook_secret,
        webhook_path=webhook_path,
        webhook_url=webhook_url,
        port=cfg.port,
        base_url=cfg.base_url,
        state_path=cfg.state_path,
        upstash=cfg.upstash,
    )
