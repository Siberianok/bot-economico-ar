"""Configuración centralizada del bot."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


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
    link_previews_enabled: bool
    link_previews_prefer_small: bool
    alerts_page_size: int
    rank_top_limit: int
    rank_proj_limit: int


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


def _get_bounded_int(
    key: str, default: int, *, min_value: Optional[int] = None, max_value: Optional[int] = None
) -> int:
    value = _get_int(key, default)
    if min_value is not None:
        value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value


def _get_bool(key: str, default: bool = False) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y"}


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

    link_previews_enabled = _get_bool("LINK_PREVIEWS_ENABLED", False)
    link_previews_prefer_small = _get_bool("LINK_PREVIEWS_PREFER_SMALL", False)
    alerts_page_size = _get_bounded_int("ALERTS_PAGE_SIZE", 10, min_value=1)
    rank_top_limit = _get_bounded_int("RANK_TOP_LIMIT", 3, min_value=1)
    rank_proj_limit = _get_bounded_int("RANK_PROJ_LIMIT", 5, min_value=1)

    return BotConfig(
        telegram_token=telegram_token,
        webhook_secret=webhook_secret,
        port=port,
        base_url=base_url,
        state_path=state_path,
        upstash=upstash,
        link_previews_enabled=link_previews_enabled,
        link_previews_prefer_small=link_previews_prefer_small,
        alerts_page_size=alerts_page_size,
        rank_top_limit=rank_top_limit,
        rank_proj_limit=rank_proj_limit,
    )


config = load_config()
