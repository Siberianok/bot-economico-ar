"""Backend configuration for the ObeconBot API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List


ROOT_DIR = Path(__file__).resolve().parents[2]


def _split_csv(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _bool_env(key: str, default: bool = False) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Settings:
    app_name: str
    app_version: str
    database_url: str
    miniapp_url: str
    api_cors_origins: List[str]
    signal_min_score: int
    dev_telegram_user_id: str
    legacy_live_fetch_enabled: bool
    legacy_import_token: str
    legacy_state_path: str
    legacy_market_fetch_timeout_seconds: float
    legacy_screener_fetch_timeout_seconds: float
    telegram_runtime_enabled: bool
    telegram_token_configured: bool
    public_base_url: str
    webhook_secret: str
    keepalive_enabled: bool
    telegram_auth_required: bool
    telegram_init_data_max_age_seconds: int


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    database_url = os.getenv("DATABASE_URL", "sqlite:///./data/obeconbot.db").strip()
    miniapp_url = os.getenv("MINIAPP_URL", "http://127.0.0.1:5173").strip()
    origins = _split_csv(
        os.getenv(
            "API_CORS_ORIGINS",
            "http://127.0.0.1:5173,http://localhost:5173",
        )
    )
    telegram_runtime_enabled = _bool_env("TELEGRAM_RUNTIME_ENABLED", False)
    isolated_legacy_state = ROOT_DIR / ".codex" / ".tmp" / "legacy-adapter-state.json"
    return Settings(
        app_name=os.getenv("APP_NAME", "ObeconBot API").strip(),
        app_version=os.getenv("APP_VERSION", "0.1.0").strip(),
        database_url=database_url,
        miniapp_url=miniapp_url,
        api_cors_origins=origins,
        signal_min_score=int(os.getenv("SIGNAL_MIN_SCORE", "70")),
        dev_telegram_user_id=os.getenv("DEV_TELEGRAM_USER_ID", "local-dev").strip(),
        legacy_live_fetch_enabled=_bool_env("LEGACY_LIVE_FETCH_ENABLED", False),
        legacy_import_token=os.getenv("LEGACY_IMPORT_TOKEN", "api-placeholder-token").strip(),
        legacy_state_path=(
            os.getenv("LEGACY_STATE_PATH", str(isolated_legacy_state)).strip()
            or str(isolated_legacy_state)
        ),
        legacy_market_fetch_timeout_seconds=max(
            5.0, float(os.getenv("LEGACY_MARKET_FETCH_TIMEOUT_SECONDS", "25"))
        ),
        legacy_screener_fetch_timeout_seconds=max(
            15.0, float(os.getenv("LEGACY_SCREENER_FETCH_TIMEOUT_SECONDS", "90"))
        ),
        telegram_runtime_enabled=telegram_runtime_enabled,
        telegram_token_configured=bool(
            os.getenv("TELEGRAM_TOKEN", "").strip() or os.getenv("BOT_TOKEN", "").strip()
        ),
        public_base_url=(
            os.getenv("BASE_URL", "").strip()
            or os.getenv("RENDER_EXTERNAL_URL", "").strip()
            or "http://127.0.0.1:8000"
        ).rstrip("/"),
        webhook_secret=os.getenv("WEBHOOK_SECRET", "").strip().strip("/"),
        keepalive_enabled=_bool_env("KEEPALIVE_ENABLED", True),
        telegram_auth_required=_bool_env(
            "TELEGRAM_AUTH_REQUIRED", telegram_runtime_enabled
        ),
        telegram_init_data_max_age_seconds=int(
            os.getenv("TELEGRAM_INIT_DATA_MAX_AGE_SECONDS", "3600")
        ),
    )
