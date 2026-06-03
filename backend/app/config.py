"""Backend configuration for the Observatorio Economico API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List


ROOT_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = ROOT_DIR.parent


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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    database_url = os.getenv("DATABASE_URL", "sqlite:///./data/observatorio.db").strip()
    miniapp_url = os.getenv("MINIAPP_URL", "http://127.0.0.1:5173").strip()
    origins = _split_csv(
        os.getenv(
            "API_CORS_ORIGINS",
            "http://127.0.0.1:5173,http://localhost:5173",
        )
    )
    return Settings(
        app_name=os.getenv("APP_NAME", "Observatorio Economico API").strip(),
        app_version=os.getenv("APP_VERSION", "0.1.0").strip(),
        database_url=database_url,
        miniapp_url=miniapp_url,
        api_cors_origins=origins,
        signal_min_score=int(os.getenv("SIGNAL_MIN_SCORE", "70")),
        dev_telegram_user_id=os.getenv("DEV_TELEGRAM_USER_ID", "local-dev").strip(),
        legacy_live_fetch_enabled=_bool_env("LEGACY_LIVE_FETCH_ENABLED", False),
        legacy_import_token=os.getenv("LEGACY_IMPORT_TOKEN", "api-placeholder-token").strip(),
        legacy_state_path=os.getenv("LEGACY_STATE_PATH", os.devnull).strip() or os.devnull,
    )

