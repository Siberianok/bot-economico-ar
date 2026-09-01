"""Validation and FastAPI dependency for Telegram Mini App initData."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from dataclasses import dataclass
from time import time
from urllib.parse import parse_qsl

from fastapi import Header

from backend.app.config import Settings, get_settings
from backend.app.utils.errors import ApiError


@dataclass(frozen=True)
class TelegramIdentity:
    user_id: str
    username: str | None = None
    first_name: str | None = None
    is_development: bool = False


def validate_telegram_init_data(
    init_data: str,
    bot_token: str,
    *,
    max_age_seconds: int = 3600,
    now: int | None = None,
) -> TelegramIdentity:
    if not bot_token:
        raise ValueError("token del bot no configurado")
    if not init_data or len(init_data) > 8192:
        raise ValueError("initData vacío o demasiado extenso")
    try:
        pairs = parse_qsl(init_data, keep_blank_values=True, strict_parsing=True)
    except ValueError as exc:
        raise ValueError("initData malformado") from exc
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("initData contiene campos duplicados")
    fields = dict(pairs)

    received_hash = fields.pop("hash", "")
    if not received_hash:
        raise ValueError("initData sin hash")

    data_check_string = "\n".join(f"{key}={fields[key]}" for key in sorted(fields))
    secret_key = hmac.new(b"WebAppData", bot_token.encode("utf-8"), hashlib.sha256).digest()
    calculated_hash = hmac.new(
        secret_key,
        data_check_string.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(received_hash, calculated_hash):
        raise ValueError("firma initData inválida")

    try:
        auth_date = int(fields["auth_date"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("initData sin auth_date válido") from exc

    current_time = int(time()) if now is None else int(now)
    if auth_date > current_time + 30:
        raise ValueError("initData con fecha futura")
    if max_age_seconds > 0 and current_time - auth_date > max_age_seconds:
        raise ValueError("initData vencido")

    try:
        user = json.loads(fields["user"])
        if not isinstance(user, dict):
            raise TypeError
        user_id_value = int(user["id"])
        if user_id_value <= 0:
            raise ValueError
        user_id = str(user_id_value)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("initData sin usuario válido") from exc

    return TelegramIdentity(
        user_id=user_id,
        username=str(user.get("username")) if user.get("username") else None,
        first_name=str(user.get("first_name")) if user.get("first_name") else None,
    )


def telegram_identity_from_header(
    init_data: str | None,
    settings: Settings,
) -> TelegramIdentity:
    if not settings.telegram_auth_required:
        return TelegramIdentity(
            user_id=settings.dev_telegram_user_id,
            username="local-dev",
            is_development=True,
        )

    if not init_data:
        raise ApiError(
            "Autenticación Telegram requerida",
            status_code=401,
            source="telegram_auth",
        )
    bot_token = (
        os.getenv("TELEGRAM_TOKEN", "").strip()
        or os.getenv("BOT_TOKEN", "").strip()
    )
    if not bot_token:
        raise ApiError(
            "Autenticación Telegram no configurada",
            status_code=503,
            source="telegram_auth",
        )
    try:
        return validate_telegram_init_data(
            init_data,
            bot_token,
            max_age_seconds=settings.telegram_init_data_max_age_seconds,
        )
    except ValueError as exc:
        raise ApiError(str(exc), status_code=401, source="telegram_auth") from exc


def get_telegram_identity(
    x_telegram_init_data: str | None = Header(
        default=None,
        alias="X-Telegram-Init-Data",
    ),
) -> TelegramIdentity:
    return telegram_identity_from_header(x_telegram_init_data, get_settings())
