import hashlib
import hmac
import json
from dataclasses import replace
from urllib.parse import urlencode

import pytest

from backend.app.config import get_settings
from backend.app.utils.errors import ApiError
from backend.app.utils.telegram_auth import (
    telegram_identity_from_header,
    validate_telegram_init_data,
)


def _signed_init_data(token: str, *, auth_date: int, user_id: int = 1234) -> str:
    fields = {
        "auth_date": str(auth_date),
        "query_id": "AAE-test",
        "user": json.dumps(
            {"id": user_id, "first_name": "Ariel", "username": "test_user"},
            separators=(",", ":"),
        ),
    }
    data_check_string = "\n".join(f"{key}={fields[key]}" for key in sorted(fields))
    secret_key = hmac.new(b"WebAppData", token.encode(), hashlib.sha256).digest()
    fields["hash"] = hmac.new(
        secret_key,
        data_check_string.encode(),
        hashlib.sha256,
    ).hexdigest()
    return urlencode(fields)


def test_validate_telegram_init_data_returns_signed_user():
    raw = _signed_init_data("bot-token", auth_date=1_700_000_000)
    identity = validate_telegram_init_data(
        raw,
        "bot-token",
        now=1_700_000_100,
    )
    assert identity.user_id == "1234"
    assert identity.username == "test_user"


def test_validate_telegram_init_data_rejects_expired_data():
    raw = _signed_init_data("bot-token", auth_date=1_700_000_000)
    with pytest.raises(ValueError, match="vencido"):
        validate_telegram_init_data(
            raw,
            "bot-token",
            max_age_seconds=60,
            now=1_700_000_100,
        )


def test_validate_telegram_init_data_rejects_non_positive_user():
    raw = _signed_init_data("bot-token", auth_date=1_700_000_000, user_id=0)
    with pytest.raises(ValueError, match="usuario válido"):
        validate_telegram_init_data(raw, "bot-token", now=1_700_000_100)


def test_validate_telegram_init_data_rejects_duplicate_fields():
    raw = _signed_init_data("bot-token", auth_date=1_700_000_000)
    with pytest.raises(ValueError, match="duplicados"):
        validate_telegram_init_data(
            f"{raw}&auth_date=1700000000",
            "bot-token",
            now=1_700_000_100,
        )


def test_development_identity_is_explicit_fallback():
    settings = replace(
        get_settings(),
        telegram_auth_required=False,
        dev_telegram_user_id="local-test",
    )
    identity = telegram_identity_from_header(None, settings)
    assert identity.user_id == "local-test"
    assert identity.is_development is True


def test_required_identity_rejects_missing_header():
    settings = replace(get_settings(), telegram_auth_required=True)
    with pytest.raises(ApiError) as exc_info:
        telegram_identity_from_header(None, settings)
    assert exc_info.value.status_code == 401
