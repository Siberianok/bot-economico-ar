"""State persistence helpers for bot-economico-ar."""

from __future__ import annotations

import abc
import json
import logging
import os
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from aiohttp import ClientError, ClientSession, ClientTimeout

log = logging.getLogger(__name__)

CURRENT_STATE_VERSION = 1


class StateStore(abc.ABC):
    """Abstract contract for reading and writing state payloads."""

    @abc.abstractmethod
    async def load(self) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    @abc.abstractmethod
    async def save(self, payload: Dict[str, Any]) -> bool:
        raise NotImplementedError


class JsonFileStore(StateStore):
    def __init__(self, path: str):
        self.path = path

    async def load(self) -> Optional[Dict[str, Any]]:
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except FileNotFoundError:
            return None
        except Exception as exc:  # pragma: no cover - logging branch
            log.warning("No se pudo leer estado desde %s: %s", self.path, exc)
            return None

    async def save(self, payload: Dict[str, Any]) -> bool:
        try:
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False)
            return True
        except Exception as exc:  # pragma: no cover - logging branch
            log.warning("No se pudo escribir estado en %s: %s", self.path, exc)
            return False


def _upstash_headers(token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


class RedisStore(StateStore):
    def __init__(
        self,
        *,
        state_key: str,
        rest_url: str = "",
        rest_token: str = "",
        redis_url: str = "",
        timeout: float = 10.0,
    ):
        self.state_key = state_key
        self.rest_url = rest_url.rstrip("/")
        self.rest_token = rest_token
        self.redis_url = redis_url
        self.timeout = timeout
        self._redis_client: Any = None
        self._redis_initialized = False

    async def load(self) -> Optional[Dict[str, Any]]:
        data = await self._load_from_rest()
        if data is None:
            data = self._load_from_redis()
        return data

    async def save(self, payload: Dict[str, Any]) -> bool:
        encoded = json.dumps(payload, ensure_ascii=False)
        primary_error: Optional[Exception] = None
        if self._use_rest:
            try:
                await self._rest_request(
                    f"set/{self.state_key}/{quote(encoded, safe='')}", method="POST"
                )
                return True
            except Exception as exc:
                primary_error = exc
            try:
                await self._rest_request(f"set/{self.state_key}", method="POST", data=encoded)
                return True
            except Exception as exc:
                if primary_error:
                    log.warning(
                        "No pude guardar estado en Upstash (fallback falló tras error primario %s): %s",
                        primary_error,
                        exc,
                    )
                else:
                    log.warning("No pude guardar estado en Upstash: %s", exc)

        client = self._get_redis_client()
        if client:
            try:
                client.set(self.state_key, encoded)
                return True
            except Exception as exc:
                log.warning("No pude guardar estado en Redis Upstash: %s", exc)

        return False

    @property
    def _use_rest(self) -> bool:
        return bool(self.rest_url and self.rest_token)

    async def _rest_request(
        self,
        path: str,
        *,
        method: str = "GET",
        data: Optional[str] = None,
        session: Optional[ClientSession] = None,
    ) -> Dict[str, Any]:
        if not self._use_rest:
            raise RuntimeError("Upstash REST no configurado")
        url = f"{self.rest_url}/{path.lstrip('/')}"
        payload = data.encode("utf-8") if data is not None else None
        owns_session = session is None
        if session is None:
            session = ClientSession(timeout=ClientTimeout(total=self.timeout))
        try:
            async with session.request(
                method,
                url,
                data=payload,
                headers=_upstash_headers(self.rest_token),
            ) as resp:
                body = await resp.text()
                if resp.status >= 400:
                    raise RuntimeError(f"Upstash HTTP {resp.status}: {body}")
        except ClientError as exc:
            raise RuntimeError(f"Upstash connection error: {exc}") from exc
        finally:
            if owns_session:
                await session.close()
        try:
            return json.loads(body)
        except Exception as exc:
            raise RuntimeError(f"Upstash invalid response: {body}") from exc

    async def _load_from_rest(self) -> Optional[Dict[str, Any]]:
        if not self._use_rest:
            return None
        try:
            resp = await self._rest_request(f"get/{self.state_key}")
        except Exception as exc:
            log.warning("No pude leer estado de Upstash: %s", exc)
            return None
        raw = resp.get("result") if isinstance(resp, dict) else None
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception as exc:
            log.warning("Estado Upstash inválido: %s", exc)
            return None

    def _get_redis_client(self):
        if self._redis_initialized:
            return self._redis_client
        self._redis_initialized = True
        if not self.redis_url:
            return None
        try:
            import redis  # type: ignore
        except Exception as exc:
            log.warning("Paquete redis no disponible: %s", exc)
            self._redis_client = None
            return None
        try:
            self._redis_client = redis.Redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_timeout=5,
            )
        except Exception as exc:
            log.warning("No pude inicializar cliente Redis de Upstash: %s", exc)
            self._redis_client = None
        return self._redis_client

    def _load_from_redis(self) -> Optional[Dict[str, Any]]:
        client = self._get_redis_client()
        if not client:
            return None
        try:
            raw = client.get(self.state_key)
        except Exception as exc:
            log.warning("No pude leer estado desde Redis Upstash: %s", exc)
            return None
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception as exc:
            log.warning("Estado Redis Upstash inválido: %s", exc)
            return None


def ensure_writable_path(candidate: str, logger: Optional[logging.Logger] = None) -> str:
    logger = logger or log
    try:
        directory = os.path.dirname(candidate) or "."
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        with open(candidate, "a", encoding="utf-8"):
            pass
        return candidate
    except Exception:
        fallback = "./state.json"
        try:
            with open(fallback, "a", encoding="utf-8"):
                pass
            logger.warning(
                "STATE_PATH no escribible (%s). Usando fallback: %s", candidate, fallback
            )
            return fallback
        except Exception as exc:  # pragma: no cover - logging branch
            logger.warning("No puedo escribir estado: %s", exc)
            return fallback


def _clean_alerts(raw: Any) -> Dict[int, List[Dict[str, Any]]]:
    result: Dict[int, List[Dict[str, Any]]] = {}
    if not isinstance(raw, dict):
        return result
    for chat_id, rules in raw.items():
        try:
            cid = int(chat_id)
        except Exception:
            continue
        if not isinstance(rules, list):
            continue
        clean_rules: List[Dict[str, Any]] = []
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            kind = rule.get("kind")
            if kind in {"fx", "metric", "ticker", "crypto"}:
                clean_rules.append(dict(rule))
        if clean_rules:
            result[cid] = clean_rules
    return result


def _clean_subs(raw: Any) -> Dict[int, Dict[str, Any]]:
    result: Dict[int, Dict[str, Any]] = {}
    if not isinstance(raw, dict):
        return result
    for chat_id, conf in raw.items():
        try:
            cid = int(chat_id)
        except Exception:
            continue
        if not isinstance(conf, dict):
            continue
        entry = dict(conf)
        if "daily" in entry and entry["daily"] is not None:
            entry["daily"] = str(entry["daily"])
        result[cid] = entry
    return result


def _clean_pf(raw: Any) -> Dict[int, Dict[str, Any]]:
    result: Dict[int, Dict[str, Any]] = {}
    if not isinstance(raw, dict):
        return result
    for chat_id, conf in raw.items():
        try:
            cid = int(chat_id)
        except Exception:
            continue
        if not isinstance(conf, dict):
            continue
        base = conf.get("base") if isinstance(conf.get("base"), dict) else {}
        items = conf.get("items") if isinstance(conf.get("items"), list) else []
        entry = {
            "base": dict(base),
            "monto": conf.get("monto", 0.0),
            "items": [i for i in items if isinstance(i, dict)],
        }
        result[cid] = entry
    return result


def deserialize_state_payload(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    version_raw = raw.get("version")
    try:
        version = int(version_raw)
    except Exception:
        version = 0

    payload = dict(raw)
    if "version" in payload:
        payload.pop("version")

    if version < 1:
        payload = _migrate_from_unversioned(payload)
        version = 1

    payload["version"] = version
    payload["alerts"] = _clean_alerts(payload.get("alerts"))
    payload["subs"] = _clean_subs(payload.get("subs"))
    payload["pf"] = _clean_pf(payload.get("pf"))
    return payload


def _migrate_from_unversioned(data: Dict[str, Any]) -> Dict[str, Any]:
    migrated = dict(data)
    return migrated


def serialize_state_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    payload = {**data}
    payload["version"] = CURRENT_STATE_VERSION
    payload["alerts"] = _clean_alerts(payload.get("alerts"))
    payload["subs"] = _clean_subs(payload.get("subs"))
    payload["pf"] = _clean_pf(payload.get("pf"))
    return payload
