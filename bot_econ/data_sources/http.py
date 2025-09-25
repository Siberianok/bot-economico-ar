from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from aiohttp import ClientError, ClientResponseError, ClientSession, ClientTimeout

log = logging.getLogger(__name__)


@dataclass(slots=True)
class CacheEntry:
    value: Any
    expires_at: float


class HttpClient:
    """Async HTTP client with retry, caching and shared session."""

    def __init__(
        self,
        *,
        timeout: float = 15.0,
        retries: int = 3,
        backoff_factor: float = 0.6,
        cache_ttl: float = 0.0,
    ) -> None:
        self._timeout = timeout
        self._retries = retries
        self._backoff_factor = backoff_factor
        self._cache_ttl = cache_ttl
        self._session: ClientSession | None = None
        self._session_lock = asyncio.Lock()
        self._cache: dict[str, CacheEntry] = {}
        self._cache_lock = asyncio.Lock()

    async def _ensure_session(self) -> ClientSession:
        async with self._session_lock:
            if self._session is None or self._session.closed:
                timeout = ClientTimeout(total=self._timeout)
                self._session = ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        async with self._session_lock:
            if self._session and not self._session.closed:
                await self._session.close()
            self._session = None

    async def _get_cached(self, key: str) -> Any | None:
        if self._cache_ttl <= 0:
            return None
        async with self._cache_lock:
            entry = self._cache.get(key)
            if entry and entry.expires_at > time.monotonic():
                return entry.value
            if entry:
                self._cache.pop(key, None)
        return None

    async def _set_cached(self, key: str, value: Any) -> None:
        if self._cache_ttl <= 0:
            return
        expires = time.monotonic() + self._cache_ttl
        async with self._cache_lock:
            self._cache[key] = CacheEntry(value=value, expires_at=expires)

    async def fetch_text(self, url: str, *, headers: dict[str, str] | None = None) -> str:
        cache_key = f"text:{url}"
        cached = await self._get_cached(cache_key)
        if cached is not None:
            return cached

        session = await self._ensure_session()
        attempt = 0
        delay = self._backoff_factor
        while True:
            try:
                async with session.get(url, headers=headers) as resp:
                    resp.raise_for_status()
                    text = await resp.text()
                    await self._set_cached(cache_key, text)
                    return text
            except (ClientResponseError, ClientError, asyncio.TimeoutError) as exc:
                attempt += 1
                if attempt >= self._retries:
                    log.warning(
                        "fetch_text failed",
                        extra={"url": url, "attempt": attempt},
                        exc_info=exc,
                    )
                    raise
                await asyncio.sleep(delay)
                delay *= 2

    async def fetch_json(self, url: str, *, headers: dict[str, str] | None = None) -> Any:
        cache_key = f"json:{url}"
        cached = await self._get_cached(cache_key)
        if cached is not None:
            return cached

        session = await self._ensure_session()
        attempt = 0
        delay = self._backoff_factor
        while True:
            try:
                async with session.get(url, headers=headers) as resp:
                    resp.raise_for_status()
                    payload = await resp.text()
                    data = json.loads(payload)
                    await self._set_cached(cache_key, data)
                    return data
            except (
                ClientResponseError,
                ClientError,
                asyncio.TimeoutError,
                json.JSONDecodeError,
            ) as exc:
                attempt += 1
                if attempt >= self._retries:
                    log.warning(
                        "fetch_json failed",
                        extra={"url": url, "attempt": attempt},
                        exc_info=exc,
                    )
                    raise
                await asyncio.sleep(delay)
                delay *= 2


_http_client = HttpClient(cache_ttl=30.0)


async def get_http_client() -> HttpClient:
    return _http_client
