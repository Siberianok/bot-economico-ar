import asyncio
import asyncio
import logging
import random
import time
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import httpx

log = logging.getLogger(__name__)


class SourceSuspendedError(Exception):
    def __init__(self, source: str, resume_at: float):
        self.source = source
        self.resume_at = resume_at
        super().__init__(f"source {source} suspended until {resume_at}")


class HttpService:
    def __init__(
        self,
        timeout: float = 5.0,
        max_retries: int = 3,
        base_backoff: float = 0.4,
        failure_threshold: int = 3,
        suspend_seconds: int = 60,
    ) -> None:
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout), follow_redirects=True
        )
        self._max_retries = max_retries
        self._base_backoff = base_backoff
        self._failure_threshold = failure_threshold
        self._suspend_seconds = suspend_seconds
        self._fail_streak: Dict[str, int] = {}
        self._suspended_until: Dict[str, float] = {}
        self.metrics: Dict[str, Dict[str, int]] = {
            "success": {},
            "failure": {},
        }
        self._lock = asyncio.Lock()

    def _source_name(self, url: str, source: Optional[str]) -> str:
        if source:
            return source
        parsed = urlparse(url)
        return parsed.netloc or "unknown"

    def _mark_success(self, source: str) -> None:
        self.metrics.setdefault("success", {})
        self.metrics.setdefault("failure", {})
        self.metrics["success"][source] = self.metrics["success"].get(source, 0) + 1
        self._fail_streak[source] = 0

    def _mark_failure(self, source: str) -> None:
        self.metrics.setdefault("failure", {})
        self.metrics.setdefault("success", {})
        self.metrics["failure"][source] = self.metrics["failure"].get(source, 0) + 1
        self._fail_streak[source] = self._fail_streak.get(source, 0) + 1
        if self._fail_streak[source] >= self._failure_threshold:
            resume_at = time.time() + self._suspend_seconds
            self._suspended_until[source] = resume_at
            log.warning(
                "source_suspended source=%s resume_at=%s failure_streak=%s",
                source,
                resume_at,
                self._fail_streak[source],
            )

    def is_suspended(self, source: str) -> Optional[float]:
        resume_at = self._suspended_until.get(source)
        if resume_at and resume_at > time.time():
            return resume_at
        if resume_at:
            self._suspended_until.pop(source, None)
        return None

    async def _request(self, method: str, url: str, *, source: Optional[str] = None, **kwargs: Any) -> httpx.Response:
        src = self._source_name(url, source)
        suspended_until = self.is_suspended(src)
        if suspended_until:
            raise SourceSuspendedError(src, suspended_until)

        last_error: Optional[Exception] = None
        for attempt in range(self._max_retries):
            try:
                resp = await self._client.request(method, url, **kwargs)
                if 200 <= resp.status_code < 300:
                    self._mark_success(src)
                    return resp
                last_error = RuntimeError(f"status={resp.status_code}")
                log.warning(
                    "http_request_failed source=%s url=%s status=%s attempt=%s",
                    src,
                    url,
                    resp.status_code,
                    attempt + 1,
                )
            except Exception as exc:
                last_error = exc
                log.warning(
                    "http_request_error source=%s url=%s attempt=%s error=%s",
                    src,
                    url,
                    attempt + 1,
                    exc,
                )
            if attempt + 1 < self._max_retries:
                backoff = self._base_backoff * (2 ** attempt) + random.uniform(0, self._base_backoff)
                await asyncio.sleep(backoff)

        self._mark_failure(src)
        if last_error:
            raise last_error
        raise RuntimeError(f"request failed for {src}")

    async def get_json(self, url: str, *, source: Optional[str] = None, headers: Optional[Dict[str, str]] = None, **kwargs: Any) -> Dict[str, Any]:
        resp = await self._request(
            "GET", url, source=source, headers=headers or {}, **kwargs
        )
        return resp.json()

    async def get_text(self, url: str, *, source: Optional[str] = None, headers: Optional[Dict[str, str]] = None, **kwargs: Any) -> str:
        resp = await self._request(
            "GET", url, source=source, headers=headers or {}, **kwargs
        )
        return resp.text

    async def aclose(self) -> None:
        async with self._lock:
            await self._client.aclose()

    def snapshot_metrics(self) -> Dict[str, Dict[str, int]]:
        return {
            "success": dict(self.metrics.get("success", {})),
            "failure": dict(self.metrics.get("failure", {})),
        }


http_service = HttpService()
