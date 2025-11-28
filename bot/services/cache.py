import json
import logging
import os
import time
from typing import Any, Dict, Iterable, Optional

log = logging.getLogger(__name__)


class ShortCache:
    def __init__(
        self,
        *,
        default_ttl: int = 45,
        redis_url: Optional[str] = None,
        namespace: str = "bot-econ:cache",
    ) -> None:
        self.default_ttl = default_ttl
        self.namespace = namespace.rstrip(":")
        self._redis_url = (redis_url or os.getenv("UPSTASH_REDIS_URL") or "").strip()
        self._client = self._init_redis()
        self._memory: Dict[str, Any] = {}
        self._memory_expiry: Dict[str, float] = {}

    def _init_redis(self):
        if not self._redis_url:
            return None
        try:
            import redis  # type: ignore

            return redis.Redis.from_url(
                self._redis_url,
                decode_responses=True,
                socket_timeout=5,
            )
        except Exception as exc:  # pragma: no cover - best effort
            log.warning("No pude inicializar cache Redis: %s", exc)
            return None

    def _k(self, key: str) -> str:
        return f"{self.namespace}:{key}"

    def get(self, key: str) -> Optional[Any]:
        now = time.time()
        if key in self._memory:
            expires = self._memory_expiry.get(key, 0)
            if expires and expires > now:
                return self._memory[key]
            self._memory.pop(key, None)
            self._memory_expiry.pop(key, None)

        if self._client:
            try:
                raw = self._client.get(self._k(key))
                if raw is None:
                    return None
                return json.loads(raw)
            except Exception as exc:  # pragma: no cover - best effort
                log.debug("Redis cache get falló, usando memoria (%s)", exc)
                return None
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        ttl_val = ttl or self.default_ttl
        self._memory[key] = value
        self._memory_expiry[key] = time.time() + ttl_val
        if self._client:
            try:
                self._client.setex(self._k(key), ttl_val, json.dumps(value, ensure_ascii=False))
            except Exception as exc:  # pragma: no cover - best effort
                log.debug("Redis cache set falló, usando memoria (%s)", exc)

    def invalidate(self, *keys: str) -> None:
        for key in keys:
            self._memory.pop(key, None)
            self._memory_expiry.pop(key, None)
            if self._client:
                try:
                    self._client.delete(self._k(key))
                except Exception as exc:  # pragma: no cover - best effort
                    log.debug("Redis cache delete falló: %s", exc)

    def invalidate_prefix(self, prefix: str) -> None:
        keys_to_delete = [k for k in self._memory if k.startswith(prefix)]
        for key in keys_to_delete:
            self._memory.pop(key, None)
            self._memory_expiry.pop(key, None)
        if self._client:
            pattern = f"{self._k(prefix)}*"
            try:
                for redis_key in self._client.scan_iter(pattern):
                    self._client.delete(redis_key)
            except Exception as exc:  # pragma: no cover - best effort
                log.debug("Redis cache invalidation falló: %s", exc)


class RateLimiter:
    def __init__(self, *, redis_url: Optional[str] = None, namespace: str = "bot-econ:throttle") -> None:
        self.namespace = namespace.rstrip(":")
        self._redis_url = (redis_url or os.getenv("UPSTASH_REDIS_URL") or "").strip()
        self._client = self._init_redis()
        self._memory_expiry: Dict[str, float] = {}

    def _init_redis(self):
        if not self._redis_url:
            return None
        try:
            import redis  # type: ignore

            return redis.Redis.from_url(
                self._redis_url,
                decode_responses=True,
                socket_timeout=5,
            )
        except Exception as exc:  # pragma: no cover - best effort
            log.warning("No pude inicializar rate limiter Redis: %s", exc)
            return None

    def _k(self, key: str) -> str:
        return f"{self.namespace}:{key}"

    def hit(self, bucket: str, ttl: int) -> bool:
        now = time.time()
        mem_expiry = self._memory_expiry.get(bucket)
        if mem_expiry and mem_expiry > now:
            return True

        if self._client:
            try:
                was_set = self._client.set(self._k(bucket), str(int(now)), nx=True, ex=ttl)
                if was_set:
                    self._memory_expiry[bucket] = now + ttl
                    return False
                return True
            except Exception as exc:  # pragma: no cover - best effort
                log.debug("Redis rate limit falló, usando memoria (%s)", exc)

        self._memory_expiry[bucket] = now + ttl
        return False

    def clear_prefix(self, prefix: str) -> None:
        keys_to_delete = [k for k in list(self._memory_expiry.keys()) if k.startswith(prefix)]
        for key in keys_to_delete:
            self._memory_expiry.pop(key, None)
        if self._client:
            pattern = f"{self._k(prefix)}*"
            try:
                for redis_key in self._client.scan_iter(pattern):
                    self._client.delete(redis_key)
            except Exception as exc:  # pragma: no cover - best effort
                log.debug("Redis rate limit invalidation falló: %s", exc)
