"""Registro simple de métricas en memoria."""

from __future__ import annotations

import threading
import time
from typing import Dict, Tuple


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: Dict[str, int] = {}
        self._latencies: Dict[str, Tuple[int, float, float]] = {}

    def increment(self, name: str, value: int = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + value

    def observe_latency_ms(self, name: str, duration_ms: float) -> None:
        with self._lock:
            count, total, max_seen = self._latencies.get(name, (0, 0.0, 0.0))
            count += 1
            total += duration_ms
            max_seen = max(max_seen, duration_ms)
            self._latencies[name] = (count, total, max_seen)

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            return {
                "counters": dict(self._counters),
                "latency_ms": {
                    name: {
                        "count": count,
                        "total": total,
                        "avg": (total / count) if count else 0.0,
                        "max": max_seen,
                    }
                    for name, (count, total, max_seen) in self._latencies.items()
                },
                "generated_at": time.time(),
            }


metrics = MetricsRegistry()

