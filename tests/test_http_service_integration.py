import asyncio
import os
import pathlib
import sys

import httpx
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
os.environ.setdefault("TELEGRAM_TOKEN", "test-token")

from bot.services.http import HttpService, SourceSuspendedError


def test_http_service_retries_then_success_and_metrics():
    async def _run():
        attempts = {"count": 0}

        async def handler(request: httpx.Request) -> httpx.Response:
            attempts["count"] += 1
            if attempts["count"] == 1:
                return httpx.Response(status_code=500, request=request)
            return httpx.Response(status_code=200, json={"ok": True}, request=request)

        service = HttpService(max_retries=2, base_backoff=0)
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True)
        service.configure_client(client)

        payload = await service.get_json("https://api.example.com/data")

        assert payload == {"ok": True}
        assert attempts["count"] == 2
        snap = service.snapshot_metrics()
        assert snap["success"]["api.example.com"] == 1
        assert snap["failure"].get("api.example.com", 0) == 0

        await client.aclose()

    asyncio.run(_run())


def test_http_service_marks_failure_and_suspends_source():
    async def _run():
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status_code=503, request=request)

        service = HttpService(max_retries=1, failure_threshold=1, suspend_seconds=60, base_backoff=0)
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True)
        service.configure_client(client)

        with pytest.raises(RuntimeError):
            await service.get_text("https://fallback.example.com/a")

        snap = service.snapshot_metrics()
        assert snap["failure"]["fallback.example.com"] == 1

        with pytest.raises(SourceSuspendedError):
            await service.get_text("https://fallback.example.com/b")

        await client.aclose()

    asyncio.run(_run())
