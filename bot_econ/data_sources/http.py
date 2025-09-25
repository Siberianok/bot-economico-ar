# bot_econ/data_sources/http.py
# -*- coding: utf-8 -*-
from typing import Any, Dict, Optional
import json
import aiohttp
import logging

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = aiohttp.ClientTimeout(total=15)
DEFAULT_HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "*/*"}

async def fetch_json(
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    timeout: Optional[aiohttp.ClientTimeout] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    try:
        async with aiohttp.ClientSession(timeout=timeout or DEFAULT_TIMEOUT) as session:
            async with session.get(url, headers={**DEFAULT_HEADERS, **(headers or {})}, params=params) as resp:
                if resp.status != 200:
                    log.warning("fetch_json HTTP %s for %s", resp.status, url)
                    return None
                ctype = (resp.headers.get("Content-Type") or "").lower()
                if "json" in ctype:
                    # No forzamos content_type para tolerar JSON mal tipado
                    return await resp.json(content_type=None)
                # No es JSON → intento parsear el texto a mano; si falla, devuelvo None
                text_payload = await resp.text()
                try:
                    return json.loads(text_payload)
                except Exception:
                    log.warning("fetch_json: non-JSON content at %s", url)
                    return None
    except Exception as e:
        log.warning("fetch_json exception %s: %s", url, e)
        return None

async def fetch_text(
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    timeout: Optional[aiohttp.ClientTimeout] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    try:
        async with aiohttp.ClientSession(timeout=timeout or DEFAULT_TIMEOUT) as session:
            async with session.get(url, headers={**DEFAULT_HEADERS, **(headers or {})}, params=params) as resp:
                if resp.status != 200:
                    log.warning("fetch_text HTTP %s for %s", resp.status, url)
                    return None
                return await resp.text()
    except Exception as e:
        log.warning("fetch_text exception %s: %s", url, e)
        return None
