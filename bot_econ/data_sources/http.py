# bot_econ/data_sources/http.py
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Optional

import aiohttp

log = logging.getLogger(__name__)

# ====== Config HTTP por defecto ======
_DEFAULT_TIMEOUT_SECS = 15
_DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (BotEcon/1.0; +https://render.com)",
    "Accept": "*/*",
}

# ====== Cliente singleton (aiohttp) ======
_session: Optional[aiohttp.ClientSession] = None

def _make_timeout() -> aiohttp.ClientTimeout:
    return aiohttp.ClientTimeout(total=_DEFAULT_TIMEOUT_SECS)

def _ensure_headers(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    if not extra:
        return dict(_DEFAULT_HEADERS)
    h = dict(_DEFAULT_HEADERS)
    h.update(extra)
    return h

async def get_http_client() -> aiohttp.ClientSession:
    """
    Compatibilidad: algunos módulos esperan 'get_http_client'.
    Devuelve un ClientSession (singleton).
    """
    global _session
    if _session is None or _session.closed:
        _session = aiohttp.ClientSession(timeout=_make_timeout())
        log.debug("HTTP client (aiohttp) creado")
    return _session

async def close_http_client() -> None:
    """
    Llamado en el apagado de la app para cerrar el HTTP client.
    Seguro de invocar múltiples veces.
    """
    global _session
    try:
        if _session is not None and not _session.closed:
            await _session.close()
            log.debug("HTTP client (aiohttp) cerrado")
    finally:
        _session = None

# ====== Helpers de red ======

async def fetch_json(
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    timeout: Optional[aiohttp.ClientTimeout] = None,
) -> Optional[Dict[str, Any]]:
    """
    GET -> JSON (dict). Devuelve None si status != 200 o si el payload no es JSON.
    """
    ses = await get_http_client()
    try:
        async with ses.get(url, headers=_ensure_headers(headers), params=params, timeout=timeout or _make_timeout()) as resp:
            if resp.status != 200:
                txt = await _safe_text(resp)
                log.warning("fetch_json failed: %s -> %s | body=%s", url, resp.status, (txt[:200] if txt else ""))
                return None
            # Intentamos json() primero; si falla por content-type, intentamos decodificar manualmente
            try:
                return await resp.json(content_type=None)
            except Exception:
                payload = await resp.text()
                try:
                    return json.loads(payload)
                except Exception:
                    log.warning("fetch_json: payload no es JSON válido (%s)", url)
                    return None
    except asyncio.CancelledError:
        raise
    except Exception as e:
        log.warning("fetch_json error %s: %s", url, e)
        return None


async def fetch_text(
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    timeout: Optional[aiohttp.ClientTimeout] = None,
) -> Optional[str]:
    """
    GET -> texto. Devuelve None si status != 200.
    """
    ses = await get_http_client()
    try:
        async with ses.get(url, headers=_ensure_headers(headers), params=params, timeout=timeout or _make_timeout()) as resp:
            if resp.status != 200:
                txt = await _safe_text(resp)
                log.warning("fetch_text failed: %s -> %s | body=%s", url, resp.status, (txt[:200] if txt else ""))
                return None
            return await resp.text()
    except asyncio.CancelledError:
        raise
    except Exception as e:
        log.warning("fetch_text error %s: %s", url, e)
        return None


async def _safe_text(resp: aiohttp.ClientResponse) -> Optional[str]:
    try:
        return await resp.text()
    except Exception:
        return None
