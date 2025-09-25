# bot_econ/data_sources/dolar.py
# -*- coding: utf-8 -*-

"""
Fuentes de cotizaciones de dólares para Argentina.

Expone:
    - fetch_dolar_quotes() -> dict[str, dict]
        Devuelve:
        {
          "oficial": {"compra": float|None, "venta": float|None, "fuente": str, "fecha": str|None},
          "mayorista": {...},
          "blue": {...},
          "mep": {...},
          "ccl": {...},
          "tarjeta": {...},
          "cripto": {...}
        }

    - fetch_oficial_blue() -> dict[str, dict]
        Compatibilidad para pipelines anteriores. Devuelve solo
        {"oficial": {...}, "blue": {...}} (si están disponibles).

Notas:
    - Path de MEP en DolarAPI es /dolares/bolsa
    - Path de CCL en DolarAPI es /dolares/contadoconliqui
    - Para "cripto" usamos CriptoYa como fuente primaria.
"""

from typing import Any, Dict, Optional, Tuple
import logging
from . import http  # helper async con fetch_json / fetch_text

log = logging.getLogger(__name__)

DOLARAPI_BASE = "https://dolarapi.com/v1"
CRYPTOYA_DOLAR_URL = "https://criptoya.com/api/dolar"

# Paths probados de DolarAPI
_DOLARAPI_PATHS: Dict[str, str] = {
    "oficial":   "/dolares/oficial",
    "mayorista": "/dolares/mayorista",
    "blue":      "/dolares/blue",
    "mep":       "/dolares/bolsa",              # ✅ correcto
    "ccl":       "/dolares/contadoconliqui",    # ✅ correcto
    "tarjeta":   "/dolares/tarjeta",
    # "cripto":  no estandar en DolarAPI; tomamos CriptoYa
}

# -------------------------- helpers --------------------------

def _coerce_float(x: Any) -> Optional[float]:
    try:
        return float(x) if x is not None else None
    except Exception:
        return None

async def _fetch_from_dolarapi(kind: str) -> Optional[Dict[str, Any]]:
    """
    Lee una cotización de DolarAPI y la normaliza.
    Retorna dict con compra/venta/fuente/fecha o None si falla.
    """
    path = _DOLARAPI_PATHS.get(kind)
    if not path:
        return None

    url = f"{DOLARAPI_BASE}{path}"
    try:
        data = await http.fetch_json(url)
    except Exception as e:
        log.warning("DolarAPI fallo %s: %s", url, e)
        return None

    if not isinstance(data, dict):
        return None

    compra = _coerce_float(data.get("compra"))
    venta = _coerce_float(data.get("venta"))
    fecha = data.get("fechaActualizacion") or data.get("fecha")
    if compra is None and venta is None:
        return None
    return {"compra": compra, "venta": venta, "fuente": "DolarAPI", "fecha": fecha}

async def _fetch_cripto_from_criptoya() -> Optional[Dict[str, Any]]:
    """
    Lee 'cripto' de CriptoYa y lo normaliza.
    """
    try:
        data = await http.fetch_json(CRYPTOYA_DOLAR_URL)
    except Exception as e:
        log.warning("CriptoYa fallo: %s", e)
        return None

    if not isinstance(data, dict):
        return None

    # CriptoYa suele exponer: { "cripto": {"compra": x, "venta": y}, ... }
    cr = data.get("cripto") or data.get("crypto") or {}
    compra = _coerce_float(cr.get("compra") or cr.get("buy"))
    venta  = _coerce_float(cr.get("venta")  or cr.get("sell"))

    if compra is None and venta is None:
        return None
    return {"compra": compra, "venta": venta, "fuente": "CriptoYa", "fecha": None}

def _safe_from_criptoya_block(block: Any) -> Tuple[Optional[float], Optional[float]]:
    if not isinstance(block, dict):
        return (None, None)
    c = _coerce_float(block.get("compra") or block.get("buy"))
    v = _coerce_float(block.get("venta")  or block.get("sell"))
    return (c, v)

# -------------------------- API pública --------------------------

async def fetch_dolar_quotes() -> Dict[str, Dict[str, Any]]:
    """
    Punto de entrada utilizado por metrics_pipeline.fetch_summary().
    Devuelve todas las cotizaciones relevantes.
    """
    out: Dict[str, Dict[str, Any]] = {}

    # 1) DolarAPI para las cotizaciones estándar
    for k in ("oficial", "mayorista", "blue", "mep", "ccl", "tarjeta"):
        row = await _fetch_from_dolarapi(k)
        if row:
            out[k] = row

    # 2) Cripto (CriptoYa)
    cripto = await _fetch_cripto_from_criptoya()
    if cripto:
        out["cripto"] = cripto

    # 3) Fallbacks desde CriptoYa cuando DolarAPI no devuelve algo
    try:
        cy = await http.fetch_json(CRYPTOYA_DOLAR_URL)
    except Exception:
        cy = None

    if isinstance(cy, dict):
        # mep / ccl
        if "mep" not in out:
            c, v = _safe_from_criptoya_block(cy.get("mep"))
            if c is not None or v is not None:
                out["mep"] = {"compra": c, "venta": v, "fuente": "CriptoYa", "fecha": None}
        if "ccl" not in out:
            c, v = _safe_from_criptoya_block(cy.get("ccl"))
            if c is not None or v is not None:
                out["ccl"] = {"compra": c, "venta": v, "fuente": "CriptoYa", "fecha": None}
        # oficial/blue/mayorista
        for alias in ("oficial", "mayorista", "blue"):
            if alias not in out:
                c, v = _safe_from_criptoya_block(cy.get(alias))
                if c is not None or v is not None:
                    out[alias] = {"compra": c, "venta": v, "fuente": "CriptoYa", "fecha": None}

    return out

async def fetch_oficial_blue() -> Dict[str, Dict[str, Any]]:
    """
    Compatibilidad con pipelines viejos.
    Devuelve solo oficial y blue (si existen).
    """
    quotes = await fetch_dolar_quotes()
    out: Dict[str, Dict[str, Any]] = {}
    if "oficial" in quotes:
        out["oficial"] = quotes["oficial"]
    if "blue" in quotes:
        out["blue"] = quotes["blue"]
    return out
