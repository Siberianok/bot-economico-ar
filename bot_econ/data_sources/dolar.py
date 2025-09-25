# bot_econ/data_sources/dolar.py
# -*- coding: utf-8 -*-

"""
Fuentes de cotizaciones de dólares para Argentina.

Expone:
    - fetch_dolar_quotes() -> dict[str, dict]
        Devuelve un diccionario con claves:
        {oficial, mayorista, blue, mep, ccl, tarjeta, cripto}
        Cada valor: {"compra": float|None, "venta": float|None, "fuente": str, "fecha": str|None}

Notas:
    - Path de MEP en DolarAPI es /dolares/bolsa  ✅
    - Path de CCL en DolarAPI es /dolares/contadoconliqui  ✅
    - Para "cripto" usamos CriptoYa como fuente primaria.
"""

from typing import Any, Dict, Optional
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
    "mep":       "/dolares/bolsa",              # 👈 CORRECTO
    "ccl":       "/dolares/contadoconliqui",    # 👈 CORRECTO
    "tarjeta":   "/dolares/tarjeta",
    # "cripto":  No está estandarizado en DolarAPI; lo tomamos de CriptoYa
}


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

    # Algunos endpoints devuelven { compra, venta, fechaActualizacion/fecha }
    compra = data.get("compra")
    venta = data.get("venta")
    fecha = data.get("fechaActualizacion") or data.get("fecha")

    try:
        compra = float(compra) if compra is not None else None
    except Exception:
        compra = None

    try:
        venta = float(venta) if venta is not None else None
    except Exception:
        venta = None

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
    compra = cr.get("compra") or cr.get("buy")
    venta = cr.get("venta") or cr.get("sell")

    try:
        compra = float(compra) if compra is not None else None
    except Exception:
        compra = None
    try:
        venta = float(venta) if venta is not None else None
    except Exception:
        venta = None

    if compra is None and venta is None:
        return None

    return {"compra": compra, "venta": venta, "fuente": "CriptoYa", "fecha": None}


async def fetch_dolar_quotes() -> Dict[str, Dict[str, Any]]:
    """
    Punto de entrada utilizado por metrics_pipeline.fetch_summary().
    Arma un mapa con todas las cotizaciones relevantes.
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

    # 3) Fallback simple: si algo faltó, intentamos rellenar con CriptoYa.
    #    (CriptoYa también devuelve 'mep' y 'ccl' en algunos casos)
    try:
        cy = await http.fetch_json(CRYPTOYA_DOLAR_URL)
    except Exception:
        cy = None

    def _safe_get(block: Any) -> (Optional[float], Optional[float]):
        if not isinstance(block, dict):
            return (None, None)
        c = block.get("compra") or block.get("buy")
        v = block.get("venta") or block.get("sell")
        try:
            c = float(c) if c is not None else None
        except Exception:
            c = None
        try:
            v = float(v) if v is not None else None
        except Exception:
            v = None
        return (c, v)

    if isinstance(cy, dict):
        # mep
        if "mep" not in out:
            c, v = _safe_get(cy.get("mep"))
            if c is not None or v is not None:
                out["mep"] = {"compra": c, "venta": v, "fuente": "CriptoYa", "fecha": None}
        # ccl
        if "ccl" not in out:
            c, v = _safe_get(cy.get("ccl"))
            if c is not None or v is not None:
                out["ccl"] = {"compra": c, "venta": v, "fuente": "CriptoYa", "fecha": None}
        # oficial/blue mayorista (si faltaran)
        for alias in ("oficial", "mayorista", "blue"):
            if alias not in out:
                c, v = _safe_get(cy.get(alias))
                if c is not None or v is not None:
                    out[alias] = {"compra": c, "venta": v, "fuente": "CriptoYa", "fecha": None}

    return out
