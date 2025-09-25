# bot_econ/data_sources/dolar.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import logging
from . import http  # requiere http.fetch_json y http.fetch_text (ya los tenés)

log = logging.getLogger(__name__)

DOLARAPI_BASE = "https://dolarapi.com/v1"
CRYPTOYA_DOLAR_URL = "https://criptoya.com/api/dolar"

# Rutas "canónicas" de DolarAPI
_DOLARAPI_PATHS: Dict[str, str] = {
    "oficial":   "/dolares/oficial",
    "mayorista": "/dolares/mayorista",          # algunos despliegues no lo tienen aquí
    "blue":      "/dolares/blue",
    "mep":       "/dolares/bolsa",              # 👈 clave: MEP es /bolsa
    "ccl":       "/dolares/contadoconliqui",
    "tarjeta":   "/dolares/tarjeta",
    "cripto":    "/ambito/dolares/cripto",      # a veces expuesto bajo /ambito
}

# Alternativas de rutas cuando las canónicas devuelven 404 (DolarAPI a veces reacomoda)
_DOLARAPI_ALTS: Dict[str, str] = {
    "mayorista": "/ambito/dolares/mayorista",
    # mantenemos "cripto" en /ambito; si el canónico funciona, no usamos este
}

def _normalize_row(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Normaliza el payload de DolarAPI a {compra: float|None, venta: float|None, fecha: str|None, fuente: str}
    """
    if not isinstance(data, dict):
        return None
    compra = data.get("compra")
    venta = data.get("venta")
    fecha = data.get("fechaActualizacion") or data.get("fecha")

    try:
        compra_f = float(compra) if compra is not None else None
    except Exception:
        compra_f = None
    try:
        venta_f = float(venta) if venta is not None else None
    except Exception:
        venta_f = None

    if compra_f is None and venta_f is None:
        # algunos endpoints devuelven solo venta; igual lo dejamos pasar
        pass

    return {"compra": compra_f, "venta": venta_f, "fecha": fecha, "fuente": "DolarAPI"}

async def _fetch_dolarapi(name: str) -> Optional[Dict[str, Any]]:
    """
    Intenta obtener un tipo de cambio desde DolarAPI probando la ruta principal
    y, si falla con 404, una alternativa (cuando aplica).
    """
    # 1) Ruta "canónica"
    path = _DOLARAPI_PATHS.get(name)
    if path:
        data = await http.fetch_json(f"{DOLARAPI_BASE}{path}")
        if data:
            row = _normalize_row(data)
            if row:
                return row
        else:
            log.warning("DolarAPI no disponible en %s (ruta canónica)", path)

    # 2) Fallback a ruta alternativa (si existe)
    alt = _DOLARAPI_ALTS.get(name)
    if alt:
        data_alt = await http.fetch_json(f"{DOLARAPI_BASE}{alt}")
        if data_alt:
            row = _normalize_row(data_alt)
            if row:
                return row
        else:
            log.warning("DolarAPI no disponible en %s (ruta alternativa)", alt)

    return None

async def _fetch_criptoya_all() -> Dict[str, Dict[str, Any]]:
    """
    Lee todos los tipos desde CriptoYa y arma un dict normalizado.
    Estructura objetivo: {tipo: {compra, venta, fuente, fecha?}}
    """
    out: Dict[str, Dict[str, Any]] = {}
    data = await http.fetch_json(CRYPTOYA_DOLAR_URL)
    if not data or not isinstance(data, dict):
        return out

    def _safe(block: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
        if not isinstance(block, dict):
            return (None, None)
        c = block.get("compra") or block.get("buy")
        v = block.get("venta") or block.get("sell")
        try:
            return (float(c) if c is not None else None, float(v) if v is not None else None)
        except Exception:
            return (None, None)

    for key_src, key_dst in [
        ("oficial", "oficial"),
        ("mayorista", "mayorista"),
        ("blue", "blue"),
        ("mep", "mep"),
        ("ccl", "ccl"),
        ("tarjeta", "tarjeta"),
        ("cripto", "cripto"),
    ]:
        c, v = _safe(data.get(key_src, {}))
        if c is not None or v is not None:
            out[key_dst] = {"compra": c, "venta": v, "fuente": "CriptoYa", "fecha": None}
    return out

async def fetch_all() -> Dict[str, Dict[str, Any]]:
    """
    Devuelve todas las cotizaciones relevantes con prioridad DolarAPI y fallback CriptoYa.
    Formato:
    {
      "oficial":  {"compra": float|None, "venta": float|None, "fecha": str|None, "fuente": "DolarAPI|CriptoYa"},
      "mayorista": {...},
      ...
    }
    """
    result: Dict[str, Dict[str, Any]] = {}
    # 1) Intento DolarAPI por cada tipo conocido
    for k in ("oficial", "mayorista", "blue", "mep", "ccl", "tarjeta", "cripto"):
        row = await _fetch_dolarapi(k)
        if row:
            result[k] = row

    # 2) Fallback con CriptoYa para lo que falte
    missing = [k for k in ("oficial", "mayorista", "blue", "mep", "ccl", "tarjeta", "cripto") if k not in result]
    if missing:
        cy = await _fetch_criptoya_all()
        for k in missing:
            if k in cy:
                result[k] = cy[k]

    return result

async def get_tc_value(kind: str, side: str = "venta") -> Optional[float]:
    """
    Devuelve el valor numérico del tipo de cambio pedido.
    kind in {"oficial","mayorista","blue","mep","ccl","tarjeta","cripto"}
    side in {"compra","venta"}  (por defecto "venta")
    """
    kind = (kind or "").lower().strip()
    side = "compra" if (side or "").lower().strip() == "compra" else "venta"
    data = await fetch_all()
    row = data.get(kind)
    if not row:
        return None
    val = row.get(side)
    try:
        return float(val) if val is not None else None
    except Exception:
        return None

# --- endpoints utilitarios (compat con código previo) ------------------------

async def fetch_oficial_blue() -> Dict[str, Dict[str, Any]]:
    """
    Devuelve solo oficial y blue (compatibilidad con llamadas antiguas).
    """
    data = await fetch_all()
    out: Dict[str, Dict[str, Any]] = {}
    for k in ("oficial", "blue"):
        if k in data:
            out[k] = data[k]
    return out

__all__ = [
    "fetch_all",
    "get_tc_value",
    "fetch_oficial_blue",
]
