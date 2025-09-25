# bot_econ/data_sources/reserves.py
# -*- coding: utf-8 -*-

import re
from typing import Dict, Any, Optional
from . import http

# LaMacro muestra HTML; no devuelve JSON.
LAMACRO_RESERVAS_URL = "https://www.lamacro.ar/variables/1"

def _parse_number_es(s: str) -> Optional[float]:
    """
    Convierte '123.456,78' -> 123456.78 ; '1.234' -> 1234.0
    """
    try:
        s2 = s.replace(".", "").replace(",", ".")
        return float(s2)
    except Exception:
        return None

async def fetch_reserves() -> Dict[str, Any]:
    """
    Devuelve:
      {
        'valor': float | None,   # en MUS$ (millones de USD)
        'fecha': str  | None,    # dd/mm/yyyy
        'fuente': 'LaMacro'
      }
    """
    html = await http.fetch_text(LAMACRO_RESERVAS_URL)
    if not html:
        return {"valor": None, "fecha": None, "fuente": "LaMacro"}

    # Busco el valor (ej: "Último dato : 24.567,0" o "Valor actual: 23.456")
    m_val = re.search(r"(?:Último dato|Valor actual)\s*:\s*([0-9\.\,]+)", html, flags=re.I)
    # Busco una fecha dd/mm/yyyy
    m_date = re.search(r"([0-3]\d/[0-1]\d/\d{4})", html)

    val = _parse_number_es(m_val.group(1)) if m_val else None
    fecha = m_date.group(1) if m_date else None

    return {"valor": val, "fecha": fecha, "fuente": "LaMacro"}
