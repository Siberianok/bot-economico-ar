# bot_econ/data_sources/reserves.py
# -*- coding: utf-8 -*-
from typing import Optional, Tuple
import re
from . import http

LAMACRO_RESERVAS_URL = "https://www.lamacro.ar/variables/1"

async def fetch_reserves() -> Optional[Tuple[float, Optional[str]]]:
    """
    Devuelve (valor_en_MUS$, fecha_dd/mm/aaaa) o None si falla.
    Parsea la página HTML de LaMacro.
    """
    html = await http.fetch_text(LAMACRO_RESERVAS_URL)
    if not html:
        return None

    # Ejemplos a cubrir:
    # "Último dato : 28.123,4" o "Valor actual : 28.123,4"
    m_val = re.search(r"(?:Último dato|Valor actual)\s*:\s*([0-9\.\,]+)", html)
    if not m_val:
        return None

    raw = m_val.group(1).replace(".", "").replace(",", ".")
    try:
        val = float(raw)
    except Exception:
        return None

    m_date = re.search(r"([0-3]\d/[0-1]\d/\d{4})", html)
    fecha = m_date.group(1) if m_date else None
    return (val, fecha)
