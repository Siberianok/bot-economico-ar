# bot_econ_full_plus_rank_alerts.py
# -*- coding: utf-8 -*-
#
# Bot Económico AR (Render webhook)
# Comandos:
#   /dolar /acciones /cedears
#   /rankings_acciones /rankings_cedears
#   /reservas /inflacion /riesgo /resumen_diario
#   /alertas /alertas_add /alertas_clear
#
import os, asyncio, logging, re, html as _html
from time import time
from math import sqrt
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, List, Tuple, Any, Optional
from urllib.parse import urlparse

from aiohttp import web, ClientSession, ClientTimeout
from telegram import (
    Update, LinkPreviewOptions, BotCommand,
    InlineKeyboardButton, InlineKeyboardMarkup
)
from telegram.constants import ParseMode
from telegram.ext import (
    Application, CommandHandler, ContextTypes, Defaults,
    ConversationHandler, CallbackQueryHandler, MessageHandler, filters
)

# ---------------- Config ----------------
TZ = ZoneInfo("America/Argentina/Buenos_Aires")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "tgwebhook").strip().strip("/")
PORT = int(os.getenv("PORT", "10000"))
BASE_URL = os.getenv("BASE_URL", os.getenv("RENDER_EXTERNAL_URL", "https://bot-economico-ar.onrender.com")).rstrip("/")
if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN no configurado.")
WEBHOOK_PATH = f"/{WEBHOOK_SECRET}"
WEBHOOK_URL = f"{BASE_URL}{WEBHOOK_PATH}"

CRYPTOYA_DOLAR_URL = "https://criptoya.com/api/dolar"
DOLARAPI_BASE = "https://dolarapi.com/v1"
ARG_DATOS_BASES = [
    "https://api.argentinadatos.com/v1/finanzas/indices",
    "https://argentinadatos.com/v1/finanzas/indices",
]
LAMACRO_RESERVAS_URL = "https://www.lamacro.ar/variables/1"

YF_URLS = [
    "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
    "https://query2.finance.yahoo.com/v8/finance/chart/{symbol}",
]
YF_HEADERS = {"User-Agent": "Mozilla/5.0"}

# RSS nacionales (incluye Clarín, La Nación, Cronista, Página12, etc.)
RSS_FEEDS = [
    "https://www.ambito.com/contenidos/economia.xml",
    "https://www.iprofesional.com/rss",
    "https://www.infobae.com/economia/rss",
    "https://www.perfil.com/rss/economia.xml",
    "https://www.baenegocios.com/rss/economia.xml",
    "https://www.telam.com.ar/rss2/economia.xml",
    "https://www.clarin.com/rss/economia/",
    "https://www.lanacion.com.ar/economia/rss/",
    "https://www.cronista.com/files/rss/economia.xml",
    "https://www.cronista.com/files/rss/news.xml",
    "https://www.pagina12.com.ar/rss/economia",
    "https://eleconomista.com.ar/rss",
]
# No filtramos dominios ahora
AVOID_DOMAINS: set = set()

ACCIONES_BA = ["GGAL.BA","YPFD.BA","PAMP.BA","CEPU.BA","ALUA.BA","TXAR.BA","TGSU2.BA","BYMA.BA","SUPV.BA","BMA.BA"]
CEDEARS_BA  = ["AAPL.BA","MSFT.BA","NVDA.BA","AMZN.BA","GOOGL.BA","TSLA.BA","META.BA","JNJ.BA","KO.BA","NFLX.BA"]

# ------------- Logging -------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("bot-econ-ar")

# ------------- Utils -------------
def fmt_number(n: Optional[float], nd=2) -> str:
    try:
        if n is None: return "—"
        s = f"{n:,.{nd}f}"
        return s.replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return str(n)

def fmt_money_ars(n: Optional[float]) -> str:
    return f"$ {fmt_number(n, 2)}"

def pct(n: Optional[float], nd: int = 2) -> str:
    try:
        return f"{n:+.{nd}f}%".replace(".", ",")
    except Exception:
        return "—"

def anchor(href: str, text: str) -> str:
    return f'<a href="{_html.escape(href, quote=True)}">{_html.escape(text)}</a>'

def html_op(op: str) -> str:
    return "&gt;" if op == ">" else "&lt;"

def fmt_fecha_ddmmyyyy_from_iso(s: Optional[str]) -> Optional[str]:
    if not s: return None
    try:
        if re.match(r"^\d{4}-\d{2}-\d{2}", s):
            return datetime.strptime(s[:10], "%Y-%m-%d").strftime("%d/%m/%Y")
    except Exception:
        pass
    return s

def last_day_of_month_str(periodo_yyyy_mm: str) -> Optional[str]:
    try:
        y = int(periodo_yyyy_mm[0:4]); m = int(periodo_yyyy_mm[5:7])
        d = (datetime(y, 12, 31) if m==12 else datetime(y, m+1, 1) - timedelta(days=1))
        return d.strftime("%d/%m/%Y")
    except Exception:
        return None

def parse_period_to_ddmmyyyy(per: Optional[str]) -> Optional[str]:
    if not per: return None
    if re.match(r"^\d{4}-\d{2}-\d{2}$", per): return fmt_fecha_ddmmyyyy_from_iso(per)
    if re.match(r"^\d{4}-\d{2}$", per): return last_day_of_month_str(per)
    return per

# ------------- HTTP -------------
REQ_HEADERS = {"User-Agent":"Mozilla/5.0", "Accept":"*/*"}
async def fetch_json(session: ClientSession, url: str, **kwargs) -> Optional[Dict[str, Any]]:
    try:
        timeout = kwargs.pop("timeout", ClientTimeout(total=15))
        headers = kwargs.pop("headers", {})
        async with session.get(url, timeout=timeout, headers={**REQ_HEADERS, **headers}, **kwargs) as resp:
            if resp.status == 200:
                return await resp.json(content_type=None)
            log.warning("GET %s -> %s", url, resp.status)
    except Exception as e:
        log.warning("fetch_json error %s: %s", url, e)
    return None

async def fetch_text(session: ClientSession, url: str, **kwargs) -> Optional[str]:
    try:
        timeout = kwargs.pop("timeout", ClientTimeout(total=15))
        headers = kwargs.pop("headers", {})
        async with session.get(url, timeout=timeout, headers={**REQ_HEADERS, **headers}, **kwargs) as resp:
            if resp.status == 200:
                return await resp.text()
            log.warning("GET %s -> %s", url, resp.status)
    except Exception as e:
        log.warning("fetch_text error %s: %s", url, e)
    return None

# ------------- Dólares -------------
async def get_dolares(session: ClientSession) -> Dict[str, Dict[str, Any]]:
    data: Dict[str, Dict[str, Any]] = {}
    cj = await fetch_json(session, CR
