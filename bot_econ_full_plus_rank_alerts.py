#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bot Económico AR — Telegram (Render Free, polling)
Comandos:
  /dolar
  /reservas
  /inflacion
  /riesgo
  /acciones
  /cedears
  /ranking_acciones
  /ranking_cedears
  /alerta_dolar <tipo> <umbral>
  /alertas
  /alerta_borrar <id>
  /resumen_diario

Fuentes:
  - Dólares: DolarAPI (principal) + fallback CriptoYa
  - Reservas / Inflación: apis.datos.gob.ar (series oficiales) [multi-intento]
  - Riesgo país: ArgentinaDatos (principal) + fallback DolarAPI
  - Acciones/CEDEARs: Yahoo Finance Spark (1d, 6mo) para armar 1m/3m/6m
  - Noticias: RSS medios locales + filtro económico (solo títulos clickeables)

Render:
  - Web service (Free). Incluye HTTP de salud en PORT (default 10000).
  - Polling de Telegram con JobQueue (prefetch periódico + alertas).
"""

import os
import re
import time
import math
import json
import asyncio
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from datetime import datetime, timedelta, timezone

import httpx
from xml.etree import ElementTree as ET

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters, Defaults
)

# ============== Config ===================

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
ALERTS_DB_PATH = os.getenv("ALERTS_DB_PATH", "/var/tmp/alerts.db")
NEWS_MAX_ITEMS = int(os.getenv("NEWS_MAX_ITEMS", "5"))

# Caches (TTL) en segundos
TTL_USD = 30          # tipos de cambio (rápido)
TTL_RIESGO = 300      # 5 min
TTL_RESERVAS = 3600   # 1 hora
TTL_IPC = 86400       # 24 hs
TTL_YF = 300          # 5 min
TTL_NEWS = 600        # 10 min

# Yahoo universos (podés ampliar libremente)
UNIVERSO_ACCIONES_BA = [
    # Bancos
    "GGAL.BA", "BMA.BA", "BBAR.BA", "SUPV.BA",
    # Energía
    "YPFD.BA", "PAMP.BA", "CEPU.BA", "EDN.BA", "TGSU2.BA", "TGNO4.BA", "TRAN.BA",
    # Industriales/Materiales
    "ALUA.BA", "TXAR.BA", "LOMA.BA", "CRES.BA",
    # Telecom/Tech
    "TECO2.BA", "MIRG.BA",
    # Consumo/Real estate/Otros
    "VALO.BA", "COME.BA", "HAVA.BA",
]

UNIVERSO_CEDEARS_BA = [
    "AAPL.BA", "AMZN.BA", "MSFT.BA", "GOOGL.BA", "META.BA", "TSLA.BA", "NVDA.BA",
    "MELI.BA", "BRKB.BA", "KO.BA", "PG.BA", "JNJ.BA", "PFE.BA", "WMT.BA", "DIS.BA",
    "NFLX.BA", "NKE.BA", "INTC.BA", "AMD.BA", "BIDU.BA"
]

# Sectores muy generales (para tarjeta). Podés ajustar mapping a gusto.
SECTORES = {
    "GGAL.BA": "Bancos", "BMA.BA": "Bancos", "BBAR.BA": "Bancos", "SUPV.BA": "Bancos",
    "YPFD.BA": "Energía", "PAMP.BA": "Energía", "CEPU.BA": "Energía", "EDN.BA": "Energía",
    "TGSU2.BA": "Energía", "TGNO4.BA": "Energía", "TRAN.BA": "Energía",
    "ALUA.BA": "Industriales", "TXAR.BA": "Industriales", "LOMA.BA": "Materiales",
    "CRES.BA": "Real Estate", "TECO2.BA": "Telecom", "MIRG.BA": "Servicios",
    "VALO.BA": "Finanzas", "COME.BA": "Consumo", "HAVA.BA": "Otros",
}

# Nombre “bonito” (si no está, mostramos el ticker)
NOMBRES = {
    "GGAL.BA": "Grupo Galicia", "BMA.BA": "Banco Macro", "BBAR.BA": "BBVA Arg", "SUPV.BA": "Supervielle",
    "YPFD.BA": "YPF", "PAMP.BA": "Pampa Energía", "CEPU.BA": "Central Puerto", "EDN.BA": "Edenor",
    "TGSU2.BA": "Transportadora Gas Sur", "TGNO4.BA": "Transportadora Gas Norte", "TRAN.BA": "Transener",
    "ALUA.BA": "Aluar", "TXAR.BA": "Ternium Arg", "LOMA.BA": "Loma Negra", "CRES.BA": "Cresud",
    "TECO2.BA": "Telecom Arg", "MIRG.BA": "Mirgor",
    "VALO.BA": "Grupo Valores", "COME.BA": "Siderar/Comercial", "HAVA.BA": "Havanna",
    "AAPL.BA": "Apple", "AMZN.BA": "Amazon", "MSFT.BA": "Microsoft", "GOOGL.BA": "Alphabet",
    "META.BA": "Meta", "TSLA.BA": "Tesla", "NVDA.BA": "NVIDIA", "MELI.BA": "MercadoLibre",
    "BRKB.BA": "Berkshire", "KO.BA": "Coca-Cola", "PG.BA": "P&G", "JNJ.BA": "J&J", "PFE.BA": "Pfizer",
    "WMT.BA": "Walmart", "DIS.BA": "Disney", "NFLX.BA": "Netflix", "NKE.BA": "Nike", "INTC.BA": "Intel",
    "AMD.BA": "AMD", "BIDU.BA": "Baidu"
}

# Series oficiales candidatos (multi-intento por si alguno falla/cambia)
SERIES_RESERVAS = [
    # Intentos (completá aquí el que usabas si lo conocés)
    "BCRA.RESERVAS_INTERNACIONALES_USD",
    "BCRA.RESERVAS_INTERNACIONALES",
]
SERIES_IPC_MOM = [
    # Variación mensual % (nivel general). Ajustá si conocés tu id exacto.
    "INDEC.IPC.NIVEL_GENERAL_Variacion_mensual",
    "INDEC.IPC.VAR_MENSUAL",
]

# RSS (medios locales, economía/finanzas)
RSS_FEEDS = [
    # Si alguna URL cambia, el bot la ignora silenciosamente.
    "https://www.ambito.com/rss/economia.xml",
    "https://www.cronista.com/files/rss/economia.xml",
    "https://www.baenegocios.com/rss/economia.xml",
    "https://www.infobae.com/feeds/rss/economia.xml",
    "https://www.telam.com.ar/rss2/economicas.xml",
    "https://www.iprofesional.com/rss/economia",
    "https://www.lanacion.com.ar/arc/outboundfeeds/rss/?outputType=xml&x=1&contentTypes=economia",
    "https://www.perfil.com/rss/economia.phtml",
]

KEYWORDS_OK = [
    "dólar", "dolar", "inflación", "inflacion", "reservas", "riesgo", "mercado",
    "acciones", "bonos", "tasa", "mev", "mep", "ccl", "pbi", "actividad", "export", "import",
    "ipim", "icm", "ipo", "inflacion", "brecha", "blue", "oficial", "mayorista"
]
KEYWORDS_BAN = [
    "salud", "fútbol", "futbol", "quiniela", "espectáculo", "espectaculo", "policial", "horóscopo", "horoscopo"
]

# ============== Utilidades comunes ===================

# Markdown V2: escapador para títulos y texto
_MDV2_RE = re.compile(r'([_*\[\]()~`>#+\-=|{}.!])')
def esc(s: str) -> str:
    return _MDV2_RE.sub(r'\\\1', s or "")

def now_utc() -> float:
    return time.time()

class TTLCache:
    def __init__(self):
        self.data = {}  # key -> (expire_ts, value)

    def get(self, key):
        item = self.data.get(key)
        if not item:
            return None
        exp, val = item
        if exp < now_utc():
            self.data.pop(key, None)
            return None
        return val

    def set(self, key, value, ttl: int):
        self.data[key] = (now_utc() + ttl, value)

CACHE = TTLCache()

# Cliente HTTP compartido (keep-alive + timeouts + UA)
HTTP_TIMEOUT = httpx.Timeout(connect=10, read=12, write=12, pool=12)
HTTP_LIMITS  = httpx.Limits(max_connections=20, max_keepalive_connections=20)
HTTP_HEADERS = {"User-Agent": "bot-econ-ar/1.0 (+github.com/Siberianok/bot-economico-ar)"}

http = httpx.AsyncClient(
    timeout=HTTP_TIMEOUT,
    limits=HTTP_LIMITS,
    headers=HTTP_HEADERS,
    follow_redirects=True,
)

async def fetch_json(url: str, params: dict | None = None):
    for i in range(2):  # dos intentos rápidos
        try:
            r = await http.get(url, params=params)
            if r.status_code == 200:
                return r.json()
        except httpx.RequestError:
            await asyncio.sleep(0.4 * (i + 1))
    return None

async def fetch_text(url: str, params: dict | None = None):
    for i in range(2):
        try:
            r = await http.get(url, params=params)
            if r.status_code == 200:
                return r.text
        except httpx.RequestError:
            await asyncio.sleep(0.4 * (i + 1))
    return None

# ============== Datos: Dólares ===================

USD_TYPES = ["blue", "mep", "ccl", "cripto", "oficial", "mayorista"]

async def get_dolares():
    cache_key = "usd_all"
    val = CACHE.get(cache_key)
    if val:
        return val
    out = {}

    # Principal: DolarAPI
    async def _dapi(tipo):
        url = f"https://dolarapi.com/v1/dolares/{tipo}"
        data = await fetch_json(url)
        if data and "compra" in data and "venta" in data:
            return {"buy": float(data["compra"]), "sell": float(data["venta"])}

    # Fallback: CriptoYa
    async def _cy(tipo):
        data = await fetch_json("https://criptoya.com/api/dolar")
        if not data:
            return None
        mapping = {
            "blue": "blue", "mep": "mep", "ccl": "ccl",
            "oficial": "oficial", "mayorista": "mayorista", "cripto": "cripto"
        }
        k = mapping.get(tipo)
        if k and k in data and isinstance(data[k], dict):
            bid = data[k].get("bid") or data[k].get("compra") or data[k].get("buy")
            ask = data[k].get("ask") or data[k].get("venta") or data[k].get("sell")
            if bid and ask:
                return {"buy": float(bid), "sell": float(ask)}
        return None

    for t in USD_TYPES:
        q = await _dapi(t)
        if not q:
            q = await _cy(t)
        if q:
            out[t] = q

    if out:
        CACHE.set(cache_key, out, TTL_USD)
    return out

def fmt_dolares_block(d):
    # bloque monoespaciado
    lines = [
        "BLUE      | Compra: $ {b1:>7,.2f} | Venta: $ {s1:>7,.2f}",
        "MEP       | Compra: $ {b2:>7,.2f} | Venta: $ {s2:>7,.2f}",
        "CCL       | Compra: $ {b3:>7,.2f} | Venta: $ {s3:>7,.2f}",
        "CRIPTO    | Compra: $ {b4:>7,.2f} | Venta: $ {s4:>7,.2f}",
        "OFICIAL   | Compra: $ {b5:>7,.2f} | Venta: $ {s5:>7,.2f}",
        "MAYORISTA | Compra: $ {b6:>7,.2f} | Venta: $ {s6:>7,.2f}",
    ]
    g = lambda t,k: d.get(t,{}).get(k, 0.0)
    s = "\n".join(lines).format(
        b1=g("blue","buy"),   s1=g("blue","sell"),
        b2=g("mep","buy"),    s2=g("mep","sell"),
        b3=g("ccl","buy"),    s3=g("ccl","sell"),
        b4=g("cripto","buy"), s4=g("cripto","sell"),
        b5=g("oficial","buy"),s5=g("oficial","sell"),
        b6=g("mayorista","buy"), s6=g("mayorista","sell"),
    )
    return "```\n" + s + "\n```"

# ============== Datos: Series oficiales (Reservas / IPC) ===================

async def series_last(ids: str):
    data = await fetch_json(
        "https://apis.datos.gob.ar/series/api/series",
        {"ids": ids, "limit": 1, "sort": "desc"},
    )
    try:
        if data and "data" in data and data["data"] and len(data["data"][0]) >= 2:
            return float(data["data"][0][1])
    except Exception:
        pass
    return None

async def get_reservas():
    cache_key = "reservas"
    v = CACHE.get(cache_key)
    if v is not None:
        return v
    val = None
    for sid in SERIES_RESERVAS:
        val = await series_last(sid)
        if val is not None:
            break
    if val is not None:
        CACHE.set(cache_key, val, TTL_RESERVAS)
    return val

async def get_ipc_mom():
    cache_key = "ipc_mom"
    v = CACHE.get(cache_key)
    if v is not None:
        return v
    val = None
    for sid in SERIES_IPC_MOM:
        val = await series_last(sid)
        if val is not None:
            break
    if val is not None:
        CACHE.set(cache_key, val, TTL_IPC)
    return val

# ============== Datos: Riesgo País ===================

async def get_riesgo_pais():
    cache_key = "riesgo"
    v = CACHE.get(cache_key)
    if v is not None:
        return v
    # Principal: ArgentinaDatos
    data = await fetch_json("https://api.argentinadatos.com/v1/finanzas/mercados/riesgo-pais")
    val = None
    try:
        # Ejemplo esperado: {"fecha":"YYYY-MM-DD","valor": XXX}
        if data:
            if isinstance(data, dict) and "valor" in data:
                val = int(round(float(data["valor"])))
            elif isinstance(data, list) and data:
                # tomar el último elemento que tenga valor
                for row in reversed(data):
                    if "valor" in row:
                        val = int(round(float(row["valor"])))
                        break
    except Exception:
        val = None

    # Fallback: DolarAPI (si existiese)
    if val is None:
        dapi = await fetch_json("https://dolarapi.com/v1/otros/riesgo-pais")
        if dapi and "valor" in dapi:
            try:
                val = int(round(float(dapi["valor"])))
            except Exception:
                val = None

    if val is not None:
        CACHE.set(cache_key, val, TTL_RIESGO)
    return val

# ============== Datos: Yahoo Finance ===================

def _pct(a: float, b: float) -> float:
    if a is None or b is None or a == 0:
        return 0.0
    return (b - a) / a * 100.0

async def yf_spark(symbols: list[str]):
    """
    Devuelve dict[symbol] = {"close": [...]} con 6 meses diarios.
    """
    out = {}
    if not symbols:
        return out
    # Yahoo Spark batch
    syms = ",".join(symbols)
    url = "https://query1.finance.yahoo.com/v7/finance/spark"
    data = await fetch_json(url, {"symbols": syms, "range": "6mo", "interval": "1d"})
    if not data or "spark" not in data or "result" not in data["spark"]:
        return out
    for item in data["spark"]["result"]:
        sym = item.get("symbol")
        close = item.get("response", [{}])[0].get("indicators", {}).get("quote", [{}])[0].get("close")
        if sym and isinstance(close, list):
            # filtrar None
            close = [c for c in close if c is not None]
            if len(close) >= 5:
                out[sym] = {"close": close}
    return out

def perf_1m_3m_6m(closes: list[float]):
    """
    Aprox: 1m=21 días hábiles, 3m=63, 6m=126
    """
    if not closes:
        return 0.0,0.0,0.0
    last = closes[-1]
    def get(idx_from_end):
        idx = max(0, len(closes) - idx_from_end - 1)
        return closes[idx]
    m1 = _pct(get(21), last)
    m3 = _pct(get(63), last)
    m6 = _pct(get(126), last)
    return m1, m3, m6

async def top_universo(symbols: list[str], topn=3):
    cache_key = f"yf_{hash(tuple(symbols))}"
    v = CACHE.get(cache_key)
    if v:
        return v
    data = await yf_spark(symbols)
    cards = []
    for sym, d in data.items():
        closes = d.get("close", [])
        if not closes:
            continue
        m1, m3, m6 = perf_1m_3m_6m(closes)
        price = closes[-1]
        nm = NOMBRES.get(sym, sym.replace(".BA",""))
        sector = SECTORES.get(sym, "—")
        cards.append((sym, nm, price, sector, m1, m3, m6))

    # orden por 3m desc
    cards.sort(key=lambda x: x[5], reverse=True)
    top = cards[:topn]
    CACHE.set(cache_key, top, TTL_YF)
    return top

async def ranking_6m(symbols: list[str], topn=5):
    # score = 0.1*1m + 0.3*3m + 0.6*6m
    data = await top_universo(symbols, topn=len(symbols))
    scored = []
    for (sym, nm, price, sector, m1, m3, m6) in data:
        score = 0.1*m1 + 0.3*m3 + 0.6*m6
        scored.append((score, sym, nm, price, sector, m1, m3, m6))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:topn]

def fmt_card(sym, nm, price, sector, m1, m3, m6):
    # Ticker (Empresa) — Precio / • Sector / Rendimientos: 1m · 3m · 6m
    # Usamos Markdown V2 -> escapamos
    t = esc(sym.replace(".BA",""))
    n = esc(nm)
    p = f"{price:,.2f}"
    s = esc(sector)
    def sign(x): return ("+" if x>=0 else "") + f"{x:,.2f}%"
    return f"*{t}* \\({n}\\) — ${p}\n• {s}\nRendimientos: {esc(sign(m1))} · {esc(sign(m3))} · {esc(sign(m6))}"

# ============== Noticias (RSS) ===================

def _pass_noticia(title: str):
    t = (title or "").lower()
    if any(b in t for b in KEYWORDS_BAN):
        return False
    if any(k in t for k in KEYWORDS_OK):
        return True
    # si no matchea nada, lo ignoramos
    return False

async def get_news():
    cache_key = "news"
    v = CACHE.get(cache_key)
    if v:
        return v
    items = []
    for url in RSS_FEEDS:
        txt = await fetch_text(url)
        if not txt:
            continue
        try:
            root = ET.fromstring(txt)
            for it in root.findall(".//item"):
                title = (it.findtext("title") or "").strip()
                link = (it.findtext("link") or "").strip()
                if title and link and link.startswith("http") and _pass_noticia(title):
                    items.append((title, link))
        except ET.ParseError:
            continue
    # dedup por título
    seen = set()
    uniq = []
    for t, u in items:
        k = t.strip().lower()
        if k in seen:
            continue
        seen.add(k)
        uniq.append((t, u))
    out = uniq[:NEWS_MAX_ITEMS]
    CACHE.set(cache_key, out, TTL_NEWS)
    return out

def fmt_news_links(items):
    # links Markdown V2 con títulos escapados (sin preview, por Defaults)
    if not items:
        return "Sin noticias filtradas ahora."
    lines = []
    for t, u in items:
        lines.append(f"• [{esc(t)}]({u})")
    return "\n".join(lines)

# ============== Alertas de dólar (sqlite) ===================

def db_init():
    os.makedirs(os.path.dirname(ALERTS_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(ALERTS_DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            tipo TEXT NOT NULL,
            umbral REAL NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def db_add_alert(chat_id: int, tipo: str, umbral: float):
    conn = sqlite3.connect(ALERTS_DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO alerts (chat_id,tipo,umbral) VALUES (?,?,?)", (chat_id, tipo, umbral))
    conn.commit()
    conn.close()

def db_list_alerts(chat_id: int):
    conn = sqlite3.connect(ALERTS_DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id,tipo,umbral FROM alerts WHERE chat_id=?", (chat_id,))
    rows = c.fetchall()
    conn.close()
    return rows

def db_del_alert(chat_id: int, alert_id: int):
    conn = sqlite3.connect(ALERTS_DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM alerts WHERE chat_id=? AND id=?", (chat_id, alert_id))
    conn.commit()
    ch = c.rowcount
    conn.close()
    return ch > 0

async def job_check_alerts(context: ContextTypes.DEFAULT_TYPE):
    usd = await get_dolares()
    if not usd:
        return
    conn = sqlite3.connect(ALERTS_DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id,chat_id,tipo,umbral FROM alerts")
    rows = c.fetchall()
    conn.close()
    for alert_id, chat_id, tipo, umbral in rows:
        q = usd.get(tipo)
        if not q:
            continue
        precio = q.get("sell") or q.get("buy")
        if not precio:
            continue
        if float(precio) >= float(umbral):
            try:
                txt = f"🔔 *Alerta dólar {esc(tipo.upper())}*\nPrecio: ${precio:,.2f} ≥ umbral ${umbral:,.2f}"
                await context.bot.send_message(chat_id=chat_id, text=txt)
                # borrar alerta dispadara
                conn = sqlite3.connect(ALERTS_DB_PATH)
                c = conn.cursor()
                c.execute("DELETE FROM alerts WHERE id=?", (alert_id,))
                conn.commit()
                conn.close()
            except Exception:
                pass

# ============== Prefetch periódico ===================

async def job_prefetch(context: ContextTypes.DEFAULT_TYPE):
    try:
        await get_dolares()
    except Exception:
        pass
    try:
        await get_reservas()
    except Exception:
        pass
    try:
        await get_ipc_mom()
    except Exception:
        pass
    try:
        await get_riesgo_pais()
    except Exception:
        pass
    try:
        await top_universo(UNIVERSO_ACCIONES_BA, topn=6)
        await top_universo(UNIVERSO_CEDEARS_BA, topn=6)
    except Exception:
        pass
    try:
        await get_news()
    except Exception:
        pass

# ============== Handlers ===================

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = (
        "👋 *Observatorio Económico AR*\n\n"
        "Comandos: /dolar /reservas /inflacion /riesgo /acciones /cedears\n"
        "/ranking_acciones /ranking_cedears\n"
        "/alerta_dolar <tipo> <umbral>  (blue|mep|ccl|cripto|oficial|mayorista)\n"
        "/alertas  /alerta_borrar <id>\n"
        "/resumen_diario"
    )
    await update.message.reply_text(txt)

async def cmd_dolar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    d = await get_dolares()
    if not d:
        await update.message.reply_text("No pude obtener cotizaciones ahora.")
        return
    block = fmt_dolares_block(d)
    await update.message.reply_text("💵 *Dólares*\n" + block)

async def cmd_reservas(update: Update, context: ContextTypes.DEFAULT_TYPE):
    v = await get_reservas()
    if v is None:
        await update.message.reply_text("💹 *Reservas BCRA*: dato no disponible ahora (fuente oficial).")
    else:
        await update.message.reply_text(f"💹 *Reservas BCRA*: USD {v:,.0f}")

async def cmd_inflacion(update: Update, context: ContextTypes.DEFAULT_TYPE):
    v = await get_ipc_mom()
    if v is None:
        await update.message.reply_text("📈 *Inflación (variación mensual)*: dato no disponible.")
    else:
        await update.message.reply_text(f"📈 *Inflación (variación mensual)*: {v:,.2f}%")

async def cmd_riesgo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    v = await get_riesgo_pais()
    if v is None:
        await update.message.reply_text("📉 *Riesgo país*: dato no disponible.")
    else:
        await update.message.reply_text(f"📉 *Riesgo país*: {v:,} pb")

async def cmd_acciones(update: Update, context: ContextTypes.DEFAULT_TYPE):
    top = await top_universo(UNIVERSO_ACCIONES_BA, topn=3)
    if not top:
        await update.message.reply_text("📊 *Acciones BYMA*: no hay datos ahora.")
        return
    lines = []
    for (sym, nm, price, sector, m1, m3, m6) in top:
        lines.append("🪪 " + fmt_card(sym, nm, price, sector, m1, m3, m6))
    await update.message.reply_text("📊 *Top 3 Acciones BYMA (3m)*\n\n" + "\n\n".join(lines))

async def cmd_cedears(update: Update, context: ContextTypes.DEFAULT_TYPE):
    top = await top_universo(UNIVERSO_CEDEARS_BA, topn=3)
    if not top:
        await update.message.reply_text("🌐 *CEDEARs*: no hay datos ahora.")
        return
    lines = []
    for (sym, nm, price, sector, m1, m3, m6) in top:
        lines.append("🪪 " + fmt_card(sym, nm, price, sector, m1, m3, m6))
    await update.message.reply_text("🌐 *Top 3 CEDEARs (3m)*\n\n" + "\n\n".join(lines))

async def cmd_rank_acc(update: Update, context: ContextTypes.DEFAULT_TYPE):
    rk = await ranking_6m(UNIVERSO_ACCIONES_BA, topn=5)
    if not rk:
        await update.message.reply_text("🏆 *Ranking Acciones BYMA*: no hay datos ahora.")
        return
    lines = []
    for (score, sym, nm, price, sector, m1, m3, m6) in rk:
        lines.append("🏅 " + fmt_card(sym, nm, price, sector, m1, m3, m6))
    await update.message.reply_text("🏆 *Ranking 6M Acciones BYMA*\n\n" + "\n\n".join(lines))

async def cmd_rank_ced(update: Update, context: ContextTypes.DEFAULT_TYPE):
    rk = await ranking_6m(UNIVERSO_CEDEARS_BA, topn=5)
    if not rk:
        await update.message.reply_text("🏆 *Ranking CEDEARs*: no hay datos ahora.")
        return
    lines = []
    for (score, sym, nm, price, sector, m1, m3, m6) in rk:
        lines.append("🏅 " + fmt_card(sym, nm, price, sector, m1, m3, m6))
    await update.message.reply_text("🏆 *Ranking 6M CEDEARs*\n\n" + "\n\n".join(lines))

async def cmd_alerta_dolar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # /alerta_dolar <tipo> <umbral>
    if len(context.args) != 2:
        await update.message.reply_text("Uso: /alerta_dolar <tipo> <umbral>\nEj: /alerta_dolar blue 1500")
        return
    tipo = context.args[0].lower().strip()
    if tipo not in USD_TYPES:
        await update.message.reply_text("Tipos válidos: blue|mep|ccl|cripto|oficial|mayorista")
        return
    try:
        umbral = float(context.args[1].replace(",", "."))
    except ValueError:
        await update.message.reply_text("Umbral inválido. Ej: 1500")
        return
    db_add_alert(update.effective_chat.id, tipo, umbral)
    await update.message.reply_text(f"✅ Alerta creada: {tipo.upper()} ≥ ${umbral:,.2f}")

async def cmd_alertas(update: Update, context: ContextTypes.DEFAULT_TYPE):
    rows = db_list_alerts(update.effective_chat.id)
    if not rows:
        await update.message.reply_text("No tenés alertas activas.")
        return
    lines = [f"#{i} — {esc(t.upper())} ≥ ${u:,.2f}" for (i,t,u) in rows]
    await update.message.reply_text("*Alertas activas*\n" + "\n".join(lines))

async def cmd_alerta_borrar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Uso: /alerta_borrar <id>")
        return
    try:
        a_id = int(context.args[0])
    except ValueError:
        await update.message.reply_text("ID inválido.")
        return
    ok = db_del_alert(update.effective_chat.id, a_id)
    await update.message.reply_text("🗑️ Alerta eliminada." if ok else "No encontré esa alerta.")

async def cmd_resumen_diario(update: Update, context: ContextTypes.DEFAULT_TYPE):
    usd = await get_dolares()
    reservas = await get_reservas()
    ipc = await get_ipc_mom()
    riesgo = await get_riesgo_pais()
    news = await get_news()

    lines = [f"🗓️ *Resumen {datetime.now().strftime('%Y-%m-%d %H:%M')}*"]
    if usd:
        lines.append("💵 *Dólares*")
        lines.append(fmt_dolares_block(usd))
    else:
        lines.append("💵 *Dólares*: no disponible")

    if reservas is None:
        lines.append("💹 *Reservas*: Dato no disponible")
    else:
        lines.append(f"💹 *Reservas*: USD {reservas:,.0f}")

    if ipc is None:
        lines.append("📈 *Inflación*: Dato no disponible")
    else:
        lines.append(f"📈 *Inflación*: {ipc:,.2f}%")

    if riesgo is None:
        lines.append("📉 *Riesgo País*: Dato no disponible")
    else:
        lines.append(f"📉 *Riesgo País*: {riesgo:,} pb")

    lines.append("\n📰 *Noticias*")
    lines.append(fmt_news_links(news))

    await update.message.reply_text("\n".join(lines))

# ============== HTTP de salud (Render Web Service) ===================

class _Health(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type","text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"ok")
    def log_message(self, *args, **kwargs):
        pass

def start_health_server():
    port = int(os.getenv("PORT", "10000"))
    server = HTTPServer(("0.0.0.0", port), _Health)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

# ============== Main ===================

def build_app():
    if not BOT_TOKEN:
        raise RuntimeError("Falta BOT_TOKEN en variables de entorno.")

    # MarkdownV2 y sin previews para links limpios
    defaults = Defaults(parse_mode=ParseMode.MARKDOWN_V2, disable_web_page_preview=True)
    app = ApplicationBuilder().token(BOT_TOKEN).defaults(defaults).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("dolar", cmd_dolar))
    app.add_handler(CommandHandler("reservas", cmd_reservas))
    app.add_handler(CommandHandler("inflacion", cmd_inflacion))
    app.add_handler(CommandHandler("riesgo", cmd_riesgo))
    app.add_handler(CommandHandler("acciones", cmd_acciones))
    app.add_handler(CommandHandler("cedears", cmd_cedears))
    app.add_handler(CommandHandler("ranking_acciones", cmd_rank_acc))
    app.add_handler(CommandHandler("ranking_cedears", cmd_rank_ced))
    app.add_handler(CommandHandler("alerta_dolar", cmd_alerta_dolar))
    app.add_handler(CommandHandler("alertas", cmd_alertas))
    app.add_handler(CommandHandler("alerta_borrar", cmd_alerta_borrar))
    app.add_handler(CommandHandler("resumen_diario", cmd_resumen_diario))
    # ignoramos todo lo demás
    app.add_handler(MessageHandler(filters.COMMAND, cmd_start))

    # Jobs programados
    if app.job_queue:
        app.job_queue.run_repeating(job_prefetch, interval=300, first=5)   # cada 5 min
        app.job_queue.run_repeating(job_check_alerts, interval=90, first=15)

    return app

def main():
    db_init()
    start_health_server()
    app = build_app()
    # Polling robusto
    app.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,
        close_loop=False,
        stop_signals=None,
    )

if __name__ == "__main__":
    main()
