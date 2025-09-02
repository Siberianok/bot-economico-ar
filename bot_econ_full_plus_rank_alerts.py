#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bot Económico AR — Telegram (Render Free, Web Service)

Mejoras:
- Concurrencia (httpx + asyncio.Semaphore) para bajar latencia.
- Caché con TTL por sección (respuestas casi instantáneas tras primer hit).
- Jobs opcionales (si hay JobQueue) para refrescar cachés cada 5 min.
- Health-check HTTP /healthz en thread (sin asyncio).
- Series oficiales ajustadas (IPC mensual) y fallbacks robustos.
"""

import os
import re
import math
import time
import sqlite3
import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Optional, Dict, Any, List, Tuple

import httpx
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# ---------- Config ----------
TZ = ZoneInfo("America/Argentina/Buenos_Aires")

# Dólares
DOLARAPI_BASE = "https://dolarapi.com/v1/ambito/dolares"
CRIPTOYA_DOLAR = "https://criptoya.com/api/dolar"
DOLAR_TIPOS = {
    "blue": f"{DOLARAPI_BASE}/blue",
    "mep": f"{DOLARAPI_BASE}/bolsa",
    "ccl": f"{DOLARAPI_BASE}/contadoconliqui",
    "cripto": f"{DOLARAPI_BASE}/cripto",
    "oficial": f"{DOLARAPI_BASE}/oficial",
    "mayorista": f"{DOLARAPI_BASE}/mayorista",
}

# Series oficiales (INDEC / datos.gob.ar)
DATOS_API = "https://apis.datos.gob.ar/series/api/series"
# IPC mensual (nivel general, tasa de variación) — base dic-2016:
IPC_MENSUAL_SERIES_VARIACION = [
    "145.3_INGNACUAL_DICI_M_38",      # (Primaria) IPC, tasa variación mensual, nivel general
    "103.1_I2N_2016_M_19",            # (Fallback) variación % (representation_mode)
]
IPC_INDICE_SERIES = [
    "103.1_I2N_2016_M_15",            # Índice IPC, por si hace falta percent_change
]
# Reservas internacionales (candidatas, la 92.2_* suele funcionar)
RESERVAS_SERIES_CANDIDATAS = [
    "92.2_RESERVAS_IRES_0_0_32_40",
    "92.1_RESERVAS_0_0_26",
    "92.1_RESERVAS_D_0_0_18",
]

# Riesgo país (ArgentinaDatos)
ARGDATOS_RIESGO_ULTIMO = "https://api.argentinadatos.com/v1/finanzas/indices/riesgo-pais/ultimo"
ARGDATOS_RIESGO_LISTA  = "https://api.argentinadatos.com/v1/finanzas/indices/riesgo-pais"

# Yahoo Finance
YF_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

# RSS económicos locales
RSS_FEEDS = [
    "https://www.ambito.com/contenidos/economia.xml",
    "https://www.cronista.com/files/rss/economia.xml",
    "https://www.baenegocios.com/rss",
    "https://www.infobae.com/feeds/rss/economia.xml",
    "https://www.perfil.com/feed/economia",
    "https://www.lanacion.com.ar/economia/rss/",
    "https://www.telam.com.ar/rss2/economia.xml",
    "https://www.iprofesional.com/rss/economia",
]
NEWS_KEYWORDS_POS = [
    "dólar","dolar","mep","ccl","reservas","inflación","ipc","riesgo país","riesgo-pais",
    "bcra","bono","bonos","acciones","cedear","merval","mercado","pbi","exportaciones",
    "tasa","liquidez","deuda","fmi","brecha","superávit","déficit","blue","cambio","oficial",
]
NEWS_KEYWORDS_NEG = ["quiniela","horóscopo","salud","covid","farandula","espectáculo","policiales"]

# Universos iniciales (ampliables)
BYMA_UNIVERSO = [
    ("GGAL.BA", "Grupo Financiero Galicia", "Bancos"),
    ("BMA.BA", "Banco Macro", "Bancos"),
    ("BBAR.BA", "BBVA Argentina", "Bancos"),
    ("SUPV.BA", "Supervielle", "Bancos"),
    ("VALO.BA", "Grupo Valores", "Servicios Financieros"),
    ("BYMA.BA", "Bolsas y Mercados Argentinos", "Servicios Financieros"),
    ("YPFD.BA", "YPF", "Energía"),
    ("PAMP.BA", "Pampa Energía", "Energía"),
    ("CEPU.BA", "Central Puerto", "Energía"),
    ("TGSU2.BA", "Transportadora de Gas del Sur", "Energía"),
    ("TGNO4.BA", "Transportadora de Gas del Norte", "Energía"),
    ("TXAR.BA", "Ternium Argentina", "Materiales"),
    ("ALUA.BA", "Aluar", "Materiales"),
    ("LOMA.BA", "Loma Negra", "Materiales"),
    ("COME.BA", "Soc. Comercial del Plata", "Industriales"),
    ("TECO2.BA", "Telecom Argentina", "Telecom"),
    ("CVH.BA", "Cablevisión Holding", "Medios/Telecom"),
    ("CRES.BA", "Cresud", "Agro/Real Estate"),
    ("IRSA.BA", "IRSA", "Real Estate"),
    ("IRCP.BA", "IRSA Prop. Comerciales", "Real Estate"),
]
CEDEARS_UNIVERSO = [
    ("AAPL.BA", "Apple", "Tecnología"),
    ("MSFT.BA", "Microsoft", "Tecnología"),
    ("GOOGL.BA", "Alphabet A", "Tecnología"),
    ("AMZN.BA", "Amazon", "Tecnología"),
    ("NVDA.BA", "NVIDIA", "Tecnología"),
    ("META.BA", "Meta Platforms", "Tecnología"),
    ("TSLA.BA", "Tesla", "Autos/Tech"),
    ("BRKB.BA", "Berkshire Hathaway B", "Finanzas"),
    ("JPM.BA", "JPMorgan", "Finanzas"),
    ("KO.BA", "Coca-Cola", "Consumo"),
    ("PEP.BA", "PepsiCo", "Consumo"),
    ("MELI.BA", "MercadoLibre (ADR)", "Tecnología"),
    ("MSTR.BA", "MicroStrategy", "Tecnología"),
]

# ---------- Utilidades ----------
def fmt_money_ars(x: float) -> str:
    try:
        s = f"{x:,.2f}"
        s = s.replace(",", "_").replace(".", ",").replace("_", ".")
        return f"$ {s}"
    except Exception:
        return str(x)

def now_str() -> str:
    return datetime.now(TZ).strftime("%Y-%m-%d %H:%M")

def pct(a: float, b: float) -> float:
    try:
        return (a/b - 1.0) * 100.0
    except Exception:
        return float('nan')

def human_pct(x: float) -> str:
    if x != x or math.isinf(x):
        return "n/d"
    return f"{x:+.2f}%"

# ---------- HTTP (concurrencia y reintentos) ----------
_HTTP_LIMIT = int(os.environ.get("HTTP_CONCURRENCY", "8"))
_SEM = asyncio.Semaphore(_HTTP_LIMIT)

_CLIENT = httpx.AsyncClient(
    timeout=httpx.Timeout(12.0, connect=8.0),
    headers={"User-Agent": "AR-EconBot/1.1"}
)

async def _get_json(url: str, params: Optional[dict] = None, retries=2):
    for i in range(retries + 1):
        try:
            async with _SEM:
                r = await _CLIENT.get(url, params=params)
            r.raise_for_status()
            return r.json()
        except Exception:
            if i == retries:
                return None
            await asyncio.sleep(0.6 * (i + 1))

async def _get_text(url: str, retries=2):
    for i in range(retries + 1):
        try:
            async with _SEM:
                r = await _CLIENT.get(url)
            r.raise_for_status()
            return r.text
        except Exception:
            if i == retries:
                return None
            await asyncio.sleep(0.6 * (i + 1))

# ---------- Caché simple (memoria) ----------
_CACHE: Dict[str, Tuple[float, Any]] = {}

def cache_get(key: str):
    item = _CACHE.get(key)
    if not item:
        return None
    ts, val = item
    return val

def cache_set(key: str, value: Any, ttl_s: int):
    _CACHE[key] = (time.time() + ttl_s, value)

def cache_valid(key: str) -> bool:
    item = _CACHE.get(key)
    if not item:
        return False
    expires, _ = item
    return time.time() < expires

# ---------- Dólares ----------
async def _fetch_dolar_ambito_one(k: str, url: str):
    j = await _get_json(url)
    if j and isinstance(j, dict) and "venta" in j:
        return k, j
    return k, None

async def fetch_dolar_ambito_all():
    tasks = [asyncio.create_task(_fetch_dolar_ambito_one(k, u)) for k, u in DOLAR_TIPOS.items()]
    pairs = await asyncio.gather(*tasks)
    return {k: v for k, v in pairs if v}

async def fetch_dolar_fallback():
    j = await _get_json(CRIPTOYA_DOLAR)
    out = {}
    if j:
        mapping = {"blue":"blue","mep":"mep","ccl":"ccl","oficial":"oficial","mayorista":"mayorista","cripto":"cripto"}
        for a,b in mapping.items():
            node = j.get(b)
            if node:
                venta = node.get("venta") or node.get("price") or node.get("ask") or node.get("promedio")
                compra = node.get("compra") or node.get("bid") or venta
                out[a] = {
                    "compra": float(compra) if compra is not None else None,
                    "venta": float(venta) if venta is not None else None,
                    "nombre": a.upper(), "moneda": "ARS", "casa": "CriptoYa",
                    "fechaActualizacion": datetime.now(TZ).isoformat(),
                }
    return out

async def get_all_dolares():
    key = "dolares"
    if cache_valid(key):
        return cache_get(key)
    data = await fetch_dolar_ambito_all()
    if len(data) < 6:
        fb = await fetch_dolar_fallback()
        data = {**fb, **data}
    cache_set(key, data, ttl_s=20)  # 20 segundos
    return data

def format_dolares_block(data: dict) -> str:
    order = ["blue","mep","ccl","cripto","oficial","mayorista"]
    lines = []
    for k in order:
        d = data.get(k)
        if not d:
            continue
        compra = d.get("compra"); venta = d.get("venta")
        compra_s = fmt_money_ars(compra) if isinstance(compra,(int,float)) else "n/d"
        venta_s  = fmt_money_ars(venta)  if isinstance(venta,(int,float))  else "n/d"
        lines.append(f"{k.upper():10s} | Compra: {compra_s:>12s} | Venta: {venta_s:>12s}")
    return "```\n" + "\n".join(lines) + "\n```"

# ---------- Series oficiales (reservas / IPC) ----------
async def fetch_series_last_value(series_id: str, representation_mode: Optional[str] = None):
    params = {"ids": series_id, "limit": 1, "format": "json"}
    if representation_mode:
        params["representation_mode"] = representation_mode
    j = await _get_json(DATOS_API, params=params)
    try:
        data = j["data"]
        if not data:
            return None
        val_raw = data[-1][1]
        val = float(val_raw) if val_raw is not None else None
        date = data[-1][0][:10]
        return {"fecha": date, "valor": val}
    except Exception:
        return None

async def get_reservas_usd():
    key = "reservas"
    if cache_valid(key):
        return cache_get(key)
    for sid in RESERVAS_SERIES_CANDIDATAS:
        r = await fetch_series_last_value(sid)
        if r and r["valor"] is not None:
            r = r | {"serie": sid, "unidad": "millones de USD"}
            cache_set(key, r, ttl_s=6*3600)  # 6 horas
            return r
    return None

async def get_ipc_mensual():
    key = "ipc_mensual"
    if cache_valid(key):
        return cache_get(key)
    # Primero: serie ya en tasa mensual
    for sid in IPC_MENSUAL_SERIES_VARIACION[:1]:
        r = await fetch_series_last_value(sid)
        if r and r["valor"] is not None and -10.0 < r["valor"] < 200.0:
            r = r | {"serie": sid, "unidad": "%"}
            cache_set(key, r, ttl_s=24*3600)  # 24h
            return r
    # Fallback: calcular percent_change a partir del índice
    for sid in IPC_INDICE_SERIES:
        r = await fetch_series_last_value(sid, representation_mode="percent_change")
        if r and r["valor"] is not None and -10.0 < r["valor"] < 200.0:
            r = r | {"serie": sid, "unidad": "%"}
            cache_set(key, r, ttl_s=24*3600)
            return r
    # Último recurso: segundo ID alternativo de variación
    for sid in IPC_MENSUAL_SERIES_VARIACION[1:]:
        r = await fetch_series_last_value(sid)
        if r and r["valor"] is not None and -10.0 < r["valor"] < 200.0:
            r = r | {"serie": sid, "unidad": "%"}
            cache_set(key, r, ttl_s=24*3600)
            return r
    return None

# ---------- Riesgo país ----------
async def get_riesgo_pais():
    key = "riesgo"
    if cache_valid(key):
        return cache_get(key)
    j = await _get_json(ARGDATOS_RIESGO_ULTIMO)
    if j and isinstance(j, dict) and "valor" in j:
        try:
            val = float(j["valor"])
        except Exception:
            val = None
        out = {"fecha": j.get("fecha"), "valor": val}
        cache_set(key, out, ttl_s=15*60)  # 15 minutos
        return out
    lst = await _get_json(ARGDATOS_RIESGO_LISTA)
    if lst and isinstance(lst, list) and lst:
        last = lst[-1]
        try:
            val = float(last.get("valor"))
        except Exception:
            val = None
        out = {"fecha": last.get("fecha"), "valor": val}
        cache_set(key, out, ttl_s=15*60)
        return out
    return None

# ---------- Yahoo Finance (concurrencia + caché) ----------
async def yf_history_close(symbol: str, period: str = "6mo", interval: str = "1d"):
    params = {"period1": "0", "range": period, "interval": interval}
    j = await _get_json(YF_CHART.format(symbol=symbol), params=params)
    try:
        res = j["chart"]["result"][0]
        closes = res["indicators"]["quote"][0]["close"]
        timestamps = res["timestamp"]
        series = [(ts, c) for ts, c in zip(timestamps, closes) if c is not None]
        return series
    except Exception:
        return None

def trailing_returns_from_series(series):
    if not series:
        return None
    closes = [c for _, c in series]
    last = closes[-1]
    def ret_at(days):
        idx = -days-1
        base = closes[idx] if len(closes) > abs(idx) else closes[0]
        return pct(last, base)
    r1 = ret_at(21); r3 = ret_at(63); r6 = ret_at(126)
    return last, r1, r3, r6

async def _compute_one(symbol, name, sector):
    series = await yf_history_close(symbol, period="6mo", interval="1d")
    if not series:
        return None
    t = trailing_returns_from_series(series)
    if not t:
        return None
    last, r1, r3, r6 = t
    return {"symbol": symbol, "name": name, "sector": sector, "price": last, "r1": r1, "r3": r3, "r6": r6}

async def compute_universe_returns(universe, cache_key: str):
    if cache_valid(cache_key):
        return cache_get(cache_key)
    sem = asyncio.Semaphore(8)
    async def wrapped(sym, nam, sec):
        async with sem:
            return await _compute_one(sym, nam, sec)
    tasks = [asyncio.create_task(wrapped(s, n, sec)) for s, n, sec in universe]
    results = await asyncio.gather(*tasks)
    data = [r for r in results if r]
    cache_set(cache_key, data, ttl_s=5*60)  # 5 minutos
    return data

def pick_top_by_3m(data, topn=3):
    filtered = [d for d in data if d["r3"] == d["r3"]]
    return sorted(filtered, key=lambda x: x["r3"], reverse=True)[:topn]

def pick_top_by_projection(data, topn=5):
    def score(d):
        s = 0.0
        for w, k in [(0.1,"r1"), (0.3,"r3"), (0.6,"r6")]:
            v = d.get(k)
            if v == v:
                s += w * v
        return s
    for d in data:
        d["_score"] = score(d)
    filtered = [d for d in data if d["_score"] == d["_score"]]
    return sorted(filtered, key=lambda x: x["_score"], reverse=True)[:topn]

def format_card(item):
    price = item.get("price")
    s_price = f"${price:.2f}" if isinstance(price,(int,float)) else "n/d"
    return (
        f"• {item['symbol']} ({item['name']}) — {s_price}\n"
        f"  • Sector: {item['sector']}\n"
        f"  • Rendimientos: {human_pct(item['r1'])} · {human_pct(item['r3'])} · {human_pct(item['r6'])}"
    )

# ---------- Noticias (concurrencia + filtro + caché) ----------
def simple_rss_parse(xml_text: str):
    items = []
    if not xml_text:
        return items
    # RSS 2.0
    for m in re.finditer(r"<item>(.*?)</item>", xml_text, re.DOTALL|re.IGNORECASE):
        block = m.group(1)
        title = re.search(r"<title>(.*?)</title>", block, re.DOTALL|re.IGNORECASE)
        link  = re.search(r"<link>(.*?)</link>", block, re.DOTALL|re.IGNORECASE)
        date  = re.search(r"<pubDate>(.*?)</pubDate>", block, re.DOTALL|re.IGNORECASE)
        t = (title.group(1).strip() if title else "").strip()
        # Quitar CDATA
        t = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", t)
        items.append({"title": t, "link": (link.group(1).strip() if link else "").strip(),
                      "published": (date.group(1).strip() if date else "").strip()})
    # Atom fallback
    if not items:
        for m in re.finditer(r"<entry>(.*?)</entry>", xml_text, re.DOTALL|re.IGNORECASE):
            block = m.group(1)
            title = re.search(r"<title>(.*?)</title>", block, re.DOTALL|re.IGNORECASE)
            link_m = re.search(r"<link[^>]*href=['\"](.*?)['\"][^>]*/?>", block, re.DOTALL|re.IGNORECASE)
            date  = re.search(r"<updated>(.*?)</updated>", block, re.DOTALL|re.IGNORECASE) or \
                    re.search(r"<published>(.*?)</published>", block, re.DOTALL|re.IGNORECASE)
            t = (title.group(1).strip() if title else "").strip()
            t = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", t)
            items.append({"title": t, "link": (link_m.group(1).strip() if link_m else "").strip(),
                          "published": (date.group(1).strip() if date else "").strip()})
    return items

async def fetch_news_filtered(limit=5):
    key = "news5"
    if cache_valid(key):
        return cache_get(key)
    async def load_feed(url):
        return await _get_text(url)
    tasks = [asyncio.create_task(load_feed(u)) for u in RSS_FEEDS]
    xmls = await asyncio.gather(*tasks)
    hits = []
    for xml in xmls:
        if not xml:
            continue
        for it in simple_rss_parse(xml)[:25]:
            t = (it["title"] or "").lower()
            if (not t) or any(bad in t for bad in NEWS_KEYWORDS_NEG):
                continue
            if NEWS_KEYWORDS_POS and not any(kw in t for kw in NEWS_KEYWORDS_POS):
                continue
            hits.append(it)
    # si no hay hits por filtro, devolvemos 5 últimas de lo que haya
    if not hits:
        for xml in xmls:
            if not xml:
                continue
            base = simple_rss_parse(xml)[:10]
            hits.extend(base)
    # ordenar y deduplicar
    def parse_dt(s):
        try:
            return datetime.strptime(s[:25], "%a, %d %b %Y %H:%M:%S")
        except Exception:
            return datetime.now()
    hits = sorted(hits, key=lambda x: parse_dt(x.get("published","")), reverse=True)
    seen, uniq = set(), []
    for h in hits:
        title = h["title"].strip()
        if title in seen or not title:
            continue
        seen.add(title)
        uniq.append(h)
        if len(uniq) >= limit:
            break
    cache_set(key, uniq, ttl_s=15*60)  # 15 minutos
    return uniq

# ---------- Alertas (SQLite) ----------
DB_PATH = os.environ.get("ALERTS_DB_PATH", "./alerts.db")

def db_init():
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id INTEGER NOT NULL,
        tipo TEXT NOT NULL,
        umbral REAL NOT NULL,
        created_at TEXT NOT NULL,
        active INTEGER NOT NULL DEFAULT 1
    );""")
    con.commit(); con.close()

def db_add_alert(chat_id: int, tipo: str, umbral: float) -> int:
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    cur.execute("INSERT INTO alerts (chat_id, tipo, umbral, created_at, active) VALUES (?,?,?,?,1)",
                (chat_id, tipo, umbral, datetime.now(TZ).isoformat()))
    con.commit(); alert_id = cur.lastrowid; con.close(); return alert_id

def db_list_alerts(chat_id: int):
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    cur.execute("SELECT id, tipo, umbral, active, created_at FROM alerts WHERE chat_id = ? ORDER BY id DESC", (chat_id,))
    rows = cur.fetchall(); con.close(); return rows

def db_delete_alert(chat_id: int, alert_id: int) -> bool:
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    cur.execute("DELETE FROM alerts WHERE id = ? AND chat_id = ?", (alert_id, chat_id))
    con.commit(); ok = cur.rowcount > 0; con.close(); return ok

def db_list_all_active():
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    cur.execute("SELECT id, chat_id, tipo, umbral FROM alerts WHERE active = 1")
    rows = cur.fetchall(); con.close(); return rows

# ---------- Handlers ----------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "👋 Bienvenido al *Bot Económico AR*.\n\n"
        "/dolar /reservas /inflacion /riesgo\n"
        "/acciones /cedears /ranking_acciones /ranking_cedears\n"
        "/alerta_dolar <tipo> <umbral>  /alertas  /alerta_borrar <id>\n"
        "/resumen_diario"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)

async def cmd_dolar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    data = await get_all_dolares()
    if not data:
        await update.message.reply_text("⚠️ No pude obtener cotizaciones ahora.")
        return
    block = format_dolares_block(data)
    await update.message.reply_text("💵 *Dólares (Ámbito / fallback CriptoYa)*\n" + block, parse_mode=ParseMode.MARKDOWN)

async def cmd_reservas(update: Update, context: ContextTypes.DEFAULT_TYPE):
    r = await get_reservas_usd()
    if not r:
        await update.message.reply_text("🏦 Reservas BCRA: dato no disponible ahora (fuente oficial).")
        return
    txt = f"🏦 *Reservas BCRA*\n`{r['fecha']}` — {r['valor']:.2f} millones de USD  \n_(Serie: {r['serie']})_"
    await update.message.reply_text(txt, parse_mode=ParseMode.MARKDOWN)

async def cmd_inflacion(update: Update, context: ContextTypes.DEFAULT_TYPE):
    r = await get_ipc_mensual()
    if not r:
        await update.message.reply_text("📈 Inflación mensual: dato no disponible ahora (INDEC).")
        return
    txt = f"📈 *Inflación (variación mensual)*\n`{r['fecha']}` — {r['valor']:.2f}%  \n_(Serie: {r['serie']})_"
    await update.message.reply_text(txt, parse_mode=ParseMode.MARKDOWN)

async def cmd_riesgo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    r = await get_riesgo_pais()
    if not r or r.get("valor") is None:
        await update.message.reply_text("📉 Riesgo país: dato no disponible ahora (ArgentinaDatos).")
        return
    txt = f"📉 *Riesgo País (EMBI+)*\n`{r['fecha']}` — {int(round(r['valor']))} pts"
    await update.message.reply_text(txt, parse_mode=ParseMode.MARKDOWN)

async def cmd_acciones(update: Update, context: ContextTypes.DEFAULT_TYPE):
    data = await compute_universe_returns(BYMA_UNIVERSO, cache_key="byma_universe")
    top = pick_top_by_3m(data, topn=3)
    if not top:
        await update.message.reply_text("📊 No pude calcular el top de acciones ahora.")
        return
    cards = "\n\n".join(format_card(x) for x in top)
    await update.message.reply_text(f"📊 *Top 3 Acciones BYMA (3M)*\n{cards}", parse_mode=ParseMode.MARKDOWN)

async def cmd_cedears(update: Update, context: ContextTypes.DEFAULT_TYPE):
    data = await compute_universe_returns(CEDEARS_UNIVERSO, cache_key="cedears_universe")
    top = pick_top_by_3m(data, topn=3)
    if not top:
        await update.message.reply_text("🌎 No pude calcular el top de CEDEARs ahora.")
        return
    cards = "\n\n".join(format_card(x) for x in top)
    await update.message.reply_text(f"🌎 *Top 3 CEDEARs (3M)*\n{cards}", parse_mode=ParseMode.MARKDOWN)

async def cmd_ranking_acciones(update: Update, context: ContextTypes.DEFAULT_TYPE):
    data = await compute_universe_returns(BYMA_UNIVERSO, cache_key="byma_universe")
    top = pick_top_by_projection(data, topn=5)
    if not top:
        await update.message.reply_text("🏁 No pude armar el ranking de acciones ahora.")
        return
    lines = [f"{i}. {x['symbol']} — score: {x['_score']:.2f}\n   {format_card(x)}" for i, x in enumerate(top, 1)]
    await update.message.reply_text("🏁 *Ranking Acciones BYMA (proyección 6M)*\n" + "\n\n".join(lines), parse_mode=ParseMode.MARKDOWN)

async def cmd_ranking_cedears(update: Update, context: ContextTypes.DEFAULT_TYPE):
    data = await compute_universe_returns(CEDEARS_UNIVERSO, cache_key="cedears_universe")
    top = pick_top_by_projection(data, topn=5)
    if not top:
        await update.message.reply_text("🏁 No pude armar el ranking de CEDEARs ahora.")
        return
    lines = [f"{i}. {x['symbol']} — score: {x['_score']:.2f}\n   {format_card(x)}" for i, x in enumerate(top, 1)]
    await update.message.reply_text("🏁 *Ranking CEDEARs (proyección 6M)*\n" + "\n\n".join(lines), parse_mode=ParseMode.MARKDOWN)

async def cmd_alerta_dolar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    parts = (update.message.text or "").strip().split()
    if len(parts) != 3:
        await update.message.reply_text("Uso: /alerta_dolar <tipo> <umbral>\nTipos: blue|mep|ccl|cripto|oficial|mayorista")
        return
    _, tipo, umbral = parts
    tipo = tipo.lower()
    if tipo not in DOLAR_TIPOS:
        await update.message.reply_text("Tipo inválido. Use: blue|mep|ccl|cripto|oficial|mayorista")
        return
    try:
        umbral = float(umbral.replace(",", "."))
    except ValueError:
        await update.message.reply_text("Umbral inválido. Ej: 1250 o 1234.50")
        return
    alert_id = db_add_alert(update.message.chat_id, tipo, umbral)
    await update.message.reply_text(f"🔔 Alerta creada (ID {alert_id}) — {tipo.upper()} ≥ {fmt_money_ars(umbral)}")

async def cmd_alertas(update: Update, context: ContextTypes.DEFAULT_TYPE):
    rows = db_list_alerts(update.message.chat_id)
    if not rows:
        await update.message.reply_text("🔔 No tenés alertas activas.")
        return
    lines = []
    for (id_, tipo, umbral, active, created_at) in rows:
        status = "ON" if active else "OFF"
        lines.append(f"#{id_:03d} — {tipo.upper()} ≥ {fmt_money_ars(umbral)} — {status} — {created_at[:16]}")
    await update.message.reply_text("🔔 *Alertas*\n" + "```\n" + "\n".join(lines) + "\n```", parse_mode=ParseMode.MARKDOWN)

async def cmd_alerta_borrar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    parts = (update.message.text or "").strip().split()
    if len(parts) != 2:
        await update.message.reply_text("Uso: /alerta_borrar <id>")
        return
    try:
        alert_id = int(parts[1])
    except ValueError:
        await update.message.reply_text("ID inválido.")
        return
    ok = db_delete_alert(update.message.chat_id, alert_id)
    await update.message.reply_text("🗑️ Alerta eliminada." if ok else "No se encontró esa alerta.")

async def cmd_resumen_diario(update: Update, context: ContextTypes.DEFAULT_TYPE):
    dolares, reservas, ipc, riesgo, news = await asyncio.gather(
        get_all_dolares(), get_reservas_usd(), get_ipc_mensual(), get_riesgo_pais(), fetch_news_filtered(limit=5)
    )
    block = format_dolares_block(dolares) if dolares else "n/d"
    reservas_txt = f"`{reservas['fecha']}` — {reservas['valor']:.2f} M USD" if reservas else "Dato no disponible"
    ipc_txt      = f"`{ipc['fecha']}` — {ipc['valor']:.2f}%" if ipc else "Dato no disponible"
    riesgo_txt   = f"`{riesgo['fecha']}` — {int(round(riesgo['valor']))} pts" if (riesgo and riesgo.get("valor") is not None) else "Dato no disponible"
    news_txt = "\n".join([f"• [{n['title']}]({n['link']})" for n in (news or [])]) if news else "_Sin noticias filtradas ahora mismo._"
    out = (
        f"🗓️ *Resumen {now_str()}*\n\n"
        f"💵 *Dólares*\n{block}\n"
        f"🏦 *Reservas*: {reservas_txt}\n"
        f"📈 *Inflación*: {ipc_txt}\n"
        f"📉 *Riesgo País*: {riesgo_txt}\n\n"
        f"📰 *Noticias*\n{news_txt}"
    )
    await update.message.reply_text(out, parse_mode=ParseMode.MARKDOWN, disable_web_page_preview=True)

# ---------- Job de alertas y refresh cachés (si JobQueue disponible) ----------
async def job_check_alerts(context: ContextTypes.DEFAULT_TYPE):
    rows = db_list_all_active()
    if not rows:
        return
    dolares = await get_all_dolares()
    for (alert_id, chat_id, tipo, umbral) in rows:
        d = dolares.get(tipo) if dolares else None
        precio = d.get("venta") if d else None
        try:
            thr = float(umbral)
        except Exception:
            continue
        if isinstance(precio, (int, float)) and precio >= thr:
            txt = f"🔔 *Alerta #{alert_id}*\n{tipo.upper()} alcanzó {fmt_money_ars(precio)} (umbral {fmt_money_ars(thr)})."
            try:
                await context.bot.send_message(chat_id=chat_id, text=txt, parse_mode=ParseMode.MARKDOWN)
            except Exception:
                pass
            db_delete_alert(chat_id, alert_id)

async def job_refresh_caches(context: ContextTypes.DEFAULT_TYPE):
    # refresca cachés para respuestas instantáneas
    await asyncio.gather(
        get_all_dolares(),
        get_reservas_usd(),
        get_ipc_mensual(),
        get_riesgo_pais(),
        compute_universe_returns(BYMA_UNIVERSO, cache_key="byma_universe"),
        compute_universe_returns(CEDEARS_UNIVERSO, cache_key="cedears_universe"),
        fetch_news_filtered(limit=5),
    )

# ---------- Health server (thread) ----------
def start_health_server_in_thread():
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    port = int(os.environ.get("PORT", "0") or "0")
    if port <= 0:
        return
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/healthz":
                self.send_response(200); self.send_header("Content-Type", "text/plain"); self.end_headers(); self.wfile.write(b"OK")
            else:
                self.send_response(200); self.send_header("Content-Type", "text/plain"); self.end_headers(); self.wfile.write(b"OK")
        def log_message(self, *args): return
    server = HTTPServer(("0.0.0.0", port), Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True); t.start()
    print(f"[health] HTTP server (thread) listening on 0.0.0.0:{port}")

# ---------- Main ----------
def build_app():
    token = os.environ.get("BOT_TOKEN")
    if not token:
        raise RuntimeError("Falta BOT_TOKEN en variables de entorno.")
    app = ApplicationBuilder().token(token).build()

    # Handlers
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help",  cmd_start))
    app.add_handler(CommandHandler("dolar", cmd_dolar))
    app.add_handler(CommandHandler("reservas", cmd_reservas))
    app.add_handler(CommandHandler("inflacion", cmd_inflacion))
    app.add_handler(CommandHandler("riesgo", cmd_riesgo))
    app.add_handler(CommandHandler("acciones", cmd_acciones))
    app.add_handler(CommandHandler("cedears", cmd_cedears))
    app.add_handler(CommandHandler("ranking_acciones", cmd_ranking_acciones))
    app.add_handler(CommandHandler("ranking_cedears", cmd_ranking_cedears))
    app.add_handler(CommandHandler("alerta_dolar", cmd_alerta_dolar))
    app.add_handler(CommandHandler("alertas", cmd_alertas))
    app.add_handler(CommandHandler("alerta_borrar", cmd_alerta_borrar))
    app.add_handler(CommandHandler("resumen_diario", cmd_resumen_diario))

    # Jobs (si está disponible JobQueue)
    jq = getattr(app, "job_queue", None)
    if jq is not None:
        jq.run_repeating(job_check_alerts, interval=90, first=10)
        jq.run_repeating(job_refresh_caches, interval=300, first=5)  # refresca cachés cada 5 min
    else:
        print("[jobs] JobQueue no disponible — instalá 'python-telegram-bot[job-queue]' para habilitar alertas y prefetch.")

    return app

def main():
    db_init()
    app = build_app()
    start_health_server_in_thread()
    app.run_polling()

if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        pass
