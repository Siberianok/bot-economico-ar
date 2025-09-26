#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bot Económico AR — Telegram
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
- Dólares: DolarAPI (Ámbito) con fallback CriptoYa
- Reservas e Inflación: apis.datos.gob.ar (series oficiales)
- Riesgo país: ArgentinaDatos con fallback a listado
- Acciones/CEDEARs: Yahoo Finance (chart)
- Noticias: RSS de medios económicos locales

Runtime: Python-telegram-bot v20.x (asyncio), httpx
Persistencia de alertas: SQLite (stdlib), archivo local ./alerts.db
Zona horaria: America/Argentina/Buenos_Aires
"""

import asyncio
import os
import re
import math
import json
import time
import sqlite3
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import httpx
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    ApplicationBuilder, CommandHandler, ContextTypes
)

# ---------- Config ----------

TZ = ZoneInfo("America/Argentina/Buenos_Aires")

DOLARAPI_BASE = "https://dolarapi.com/v1/ambito/dolares"
CRIPTOYA_DOLAR = "https://criptoya.com/api/dolar"

ARGDATOS_RIESGO_ULTIMO = "https://api.argentinadatos.com/v1/finanzas/indices/riesgo-pais/ultimo"
ARGDATOS_RIESGO_LISTA = "https://api.argentinadatos.com/v1/finanzas/indices/riesgo-pais"

# Series oficiales (probaremos varias hasta obtener dato)
# Reservas internacionales (millones de USD)
RESERVAS_SERIES_CANDIDATAS = [
    # Serie de Reservas del dataset SSPM (nombres varían según versiones)
    "92.2_RESERVAS_IRES_0_0_32_40",  # frecuente
    "92.1_RESERVA_0_0_26",
    "92.1_RESERVAS_0_0_26",
    "92.1_RESERVAS_D_0_0_18",
]
# IPC mensual (variación m/m). Intentamos series de variación y, si falla, derivamos desde índice
IPC_MENSUAL_SERIES_VARIACION = [
    "148.3_INIVELNAL_DICI_M_26",  # INDEC Nivel general (variación m/m) — común
    "148.3_INIVELNAL_DICI_M_12",
    "103.1_I2N_2016_M_19",        # IPC Total Nacional variación m/m (algunas versiones)
]
IPC_INDICE_SERIES = [
    "103.1_I2N_2016_M_15",        # Índice nivel general (Nacional)
]

DATOS_API = "https://apis.datos.gob.ar/series/api/series"

# Universos (pueden ampliarse fácilmente)
BYMA_UNIVERSO = [
    # Bancos / Financieras
    ("GGAL.BA", "Grupo Financiero Galicia", "Bancos"),
    ("BMA.BA", "Banco Macro", "Bancos"),
    ("BBAR.BA", "BBVA Argentina", "Bancos"),
    ("SUPV.BA", "Supervielle", "Bancos"),
    ("VALO.BA", "Grupo Valores", "Servicios Financieros"),
    ("BYMA.BA", "Bolsas y Mercados Argentinos", "Servicios Financieros"),
    # Energía
    ("YPFD.BA", "YPF", "Energía"),
    ("PAMP.BA", "Pampa Energía", "Energía"),
    ("CEPU.BA", "Central Puerto", "Energía"),
    ("TGSU2.BA", "Transportadora de Gas del Sur", "Energía"),
    ("TGNO4.BA", "Transportadora de Gas del Norte", "Energía"),
    # Industriales / Materiales
    ("TXAR.BA", "Ternium Argentina", "Materiales"),
    ("ALUA.BA", "Aluar", "Materiales"),
    ("LOMA.BA", "Loma Negra", "Materiales"),
    ("COME.BA", "Soc. Comercial del Plata", "Industriales"),
    # Telecom / Medios / Real Estate / Agro
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

# RSS de medios económicos en AR
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
    "dólar", "dolar", "mep", "ccl", "reservas", "inflación", "ipc", "riesgo país", "riesgo-pais",
    "bcra", "bono", "bonos", "acciones", "cedear", "merval", "mercado", "pbi", "exportaciones",
    "tasa", "liquidez", "deuda", "fmi", "brecha", "superávit", "déficit", "blue", "cambio", "oficial",
]
NEWS_KEYWORDS_NEG = ["quiniela", "horóscopo", "salud", "covid", "farandula", "espectáculo", "policiales"]

# ---------- Utilidades ----------

def fmt_money_ars(x: float) -> str:
    """Formatea ARS con punto de miles y coma decimal, sin redondeos extraños."""
    try:
        s = f"{x:,.2f}"
    except Exception:
        return str(x)
    s = s.replace(",", "_").replace(".", ",").replace("_", ".")
    return f"$ {s}"

def now_str() -> str:
    return datetime.now(TZ).strftime("%Y-%m-%d %H:%M")

def take_last_safe(seq):
    return seq[-1] if seq else None

def pct(a: float, b: float) -> float:
    try:
        return (a/b - 1.0) * 100.0
    except Exception:
        return float('nan')

def human_pct(x: float) -> str:
    if x != x or math.isinf(x):  # NaN or inf
        return "n/d"
    return f"{x:+.2f}%"

# ---------- HTTP ----------

_http_client = httpx.AsyncClient(timeout=httpx.Timeout(12.0, connect=8.0))

async def http_json(url: str, params: dict | None = None):
    try:
        r = await _http_client.get(url, params=params, headers={"User-Agent": "AR-EconBot/1.0"})
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

async def http_text(url: str):
    try:
        r = await _http_client.get(url, headers={"User-Agent": "AR-EconBot/1.0"})
        r.raise_for_status()
        return r.text
    except Exception:
        return None

# ---------- Dólares ----------

DOLAR_TIPOS = {
    "blue": f"{DOLARAPI_BASE}/blue",
    "mep": f"{DOLARAPI_BASE}/bolsa",
    "ccl": f"{DOLARAPI_BASE}/contadoconliqui",
    "cripto": f"{DOLARAPI_BASE}/cripto",
    "oficial": f"{DOLARAPI_BASE}/oficial",
    "mayorista": f"{DOLARAPI_BASE}/mayorista",
}

async def fetch_dolar_ambito():
    out = {}
    for k, url in DOLAR_TIPOS.items():
        j = await http_json(url)
        if j and isinstance(j, dict) and "venta" in j:
            out[k] = j
    return out

async def fetch_dolar_fallback():
    j = await http_json(CRIPTOYA_DOLAR)
    out = {}
    if j:
        # Mapear estructura criptoya -> similar a dolarapi
        mapping = {
            "blue": "blue",
            "mep": "mep",
            "ccl": "ccl",
            "oficial": "oficial",
            "mayorista": "mayorista",
            "cripto": "cripto",
        }
        for a,b in mapping.items():
            node = j.get(b)
            if node:
                # algunos campos pueden variar; normalizamos
                venta = node.get("venta") or node.get("price") or node.get("ask") or node.get("promedio")
                compra = node.get("compra") or node.get("bid") or venta
                out[a] = {
                    "compra": float(compra) if compra is not None else None,
                    "venta": float(venta) if venta is not None else None,
                    "nombre": a.upper(),
                    "moneda": "ARS",
                    "casa": "CriptoYa",
                    "fechaActualizacion": datetime.now(TZ).isoformat(),
                }
    return out

async def get_all_dolares():
    data = await fetch_dolar_ambito()
    if len(data) < 6:
        fb = await fetch_dolar_fallback()
        data = {**fb, **data}  # preferimos Ámbito, completamos con fallback
    return data

def format_dolares_block(data: dict) -> str:
    order = ["blue", "mep", "ccl", "cripto", "oficial", "mayorista"]
    lines = []
    for k in order:
        d = data.get(k)
        if not d:
            continue
        compra = d.get("compra")
        venta = d.get("venta")
        compra_s = fmt_money_ars(compra) if isinstance(compra, (int, float)) else "n/d"
        venta_s  = fmt_money_ars(venta)  if isinstance(venta, (int, float)) else "n/d"
        name = k.upper()
        lines.append(f"{name:10s} | Compra: {compra_s:>12s} | Venta: {venta_s:>12s}")
    return "```\n" + "\n".join(lines) + "\n```"

# ---------- Series oficiales (reservas / IPC) ----------

async def fetch_series_last_value(series_id: str, representation_mode: str | None = None):
    params = {"ids": series_id, "limit": 1}
    if representation_mode:
        params["representation_mode"] = representation_mode
    j = await http_json(DATOS_API, params=params)
    try:
        data = j["data"]
        if not data:
            return None
        val = data[-1][1]
        date = data[-1][0][:10]
        return {"fecha": date, "valor": float(val)}
    except Exception:
        return None

async def get_reservas_usd():
    # intentamos series candidatas
    for sid in RESERVAS_SERIES_CANDIDATAS:
        r = await fetch_series_last_value(sid)
        if r and r["valor"] is not None:
            return r | {"serie": sid, "unidad": "millones de USD"}
    return None

async def get_ipc_mensual():
    # primero intentamos series de variación ya calculadas
    for sid in IPC_MENSUAL_SERIES_VARIACION:
        r = await fetch_series_last_value(sid)
        if r and r["valor"] is not None and -10.0 < r["valor"] < 100.0:
            return r | {"serie": sid, "unidad": "%"}
    # Si no hay, derivamos desde índice nivel general con percent_change
    for sid in IPC_INDICE_SERIES:
        r = await fetch_series_last_value(sid, representation_mode="percent_change")
        if r and r["valor"] is not None and -10.0 < r["valor"] < 100.0:
            return r | {"serie": sid, "unidad": "%"}
    return None

# ---------- Riesgo país ----------

async def get_riesgo_pais():
    j = await http_json(ARGDATOS_RIESGO_ULTIMO)
    if j and isinstance(j, dict) and "valor" in j:
        return {"fecha": j.get("fecha"), "valor": float(j["valor"])}
    # fallback list
    lst = await http_json(ARGDATOS_RIESGO_LISTA)
    if lst and isinstance(lst, list) and lst:
        last = lst[-1]
        return {"fecha": last.get("fecha"), "valor": float(last.get("valor"))}
    return None

# ---------- Yahoo Finance ----------

YF_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

async def yf_history_close(symbol: str, period: str = "6mo", interval: str = "1d"):
    params = {"period1": "0", "range": period, "interval": interval}
    j = await http_json(YF_CHART.format(symbol=symbol), params=params)
    try:
        res = j["chart"]["result"][0]
        closes = res["indicators"]["quote"][0]["close"]
        timestamps = res["timestamp"]
        # limpiamos NaN
        series = [(ts, c) for ts, c in zip(timestamps, closes) if c is not None]
        return series
    except Exception:
        return None

def trailing_returns_from_series(series: list[tuple[int, float]]):
    # series: list of (timestamp, close) ascending
    if not series:
        return None
    closes = [c for _, c in series]
    last = closes[-1]
    # 21, 63, 126 ~ 1m, 3m, 6m de ruedas
    def ret_at(days):
        idx = -days-1
        if len(closes) > abs(idx):
            base = closes[idx]
        else:
            base = closes[0]
        return pct(last, base)
    r1 = ret_at(21)
    r3 = ret_at(63)
    r6 = ret_at(126)
    return last, r1, r3, r6

async def compute_universe_returns(universe):
    out = []
    for symbol, name, sector in universe:
        series = await yf_history_close(symbol, period="6mo", interval="1d")
        if not series:
            continue
        last, r1, r3, r6 = trailing_returns_from_series(series)
        out.append({
            "symbol": symbol, "name": name, "sector": sector,
            "price": last, "r1": r1, "r3": r3, "r6": r6
        })
    return out

def pick_top_by_3m(data, topn=3):
    # ordena por r3 desc, filtra NaN
    filtered = [d for d in data if d["r3"] == d["r3"]]  # not NaN
    return sorted(filtered, key=lambda x: x["r3"], reverse=True)[:topn]

def pick_top_by_projection(data, topn=5):
    # score = 0.1*1m + 0.3*3m + 0.6*6m
    def score(d):
        s = 0.0
        for w, k in [(0.1, "r1"), (0.3, "r3"), (0.6, "r6")]:
            v = d.get(k)
            if v == v:  # not NaN
                s += w * v
        return s
    for d in data:
        d["_score"] = score(d)
    filtered = [d for d in data if d["_score"] == d["_score"]]
    return sorted(filtered, key=lambda x: x["_score"], reverse=True)[:topn]

def format_card(item):
    # Ticker (Empresa) — Precio / • Sector / Rendimientos: 1m · 3m · 6m
    price = item.get("price")
    s_price = f"${price:.2f}" if isinstance(price, (int, float)) else "n/d"
    s = f"• {item['symbol']} ({item['name']}) — {s_price}\n" \
        f"  • Sector: {item['sector']}\n" \
        f"  • Rendimientos: {human_pct(item['r1'])} · {human_pct(item['r3'])} · {human_pct(item['r6'])}"
    return s

# ---------- Noticias ----------

def simple_rss_parse(xml_text: str):
    # Parser mínimo compatible con RSS/Atom (sin dependencia extra)
    # Devuelve lista de dicts: {title, link, published}
    items = []
    if not xml_text:
        return items
    # RSS
    for m in re.finditer(r"<item>(.*?)</item>", xml_text, re.DOTALL | re.IGNORECASE):
        block = m.group(1)
        title = re.search(r"<title>(.*?)</title>", block, re.DOTALL | re.IGNORECASE)
        link = re.search(r"<link>(.*?)</link>", block, re.DOTALL | re.IGNORECASE)
        date = re.search(r"<pubDate>(.*?)</pubDate>", block, re.DOTALL | re.IGNORECASE)
        items.append({
            "title": (title.group(1).strip() if title else "").strip(),
            "link": (link.group(1).strip() if link else "").strip(),
            "published": (date.group(1).strip() if date else "").strip(),
        })
    # Atom
    if not items:
        for m in re.finditer(r"<entry>(.*?)</entry>", xml_text, re.DOTALL | re.IGNORECASE):
            block = m.group(1)
            title = re.search(r"<title>(.*?)</title>", block, re.DOTALL | re.IGNORECASE)
            link_m = re.search(r"<link[^>]*href=['\"](.*?)['\"][^>]*/?>", block, re.DOTALL | re.IGNORECASE)
            date = re.search(r"<updated>(.*?)</updated>", block, re.DOTALL | re.IGNORECASE) or \
                   re.search(r"<published>(.*?)</published>", block, re.DOTALL | re.IGNORECASE)
            items.append({
                "title": (title.group(1).strip() if title else "").strip(),
                "link": (link_m.group(1).strip() if link_m else "").strip(),
                "published": (date.group(1).strip() if date else "").strip(),
            })
    return items

async def fetch_news_filtered(limit=5):
    hits = []
    for url in RSS_FEEDS:
        xml = await http_text(url)
        if not xml:
            continue
        for it in simple_rss_parse(xml)[:20]:
            t = (it["title"] or "").lower()
            if t and any(k in t for k in NEWS_KEYWORDS_POS) and not any(k in t for k in NEWS_KEYWORDS_NEG):
                hits.append(it)
    # ordenar por 'published' si posible, sin romper
    def parse_dt(s):
        try:
            return datetime.strptime(s[:25], "%a, %d %b %Y %H:%M:%S")
        except Exception:
            return datetime.now()
    hits = sorted(hits, key=lambda x: parse_dt(x.get("published","")), reverse=True)
    # dedup by title
    seen = set()
    uniq = []
    for h in hits:
        key = h["title"]
        if key in seen:
            continue
        seen.add(key)
        uniq.append(h)
        if len(uniq) >= limit:
            break
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
    con.commit()
    con.close()

def db_add_alert(chat_id: int, tipo: str, umbral: float) -> int:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "INSERT INTO alerts (chat_id, tipo, umbral, created_at, active) VALUES (?,?,?,?,1)",
        (chat_id, tipo, umbral, datetime.now(TZ).isoformat())
    )
    con.commit()
    alert_id = cur.lastrowid
    con.close()
    return alert_id

def db_list_alerts(chat_id: int):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT id, tipo, umbral, active, created_at FROM alerts WHERE chat_id = ? ORDER BY id DESC", (chat_id,))
    rows = cur.fetchall()
    con.close()
    return rows

def db_delete_alert(chat_id: int, alert_id: int) -> bool:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("DELETE FROM alerts WHERE id = ? AND chat_id = ?", (alert_id, chat_id))
    con.commit()
    ok = cur.rowcount > 0
    con.close()
    return ok

def db_list_all_active():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT id, chat_id, tipo, umbral FROM alerts WHERE active = 1")
    rows = cur.fetchall()
    con.close()
    return rows

# ---------- Bot Handlers ----------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "👋 Bienvenido al *Bot Económico AR*.\n\n"
        "Comandos disponibles:\n"
        "/dolar — Tipos de cambio (Blue, MEP, CCL, Cripto, Oficial, Mayorista)\n"
        "/reservas — Reservas del BCRA\n"
        "/inflacion — Inflación mensual (variación)\n"
        "/riesgo — Riesgo país (EMBI+)\n"
        "/acciones — Top 3 acciones BYMA por rendimiento 3M\n"
        "/cedears — Top 3 CEDEARs por rendimiento 3M\n"
        "/ranking_acciones — Top 5 acciones BYMA por proyección 6M\n"
        "/ranking_cedears — Top 5 CEDEARs por proyección 6M\n"
        "/alerta_dolar <tipo> <umbral> — Crea alerta (blue|mep|ccl|cripto|oficial|mayorista)\n"
        "/alertas — Lista alertas activas\n"
        "/alerta_borrar <id> — Elimina una alerta\n"
        "/resumen_diario — Dólares + Reservas + Inflación + Riesgo + 5 noticias\n"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)

async def cmd_dolar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    data = await get_all_dolares()
    if not data:
        await update.message.reply_text("⚠️ No pude obtener cotizaciones en este momento.")
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
        await update.message.reply_text("📈 Inflación mensual: dato no disponible ahora (INDEC, serie oficial).")
        return
    txt = f"📈 *Inflación (variación mensual)*\n`{r['fecha']}` — {r['valor']:.2f}%  \n_(Serie: {r['serie']})_"
    await update.message.reply_text(txt, parse_mode=ParseMode.MARKDOWN)

async def cmd_riesgo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    r = await get_riesgo_pais()
    if not r:
        await update.message.reply_text("📉 Riesgo país: dato no disponible ahora (ArgentinaDatos).")
        return
    txt = f"📉 *Riesgo País (EMBI+)*\n`{r['fecha']}` — {int(round(r['valor']))} pts"
    await update.message.reply_text(txt, parse_mode=ParseMode.MARKDOWN)

async def cmd_acciones(update: Update, context: ContextTypes.DEFAULT_TYPE):
    data = await compute_universe_returns(BYMA_UNIVERSO)
    top = pick_top_by_3m(data, topn=3)
    if not top:
        await update.message.reply_text("📊 No pude calcular el top de acciones ahora.")
        return
    cards = "\n\n".join(format_card(x) for x in top)
    await update.message.reply_text(f"📊 *Top 3 Acciones BYMA (3M)*\n{cards}", parse_mode=ParseMode.MARKDOWN)

async def cmd_cedears(update: Update, context: ContextTypes.DEFAULT_TYPE):
    data = await compute_universe_returns(CEDEARS_UNIVERSO)
    top = pick_top_by_3m(data, topn=3)
    if not top:
        await update.message.reply_text("🌎 No pude calcular el top de CEDEARs ahora.")
        return
    cards = "\n\n".join(format_card(x) for x in top)
    await update.message.reply_text(f"🌎 *Top 3 CEDEARs (3M)*\n{cards}", parse_mode=ParseMode.MARKDOWN)

async def cmd_ranking_acciones(update: Update, context: ContextTypes.DEFAULT_TYPE):
    data = await compute_universe_returns(BYMA_UNIVERSO)
    top = pick_top_by_projection(data, topn=5)
    if not top:
        await update.message.reply_text("🏁 No pude armar el ranking de acciones ahora.")
        return
    lines = []
    for i, x in enumerate(top, 1):
        lines.append(f"{i}. {x['symbol']} — score: {x['_score']:.2f}\n   {format_card(x)}")
    await update.message.reply_text("🏁 *Ranking Acciones BYMA (proyección 6M)*\n" + "\n\n".join(lines), parse_mode=ParseMode.MARKDOWN)

async def cmd_ranking_cedears(update: Update, context: ContextTypes.DEFAULT_TYPE):
    data = await compute_universe_returns(CEDEARS_UNIVERSO)
    top = pick_top_by_projection(data, topn=5)
    if not top:
        await update.message.reply_text("🏁 No pude armar el ranking de CEDEARs ahora.")
        return
    lines = []
    for i, x in enumerate(top, 1):
        lines.append(f"{i}. {x['symbol']} — score: {x['_score']:.2f}\n   {format_card(x)}")
    await update.message.reply_text("🏁 *Ranking CEDEARs (proyección 6M)*\n" + "\n\n".join(lines), parse_mode=ParseMode.MARKDOWN)

async def cmd_alerta_dolar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # /alerta_dolar <tipo> <umbral>
    msg = (update.message.text or "").strip()
    parts = msg.split()
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
    if ok:
        await update.message.reply_text(f"🗑️ Alerta #{alert_id} eliminada.")
    else:
        await update.message.reply_text("No se encontró esa alerta.")

async def cmd_resumen_diario(update: Update, context: ContextTypes.DEFAULT_TYPE):
    dolares = await get_all_dolares()
    reservas = await get_reservas_usd()
    ipc = await get_ipc_mensual()
    riesgo = await get_riesgo_pais()
    news = await fetch_news_filtered(limit=5)

    # Dólares
    block = format_dolares_block(dolares) if dolares else "n/d"

    # Reservas
    if reservas:
        reservas_txt = f"`{reservas['fecha']}` — {reservas['valor']:.2f} M USD"
    else:
        reservas_txt = "Dato no disponible"

    # Inflación
    if ipc:
        ipc_txt = f"`{ipc['fecha']}` — {ipc['valor']:.2f}%"
    else:
        ipc_txt = "Dato no disponible"

    # Riesgo
    if riesgo:
        riesgo_txt = f"`{riesgo['fecha']}` — {int(round(riesgo['valor']))} pts"
    else:
        riesgo_txt = "Dato no disponible"

    # Noticias
    if news:
        news_lines = [f"• [{n['title']}]({n['link']})" for n in news]
        news_txt = "\n".join(news_lines)
    else:
        news_txt = "_Sin noticias filtradas ahora mismo._"

    out = (
        f"🗓️ *Resumen {now_str()}*\n\n"
        f"💵 *Dólares*\n{block}\n"
        f"🏦 *Reservas*: {reservas_txt}\n"
        f"📈 *Inflación*: {ipc_txt}\n"
        f"📉 *Riesgo País*: {riesgo_txt}\n\n"
        f"📰 *Noticias*\n{news_txt}"
    )
    await update.message.reply_text(out, parse_mode=ParseMode.MARKDOWN, disable_web_page_preview=True)

# ---------- Loop de alertas ----------

async def alert_loop(app):
    await asyncio.sleep(5)  # pequeña espera al iniciar
    while True:
        try:
            rows = db_list_all_active()
            if rows:
                dolares = await get_all_dolares()
                for (alert_id, chat_id, tipo, umbral) in rows:
                    d = dolares.get(tipo) if dolares else None
                    precio = d.get("venta") if d else None
                    if isinstance(precio, (int, float)) and precio >= float(umbral):
                        # disparar alerta
                        txt = f"🔔 *Alerta #{alert_id}*\n{tipo.upper()} alcanzó {fmt_money_ars(precio)} (umbral {fmt_money_ars(umbral)})."
                        try:
                            await app.bot.send_message(chat_id=chat_id, text=txt, parse_mode=ParseMode.MARKDOWN)
                        except Exception:
                            pass
                        # eliminamos la alerta luego de disparar (alertas one-shot)
                        db_delete_alert(chat_id, alert_id)
            # frecuencia de chequeo
            await asyncio.sleep(90)  # 1.5 minutos
        except Exception:
            await asyncio.sleep(120)

# ---------- Main ----------

def build_app():
    token = os.environ.get("BOT_TOKEN")
    if not token:
        raise RuntimeError("Falta BOT_TOKEN en variables de entorno.")
    app = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_start))

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

    return app

async def main():
    db_init()
    app = build_app()
    # lanzamos loop de alertas paralelo
    asyncio.get_event_loop().create_task(alert_loop(app))
    await app.run_polling(close_loop=False)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
