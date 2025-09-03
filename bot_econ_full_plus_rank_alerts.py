#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bot Económico AR — Telegram (Render Free, WEBHOOK estable)
Comandos:
  /dolar, /reservas, /inflacion, /riesgo, /acciones, /cedears,
  /ranking_acciones, /ranking_cedears,
  /alerta_dolar <tipo> <umbral>, /alertas, /alerta_borrar <id>, /resumen_diario
"""

import os
import re
import time
import json
import sqlite3
import asyncio
from datetime import datetime
from xml.etree import ElementTree as ET

import httpx

from telegram import Update, LinkPreviewOptions
from telegram.constants import ParseMode
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters, Defaults
)
from telegram.request import HTTPXRequest

# ===================== Configuración =====================

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
PUBLIC_URL = os.getenv("PUBLIC_URL", "").strip().rstrip("/")
WEBHOOK_PATH = os.getenv("WEBHOOK_PATH", "").strip()  # si está vacío, usamos el BOT_TOKEN
ALERTS_DB_PATH = os.getenv("ALERTS_DB_PATH", "/var/tmp/alerts.db")
NEWS_MAX_ITEMS = int(os.getenv("NEWS_MAX_ITEMS", "5"))

# Caches (TTL) en segundos
TTL_USD = 30
TTL_RIESGO = 300
TTL_RESERVAS = 3600
TTL_IPC = 86400
TTL_YF = 300
TTL_NEWS = 600

# Universos
UNIVERSO_ACCIONES_BA = [
    "GGAL.BA","BMA.BA","BBAR.BA","SUPV.BA",
    "YPFD.BA","PAMP.BA","CEPU.BA","EDN.BA","TGSU2.BA","TGNO4.BA","TRAN.BA",
    "ALUA.BA","TXAR.BA","LOMA.BA","CRES.BA",
    "TECO2.BA","MIRG.BA",
    "VALO.BA","COME.BA","HAVA.BA",
]
UNIVERSO_CEDEARS_BA = [
    "AAPL.BA","AMZN.BA","MSFT.BA","GOOGL.BA","META.BA","TSLA.BA","NVDA.BA",
    "MELI.BA","BRKB.BA","KO.BA","PG.BA","JNJ.BA","PFE.BA","WMT.BA","DIS.BA",
    "NFLX.BA","NKE.BA","INTC.BA","AMD.BA","BIDU.BA"
]
SECTORES = {
    "GGAL.BA":"Bancos","BMA.BA":"Bancos","BBAR.BA":"Bancos","SUPV.BA":"Bancos",
    "YPFD.BA":"Energía","PAMP.BA":"Energía","CEPU.BA":"Energía","EDN.BA":"Energía",
    "TGSU2.BA":"Energía","TGNO4.BA":"Energía","TRAN.BA":"Energía",
    "ALUA.BA":"Industriales","TXAR.BA":"Industriales","LOMA.BA":"Materiales",
    "CRES.BA":"Real Estate","TECO2.BA":"Telecom","MIRG.BA":"Servicios",
    "VALO.BA":"Finanzas","COME.BA":"Consumo","HAVA.BA":"Otros",
}
NOMBRES = {
    "GGAL.BA":"Grupo Galicia","BMA.BA":"Banco Macro","BBAR.BA":"BBVA Arg","SUPV.BA":"Supervielle",
    "YPFD.BA":"YPF","PAMP.BA":"Pampa Energía","CEPU.BA":"Central Puerto","EDN.BA":"Edenor",
    "TGSU2.BA":"Transp. Gas Sur","TGNO4.BA":"Transp. Gas Norte","TRAN.BA":"Transener",
    "ALUA.BA":"Aluar","TXAR.BA":"Ternium Arg","LOMA.BA":"Loma Negra","CRES.BA":"Cresud",
    "TECO2.BA":"Telecom Arg","MIRG.BA":"Mirgor",
    "VALO.BA":"Grupo Valores","COME.BA":"Siderar/Comercial","HAVA.BA":"Havanna",
    "AAPL.BA":"Apple","AMZN.BA":"Amazon","MSFT.BA":"Microsoft","GOOGL.BA":"Alphabet",
    "META.BA":"Meta","TSLA.BA":"Tesla","NVDA.BA":"NVIDIA","MELI.BA":"MercadoLibre",
    "BRKB.BA":"Berkshire","KO.BA":"Coca-Cola","PG.BA":"P&G","JNJ.BA":"J&J","PFE.BA":"Pfizer",
    "WMT.BA":"Walmart","DIS.BA":"Disney","NFLX.BA":"Netflix","NKE.BA":"Nike","INTC.BA":"Intel",
    "AMD.BA":"AMD","BIDU.BA":"Baidu"
}

# Series oficiales
SERIES_RESERVAS = [s for s in [
    os.getenv("SERIES_RESERVAS_ID"),
    "BCRA.RESERVAS_INTERNACIONALES_USD",
    "BCRA.RESERVAS_INTERNACIONALES",
] if s]
SERIES_IPC_MOM = [s for s in [
    os.getenv("SERIES_IPC_MOM_ID"),
    "INDEC.IPC.NIVEL_GENERAL_Variacion_mensual",
    "INDEC.IPC.VAR_MENSUAL",
] if s]

# RSS locales (economía)
RSS_FEEDS = [
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
    "dólar","dolar","inflación","inflacion","reservas","riesgo","mercado",
    "acciones","bonos","tasa","mep","ccl","brecha","blue","oficial","mayorista",
    "pbi","actividad","export","import","ipim","icm"
]
KEYWORDS_BAN = ["salud","fútbol","futbol","quiniela","espectáculo","espectaculo","policial","horóscopo","horoscopo"]

# ===================== Utilidades =====================

_MDV2_RE = re.compile(r'([_*\[\]()~`>#+\-=|{}.!])')
def esc(s: str) -> str:
    return _MDV2_RE.sub(r'\\\1', s or "")

class TTLCache:
    def __init__(self): self.data = {}
    def get(self, key):
        v = self.data.get(key)
        if not v: return None
        exp, val = v
        if exp < time.time():
            self.data.pop(key, None); return None
        return val
    def set(self, key, value, ttl): self.data[key] = (time.time()+ttl, value)

CACHE = TTLCache()

# HTTP cliente (nuestros fetch a APIs)
HTTP_TIMEOUT = httpx.Timeout(connect=10, read=12, write=12, pool=12)
HTTP_LIMITS  = httpx.Limits(max_connections=20, max_keepalive_connections=20)
HTTP_HEADERS = {"User-Agent": "bot-econ-ar/1.0 (+github.com/Siberianok/bot-economico-ar)"}
http = httpx.AsyncClient(timeout=HTTP_TIMEOUT, limits=HTTP_LIMITS, headers=HTTP_HEADERS, follow_redirects=True)

async def fetch_json(url: str, params: dict | None = None):
    for i in range(2):
        try:
            r = await http.get(url, params=params)
            if r.status_code == 200:
                return r.json()
        except httpx.RequestError:
            await asyncio.sleep(0.4*(i+1))
    return None

async def fetch_text(url: str):
    for i in range(2):
        try:
            r = await http.get(url)
            if r.status_code == 200:
                return r.text
        except httpx.RequestError:
            await asyncio.sleep(0.4*(i+1))
    return None

# ===================== Dólares =====================

USD_TYPES = ["blue","mep","ccl","cripto","oficial","mayorista"]

async def get_dolares():
    v = CACHE.get("usd_all")
    if v: return v
    out = {}

    async def _dapi(tipo):
        data = await fetch_json(f"https://dolarapi.com/v1/dolares/{tipo}")
        if data and "compra" in data and "venta" in data:
            return {"buy": float(data["compra"]), "sell": float(data["venta"])}

    async def _cy(tipo):
        data = await fetch_json("https://criptoya.com/api/dolar")
        if not data: return None
        k = {"blue":"blue","mep":"mep","ccl":"ccl","oficial":"oficial","mayorista":"mayorista","cripto":"cripto"}.get(tipo)
        if not k or k not in data or not isinstance(data[k], dict): return None
        bid = data[k].get("bid") or data[k].get("compra") or data[k].get("buy")
        ask = data[k].get("ask") or data[k].get("venta") or data[k].get("sell")
        if bid and ask: return {"buy": float(bid), "sell": float(ask)}
        return None

    for t in USD_TYPES:
        q = await _dapi(t)
        if not q: q = await _cy(t)
        if q: out[t] = q

    if out: CACHE.set("usd_all", out, TTL_USD)
    return out

def fmt_dolares_block(d):
    g = lambda t,k: d.get(t,{}).get(k, 0.0)
    lines = [
        "BLUE      | Compra: $ {b1:>7,.2f} | Venta: $ {s1:>7,.2f}",
        "MEP       | Compra: $ {b2:>7,.2f} | Venta: $ {s2:>7,.2f}",
        "CCL       | Compra: $ {b3:>7,.2f} | Venta: $ {s3:>7,.2f}",
        "CRIPTO    | Compra: $ {b4:>7,.2f} | Venta: $ {s4:>7,.2f}",
        "OFICIAL   | Compra: $ {b5:>7,.2f} | Venta: $ {s5:>7,.2f}",
        "MAYORISTA | Compra: $ {b6:>7,.2f} | Venta: $ {s6:>7,.2f}",
    ]
    s = "\n".join(lines).format(
        b1=g("blue","buy"), s1=g("blue","sell"),
        b2=g("mep","buy"),  s2=g("mep","sell"),
        b3=g("ccl","buy"),  s3=g("ccl","sell"),
        b4=g("cripto","buy"), s4=g("cripto","sell"),
        b5=g("oficial","buy"), s5=g("oficial","sell"),
        b6=g("mayorista","buy"), s6=g("mayorista","sell"),
    )
    return "```\n" + s + "\n```"

# ===================== Series oficiales =====================

async def series_last(ids: str):
    data = await fetch_json("https://apis.datos.gob.ar/series/api/series",
                            {"ids": ids, "limit": 1, "sort": "desc"})
    try:
        if data and "data" in data and data["data"] and len(data["data"][0]) >= 2:
            return float(data["data"][0][1])
    except Exception:
        pass
    return None

async def get_reservas():
    v = CACHE.get("reservas")
    if v is not None: return v
    val = None
    for sid in SERIES_RESERVAS:
        val = await series_last(sid)
        if val is not None: break
    if val is not None: CACHE.set("reservas", val, TTL_RESERVAS)
    return val

async def get_ipc_mom():
    v = CACHE.get("ipc_mom")
    if v is not None: return v
    val = None
    for sid in SERIES_IPC_MOM:
        val = await series_last(sid)
        if val is not None: break
    if val is not None: CACHE.set("ipc_mom", val, TTL_IPC)
    return val

# ===================== Riesgo país =====================

async def get_riesgo_pais():
    v = CACHE.get("riesgo")
    if v is not None: return v
    val = None
    data = await fetch_json("https://api.argentinadatos.com/v1/finanzas/mercados/riesgo-pais")
    try:
        if isinstance(data, dict) and "valor" in data:
            val = int(round(float(data["valor"])))
        elif isinstance(data, list) and data:
            for row in reversed(data):
                if "valor" in row:
                    val = int(round(float(row["valor"])))
                    break
    except Exception:
        val = None
    if val is None:
        dapi = await fetch_json("https://dolarapi.com/v1/otros/riesgo-pais")
        if dapi and "valor" in dapi:
            try: val = int(round(float(dapi["valor"])))
            except Exception: val = None
    if val is not None: CACHE.set("riesgo", val, TTL_RIESGO)
    return val

# ===================== Yahoo Finance =====================

def _pct(a: float, b: float) -> float:
    if a is None or b is None or a == 0: return 0.0
    return (b - a) / a * 100.0

async def yf_spark(symbols: list[str]):
    out = {}
    if not symbols: return out
    syms = ",".join(symbols)
    url = "https://query1.finance.yahoo.com/v7/finance/spark"
    data = await fetch_json(url, {"symbols": syms, "range":"6mo", "interval":"1d"})
    if not data or "spark" not in data or "result" not in data["spark"]: return out
    for item in data["spark"]["result"]:
        sym = item.get("symbol")
        close = item.get("response",[{}])[0].get("indicators",{}).get("quote",[{}])[0].get("close")
        if sym and isinstance(close, list):
            close = [c for c in close if c is not None]
            if len(close) >= 5:
                out[sym] = {"close": close}
    return out

def perf_1m_3m_6m(closes: list[float]):
    if not closes: return 0.0,0.0,0.0
    last = closes[-1]
    def px(n): return closes[max(0, len(closes)-n-1)]
    m1 = _pct(px(21), last)
    m3 = _pct(px(63), last)
    m6 = _pct(px(126), last)
    return m1, m3, m6

async def top_universo(symbols: list[str], topn=3):
    ck = f"yf_{hash(tuple(symbols))}"
    v = CACHE.get(ck)
    if v: return v
    data = await yf_spark(symbols)
    cards = []
    for sym, d in data.items():
        closes = d.get("close", [])
        if not closes: continue
        m1, m3, m6 = perf_1m_3m_6m(closes)
        price = closes[-1]
        nm = NOMBRES.get(sym, sym.replace(".BA",""))
        sector = SECTORES.get(sym, "—")
        cards.append((sym, nm, price, sector, m1, m3, m6))
    cards.sort(key=lambda x: x[5], reverse=True)  # orden por 3m
    top = cards[:topn]
    CACHE.set(ck, top, TTL_YF)
    return top

async def ranking_6m(symbols: list[str], topn=5):
    base = await top_universo(symbols, topn=len(symbols))
    scored = []
    for (sym,nm,price,sector,m1,m3,m6) in base:
        score = 0.1*m1 + 0.3*m3 + 0.6*m6
        scored.append((score, sym, nm, price, sector, m1, m3, m6))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:topn]

def fmt_card(sym, nm, price, sector, m1, m3, m6):
    t = esc(sym.replace(".BA","")); n = esc(nm); p = f"{price:,.2f}"; s = esc(sector)
    def sign(x): return ("+" if x>=0 else "") + f"{x:,.2f}%"
    return f"*{t}* \\({n}\\) — ${p}\n• {s}\nRendimientos: {esc(sign(m1))} · {esc(sign(m3))} · {esc(sign(m6))}"

# ===================== Noticias (RSS) =====================

def _pass_noticia(title: str):
    t = (title or "").lower()
    if any(b in t for b in KEYWORDS_BAN): return False
    if any(k in t for k in KEYWORDS_OK): return True
    return False

def _escape_url(u: str) -> str:
    # Para MarkdownV2: escapamos paréntesis en URLs para evitar parse errors
    return u.replace(")", "%29").replace("(", "%28")

async def get_news():
    v = CACHE.get("news")
    if v: return v
    items = []
    for url in RSS_FEEDS:
        txt = await fetch_text(url)
        if not txt: continue
        try:
            root = ET.fromstring(txt)
            for it in root.findall(".//item"):
                title = (it.findtext("title") or "").strip()
                link = (it.findtext("link") or "").strip()
                if title and link and link.startswith("http") and _pass_noticia(title):
                    items.append((title, link))
        except ET.ParseError:
            continue
    seen = set(); uniq = []
    for t, u in items:
        k = t.strip().lower()
        if k in seen: continue
        seen.add(k); uniq.append((t, u))
    out = uniq[:NEWS_MAX_ITEMS]
    CACHE.set("news", out, TTL_NEWS)
    return out

def fmt_news_links(items):
    if not items: return "Sin noticias filtradas ahora."
    return "\n".join(f"• [{esc(t)}]({_escape_url(u)})" for t, u in items)

# ===================== Alertas (sqlite) =====================

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
    conn.commit(); conn.close()

def db_add_alert(chat_id: int, tipo: str, umbral: float):
    conn = sqlite3.connect(ALERTS_DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO alerts (chat_id,tipo,umbral) VALUES (?,?,?)", (chat_id, tipo, umbral))
    conn.commit(); conn.close()

def db_list_alerts(chat_id: int):
    conn = sqlite3.connect(ALERTS_DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id,tipo,umbral FROM alerts WHERE chat_id=?", (chat_id,))
    rows = c.fetchall(); conn.close(); return rows

def db_del_alert(chat_id: int, alert_id: int):
    conn = sqlite3.connect(ALERTS_DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM alerts WHERE chat_id=? AND id=?", (chat_id, alert_id))
    conn.commit(); ok = c.rowcount > 0; conn.close(); return ok

async def job_check_alerts(context: ContextTypes.DEFAULT_TYPE):
    usd = await get_dolares()
    if not usd: return
    conn = sqlite3.connect(ALERTS_DB_PATH); c = conn.cursor()
    c.execute("SELECT id,chat_id,tipo,umbral FROM alerts"); rows = c.fetchall(); conn.close()
    for alert_id, chat_id, tipo, umbral in rows:
        q = usd.get(tipo); 
        if not q: continue
        precio = q.get("sell") or q.get("buy")
        if precio and float(precio) >= float(umbral):
            try:
                txt = f"🔔 *Alerta dólar {esc(tipo.upper())}*\nPrecio: ${float(precio):,.2f} ≥ umbral ${float(umbral):,.2f}"
                await context.bot.send_message(chat_id=chat_id, text=txt)
                conn = sqlite3.connect(ALERTS_DB_PATH); c = conn.cursor()
                c.execute("DELETE FROM alerts WHERE id=?", (alert_id,))
                conn.commit(); conn.close()
            except Exception:
                pass

async def job_prefetch(context: ContextTypes.DEFAULT_TYPE):
    for fn in (get_dolares, get_reservas, get_ipc_mom, get_riesgo_pais, get_news):
        try: await fn()
        except Exception: pass
    try: await top_universo(UNIVERSO_ACCIONES_BA, topn=6)
    except Exception: pass
    try: await top_universo(UNIVERSO_CEDEARS_BA, topn=6)
    except Exception: pass

# ===================== Handlers =====================

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = (
        "👋 *Observatorio Económico AR*\n\n"
        "Comandos:\n"
        "```\n"
        "/dolar\n/reservas\n/inflacion\n/riesgo\n/acciones\n/cedears\n"
        "/ranking_acciones\n/ranking_cedears\n"
        "/alerta_dolar <tipo> <umbral>\n/alertas\n/alerta_borrar <id>\n/resumen_diario\n"
        "```\n"
        "Tipos válidos alerta dólar: blue|mep|ccl|cripto|oficial|mayorista"
    )
    await update.message.reply_text(txt)

async def cmd_dolar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    d = await get_dolares()
    if not d:
        await update.message.reply_text("No pude obtener cotizaciones ahora.")
        return
    await update.message.reply_text("💵 *Dólares*\n" + fmt_dolares_block(d))

async def cmd_reservas(update: Update, context: ContextTypes.DEFAULT_TYPE):
    v = await get_reservas()
    if v is None: await update.message.reply_text("💹 *Reservas BCRA*: dato no disponible ahora (fuente oficial).")
    else:        await update.message.reply_text(f"💹 *Reservas BCRA*: USD {v:,.0f}")

async def cmd_inflacion(update: Update, context: ContextTypes.DEFAULT_TYPE):
    v = await get_ipc_mom()
    if v is None: await update.message.reply_text("📈 *Inflación (variación mensual)*: dato no disponible.")
    else:        await update.message.reply_text(f"📈 *Inflación (variación mensual)*: {v:,.2f}%")

async def cmd_riesgo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    v = await get_riesgo_pais()
    if v is None: await update.message.reply_text("📉 *Riesgo país*: dato no disponible.")
    else:        await update.message.reply_text(f"📉 *Riesgo país*: {v:,} pb")

async def _cards_reply(update, title, items):
    if not items:
        await update.message.reply_text(title + ": no hay datos ahora."); return
    lines = []
    for (sym, nm, price, sector, m1, m3, m6) in items:
        lines.append("🪪 " + fmt_card(sym, nm, price, sector, m1, m3, m6))
    await update.message.reply_text(f"{title}\n\n" + "\n\n".join(lines))

async def cmd_acciones(update: Update, context: ContextTypes.DEFAULT_TYPE):
    top = await top_universo(UNIVERSO_ACCIONES_BA, topn=3)
    await _cards_reply(update, "📊 *Top 3 Acciones BYMA (3m)*", top)

async def cmd_cedears(update: Update, context: ContextTypes.DEFAULT_TYPE):
    top = await top_universo(UNIVERSO_CEDEARS_BA, topn=3)
    await _cards_reply(update, "🌐 *Top 3 CEDEARs (3m)*", top)

async def cmd_rank_acc(update: Update, context: ContextTypes.DEFAULT_TYPE):
    rk = await ranking_6m(UNIVERSO_ACCIONES_BA, topn=5)
    if not rk:
        await update.message.reply_text("🏆 *Ranking Acciones BYMA*: no hay datos ahora."); return
    lines = []
    for (score, sym, nm, price, sector, m1, m3, m6) in rk:
        lines.append("🏅 " + fmt_card(sym, nm, price, sector, m1, m3, m6))
    await update.message.reply_text("🏆 *Ranking 6M Acciones BYMA*\n\n" + "\n\n".join(lines))

async def cmd_rank_ced(update: Update, context: ContextTypes.DEFAULT_TYPE):
    rk = await ranking_6m(UNIVERSO_CEDEARS_BA, topn=5)
    if not rk:
        await update.message.reply_text("🏆 *Ranking CEDEARs*: no hay datos ahora."); return
    lines = []
    for (score, sym, nm, price, sector, m1, m3, m6) in rk:
        lines.append("🏅 " + fmt_card(sym, nm, price, sector, m1, m3, m6))
    await update.message.reply_text("🏆 *Ranking 6M CEDEARs*\n\n" + "\n\n".join(lines))

async def cmd_alerta_dolar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) != 2:
        await update.message.reply_text("Uso:\n```\n/alerta_dolar <tipo> <umbral>\nEj: /alerta_dolar blue 1500\n```"); return
    tipo = context.args[0].lower().strip()
    if tipo not in USD_TYPES:
        await update.message.reply_text("Tipos válidos:\n```\nblue | mep | ccl | cripto | oficial | mayorista\n```"); return
    try:
        umbral = float(context.args[1].replace(",", "."))
    except ValueError:
        await update.message.reply_text("Umbral inválido. Ej: 1500"); return
    db_add_alert(update.effective_chat.id, tipo, umbral)
    await update.message.reply_text(f"✅ Alerta creada: {esc(tipo.upper())} ≥ ${umbral:,.2f}")

async def cmd_alertas(update: Update, context: ContextTypes.DEFAULT_TYPE):
    rows = db_list_alerts(update.effective_chat.id)
    if not rows: await update.message.reply_text("No tenés alertas activas."); return
    lines = [f"#{i} — {esc(t.upper())} ≥ ${u:,.2f}" for (i,t,u) in rows]
    await update.message.reply_text("*Alertas activas*\n" + "\n".join(lines))

async def cmd_alerta_borrar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Uso:\n```\n/alerta_borrar <id>\n```"); return
    try:
        a_id = int(context.args[0])
    except ValueError:
        await update.message.reply_text("ID inválido."); return
    ok = db_del_alert(update.effective_chat.id, a_id)
    await update.message.reply_text("🗑️ Alerta eliminada." if ok else "No encontré esa alerta.")

async def cmd_resumen_diario(update: Update, context: ContextTypes.DEFAULT_TYPE):
    usd, reservas, ipc, riesgo, news = await asyncio.gather(
        get_dolares(), get_reservas(), get_ipc_mom(), get_riesgo_pais(), get_news()
    )
    lines = [f"🗓️ *Resumen {datetime.now().strftime('%Y-%m-%d %H:%M')}*"]
    if usd:
        lines.append("💵 *Dólares*"); lines.append(fmt_dolares_block(usd))
    else:
        lines.append("💵 *Dólares*: no disponible")
    lines.append("💹 *Reservas*: " + (f"USD {reservas:,.0f}" if reservas is not None else "Dato no disponible"))
    lines.append("📈 *Inflación*: " + (f"{ipc:,.2f}%" if ipc is not None else "Dato no disponible"))
    lines.append("📉 *Riesgo País*: " + (f"{riesgo:,} pb" if riesgo is not None else "Dato no disponible"))
    lines.append("\n📰 *Noticias*"); lines.append(fmt_news_links(news))
    await update.message.reply_text("\n".join(lines))

# ===================== App (Defaults + Jobs + Timeouts PTB) =====================

def build_app():
    if not BOT_TOKEN:
        raise RuntimeError("Falta BOT_TOKEN en variables de entorno.")
    defaults = Defaults(
        parse_mode=ParseMode.MARKDOWN_V2,
        link_preview_options=LinkPreviewOptions(is_disabled=True),
    )
    # Timeouts más altos al hablar con Telegram (evita TimedOut en cold start)
    req = HTTPXRequest(connect_timeout=30.0, read_timeout=30.0, write_timeout=30.0, pool_timeout=30.0)
    app = ApplicationBuilder().token(BOT_TOKEN).request(req).defaults(defaults).build()

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
    app.add_handler(MessageHandler(filters.COMMAND, cmd_start))

    if app.job_queue:
        app.job_queue.run_repeating(job_prefetch, interval=300, first=5)
        app.job_queue.run_repeating(job_check_alerts, interval=90, first=15)
    return app

# ===================== MAIN — WEBHOOK simple =====================

def main():
    db_init()
    if not PUBLIC_URL or not PUBLIC_URL.startswith("http"):
        raise RuntimeError("Falta PUBLIC_URL (ej: https://bot-economico-ar.onrender.com)")
    secret_path = WEBHOOK_PATH or BOT_TOKEN
    port = int(os.getenv("PORT", "10000"))

    app = build_app()
    # Bloqueante; PTB maneja su propio event loop. No hacer reintentos acá.
    app.run_webhook(
        listen="0.0.0.0",
        port=port,
        url_path=secret_path,
        webhook_url=f"{PUBLIC_URL}/{secret_path}",
        drop_pending_updates=True,
        stop_signals=None,
    )

if __name__ == "__main__":
    main()
