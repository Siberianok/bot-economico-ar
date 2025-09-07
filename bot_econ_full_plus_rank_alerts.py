# bot_econ_full_plus_rank_alerts.py
# Bot Económico AR — Webhook only (Render)
# Comandos: /dolar /reservas /inflacion /riesgo /acciones /cedears
#           /ranking_acciones /ranking_cedears
#           /resumen_diario
#           /alerta_dolar <tipo> <umbral> /alertas /alerta_borrar <id>
#
# Notas:
# - Usa PTB 20.8 con extras [job-queue, webhooks] (ver requirements.txt).
# - Sólo webhooks (NO polling). No mezclar getUpdates.
# - Formato de salida en HTML (links clickeables y <code> monoespaciado).
# - Caché con TTL + precarga vía JobQueue para bajar latencia.
# - Dólares: DolarAPI (principal) / CriptoYa (fallback).
# - Reservas/Inflación: apis.datos.gob.ar (series oficiales). (Si la serie falla, se muestra “Dato no disponible”)
# - Riesgo país: ArgentinaDatos (fallback DolarAPI si no hay).
# - Precios y rankings: Yahoo Finance (spark chart).
# - Noticias: RSS locales (títulos con link), filtradas por keywords económicas.

import os
import sys
import re
import json
import math
import time
import html
import asyncio
import logging
from datetime import datetime, timedelta, timezone

import httpx
from xml.etree import ElementTree as ET

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    ApplicationBuilder,
    Application,
    Defaults,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
    JobQueue,
)

# =========================
# CONFIG & CONSTANTES
# =========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("bot-econ-ar")

TZ_AR = timezone(timedelta(hours=-3))  # America/Argentina/Buenos_Aires

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
PUBLIC_URL = os.getenv("PUBLIC_URL", "").strip().rstrip("/")
WEBHOOK_PATH = os.getenv("WEBHOOK_PATH", "tgwebhook").strip()

if not BOT_TOKEN:
    log.error("Falta BOT_TOKEN en variables de entorno.")
    sys.exit(1)
if not PUBLIC_URL.startswith("http"):
    log.error("Falta PUBLIC_URL válida (ej: https://bot-economico-ar.onrender.com).")
    sys.exit(1)

# Tiempo de espera por request externo
HTTP_TIMEOUT = httpx.Timeout(connect=6.0, read=8.0, write=8.0, pool=6.0)

# TTLs de cache
TTL_DOLARES = 60            # 1 min
TTL_RESERVAS = 6 * 3600     # 6 hs
TTL_INFLACION = 24 * 3600   # 1 día
TTL_RIESGO = 900            # 15 min
TTL_YF = 900                # 15 min
TTL_NEWS = 600              # 10 min

# Universo BYMA (ampliado) y CEDEARs (grandes) — usaremos estos por defecto
BYMA_TICKERS = [
    "GGAL.BA","BMA.BA","SUPV.BA","BBAR.BA","PAMP.BA","CEPU.BA","EDN.BA",
    "YPFD.BA","TGSU2.BA","TGNO4.BA","TRAN.BA","ALUA.BA","HARG.BA","TXAR.BA",
    "LOMA.BA","CRES.BA","COME.BA","BYMA.BA","MIRG.BA","VALO.BA"
]
CEDEAR_TICKERS = [
    "AAPL.BA","MSFT.BA","AMZN.BA","TSLA.BA","NVDA.BA","META.BA","GOOGL.BA",
    "KO.BA","JPM.BA","PFE.BA","XOM.BA","WMT.BA","DIS.BA"
]

# Sectores (básico) para tarjetas (si no se conoce, se deja “—”)
SECTOR_MAP = {
    "GGAL.BA": "Bancos", "BMA.BA": "Bancos", "SUPV.BA": "Bancos", "BBAR.BA":"Bancos",
    "PAMP.BA": "Energía", "CEPU.BA": "Energía", "EDN.BA":"Energía", "YPFD.BA":"Energía",
    "TGSU2.BA":"Energía", "TGNO4.BA":"Energía", "TRAN.BA":"Energía",
    "ALUA.BA":"Industriales","HARG.BA":"Real Estate","TXAR.BA":"Industriales",
    "LOMA.BA":"Materiales","CRES.BA":"Real Estate","COME.BA":"Telecom","BYMA.BA":"Servicios Fin.",
    "MIRG.BA":"Consumo","VALO.BA":"Bancos",
}

# RSS locales (puede que alguno no responda; con 2-3 alcanza)
RSS_SOURCES = [
    "https://www.ambito.com/rss/economia.xml",
    "https://www.cronista.com/files/rss/economia.xml",
    "https://www.baenegocios.com/rss/tema/economia.xml",
    "https://www.lanacion.com.ar/rss/economia/",
    "https://www.telam.com.ar/rss2/economia.xml",
    "https://www.iprofesional.com/rss/economia",
]

NEWS_KEYWORDS = [
    "dólar","dolar","inflación","reservas","BCRA","riesgo","MEP","CCL","blue",
    "bonos","deuda","suba","baja","tasa","IPC","INDEC","actividad","PBI",
]

# =========================
# HELPERS
# =========================

_cache = {}  # key -> (timestamp, data)

def _cache_get(key: str, ttl: int):
    now = time.time()
    item = _cache.get(key)
    if not item:
        return None
    ts, data = item
    if now - ts <= ttl:
        return data
    return None

def _cache_set(key: str, data):
    _cache[key] = (time.time(), data)

def fmt_pct(x):
    try:
        return f"{x:+.2f}%"
    except:
        return "—"

def fmt_num(x):
    try:
        return f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except:
        return "—"

def html_code(s: str) -> str:
    return f"<code>{html.escape(s)}</code>"

def as_link(title: str, url: str) -> str:
    t = html.escape(title)
    u = html.escape(url, quote=True)
    return f'<a href="{u}">{t}</a>'

async def fetch_json(client: httpx.AsyncClient, url: str):
    r = await client.get(url)
    r.raise_for_status()
    return r.json()

async def fetch_text(client: httpx.AsyncClient, url: str):
    r = await client.get(url)
    r.raise_for_status()
    return r.text

# =========================
# DATA FETCHERS
# =========================

async def get_dolares() -> dict:
    """Devuelve {'blue': {...}, 'mep': {...}, 'ccl': {...}, 'cripto': {...}, 'oficial': {...}, 'mayorista': {...}}"""
    cache_key = "dolares"
    data = _cache_get(cache_key, TTL_DOLARES)
    if data:
        return data

    out = {}
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        base = "https://dolarapi.com/v1/dolares"
        tipos = {
            "blue":"blue",
            "mep":"mep",
            "ccl":"ccl",
            "cripto":"cripto",
            "oficial":"oficial",
            "mayorista":"mayorista",
        }
        try:
            for k, path in tipos.items():
                try:
                    j = await fetch_json(client, f"{base}/{path}")
                    out[k] = {
                        "nombre": j.get("nombre", k),
                        "compra": j.get("compra"),
                        "venta": j.get("venta"),
                        "fecha": j.get("fechaActualizacion") or j.get("fecha"),
                    }
                except Exception as e:
                    log.warning(f"DolarAPI fallo {k}: {e}")
        except Exception as e:
            log.error(f"DolarAPI general: {e}")

        # Fallback rápido desde CriptoYa si algo quedó vacío
        try:
            if any(out.get(k) is None for k in tipos.keys()):
                j = await fetch_json(client, "https://criptoya.com/api/dolar")
                # mapeo aproximado
                out.setdefault("blue", {"nombre":"Blue","venta": j.get("blue")})
                out.setdefault("oficial", {"nombre":"Oficial","venta": j.get("oficial")})
                out.setdefault("mep", {"nombre":"MEP","venta": j.get("mep")})
                out.setdefault("ccl", {"nombre":"CCL","venta": j.get("ccl")})
                out.setdefault("mayorista", {"nombre":"Mayorista","venta": j.get("mayorista")})
                out.setdefault("cripto", {"nombre":"Cripto","venta": j.get("usdt") or j.get("crypto")})
        except Exception as e:
            log.warning(f"CriptoYa fallback error: {e}")

    _cache_set(cache_key, out)
    return out

async def get_reservas() -> dict | None:
    """Reservas BCRA (último dato) desde apis.datos.gob.ar (serie oficial).
       Si falla, devuelve None."""
    cache_key = "reservas"
    c = _cache_get(cache_key, TTL_RESERVAS)
    if c:
        return c

    # Serie oficial (INTERNATIONAL RESERVES – BCRA)
    # Referencia comúnmente usada: "BCRA.RRNR" (ejemplo). Si no está disponible,
    # probamos variantes más difundidas.
    series_ids = [
        "BCRA.RRNR",        # Reservas Internacionales netas? (nombres varían)
        "BCRA.RR",          # Total Reservas
        "BCRA.RI_M",        # otra denominación usual
    ]
    base = "https://apis.datos.gob.ar/series/api/series/?limit=1&format=json&ids="
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        for sid in series_ids:
            try:
                j = await fetch_json(client, base + sid)
                data = j.get("data", [])
                if data and data[0]:
                    # data: [[fecha, valor]]
                    fecha_num, valor = data[0][0], data[0][1]
                    # La API devuelve fecha como 2025-08-31 o epoch; normalizamos
                    if isinstance(fecha_num, str):
                        f = fecha_num
                    else:
                        # epoch en ms
                        f = datetime.fromtimestamp(fecha_num/1000, tz=TZ_AR).strftime("%Y-%m-%d")
                    out = {"serie": sid, "fecha": f, "valor": valor}
                    _cache_set(cache_key, out)
                    return out
            except Exception as e:
                log.warning(f"Reservas fallo serie {sid}: {e}")
    return None

async def get_inflacion() -> dict | None:
    """Inflación (variación mensual última – IPC INDEC)"""
    cache_key = "inflacion"
    c = _cache_get(cache_key, TTL_INFLACION)
    if c:
        return c

    # Serie IPC variación mensual (INDEC). Distintas instalaciones tienen IDs distintos;
    # probamos ids frecuentes en apis.datos.gob.ar. Si ninguna responde, devolvemos None.
    series_ids = [
        "148.3_IPC_2_M_2016_100",  # Nivel general IPC (índice, base 2016=100)
        "148.3_1_1_0_M_2016_100",  # variante
    ]
    base = "https://apis.datos.gob.ar/series/api/series/?limit=2&format=json&ids="
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        for sid in series_ids:
            try:
                j = await fetch_json(client, base + sid)
                data = j.get("data", [])
                if len(data) >= 2:
                    # Tomamos los 2 últimos puntos y calculamos variación mensual %
                    # (indice_t / indice_{t-1} - 1) * 100
                    x0 = data[0][1]
                    x1 = data[1][1]
                    if x0 and x1 and x1 != 0:
                        var = (x0 / x1 - 1) * 100.0
                        fecha = data[0][0]
                        if isinstance(fecha, str):
                            f = fecha[:7]  # YYYY-MM
                        else:
                            f = datetime.fromtimestamp(fecha/1000, tz=TZ_AR).strftime("%Y-%m")
                        out = {"serie": sid, "fecha": f, "variacion_mensual": var}
                        _cache_set(cache_key, out)
                        return out
            except Exception as e:
                log.warning(f"Inflacion fallo serie {sid}: {e}")
    return None

async def get_riesgo_pais() -> dict | None:
    """Riesgo país último valor."""
    cache_key = "riesgo"
    c = _cache_get(cache_key, TTL_RIESGO)
    if c:
        return c

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        # Intento 1: ArgentinaDatos (si endpoint disponible)
        try:
            # Endpoint típico (puede variar). Si falla, capturamos y probamos fallback.
            j = await fetch_json(client, "https://api.argentinadatos.com/v1/finanzas/mercado/riesgo-pais")
            # Suponemos { "fecha": "...", "valor": 1234 } o lista
            if isinstance(j, dict) and "valor" in j:
                _cache_set(cache_key, j)
                return j
            if isinstance(j, list) and j:
                ult = j[-1]
                _cache_set(cache_key, ult)
                return ult
        except Exception as e:
            log.warning(f"ArgentinaDatos riesgo fallo: {e}")

        # Fallback 2: DolarAPI (riesgo-pais general)
        try:
            j = await fetch_json(client, "https://dolarapi.com/v1/finanzas/riesgo-pais")
            _cache_set(cache_key, j)
            return j
        except Exception as e:
            log.warning(f"DolarAPI riesgo fallo: {e}")

    return None

# ------------ Yahoo Finance ---------------

async def yf_get_prices(ticker: str) -> list[tuple[datetime, float]]:
    """Devuelve [(fecha, close), ...] 6 meses diarios aprox."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range=6mo&interval=1d"
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        j = await fetch_json(client, url)
    result = []
    try:
        chart = j["chart"]["result"][0]
        ts = chart["timestamp"]
        closes = chart["indicators"]["quote"][0]["close"]
        for t, c in zip(ts, closes):
            if c is not None:
                result.append((datetime.fromtimestamp(t, tz=TZ_AR), float(c)))
    except Exception as e:
        log.warning(f"Yahoo parse {ticker} fallo: {e}")
    return result

def _pct_change(prices: list[tuple[datetime,float]], days: int) -> float | None:
    if not prices:
        return None
    cutoff = datetime.now(TZ_AR) - timedelta(days=days)
    # precio más cercano <= hoy, y el más cercano <= cutoff
    last = None
    base = None
    for d, p in prices:
        if d <= datetime.now(TZ_AR):
            last = p
        if d <= cutoff:
            base = p
    if last is None or base is None or base == 0:
        return None
    return (last/base - 1.0) * 100.0

async def build_perf_card(ticker: str) -> dict | None:
    """Arma tarjeta: {'ticker','empresa','precio','sector','r1m','r3m','r6m'}"""
    prices = await yf_get_prices(ticker)
    if not prices:
        return None
    r1 = _pct_change(prices, 30)
    r3 = _pct_change(prices, 90)
    r6 = _pct_change(prices, 180)
    last_price = prices[-1][1]
    empresa = ticker.replace(".BA","")
    sector = SECTOR_MAP.get(ticker, "—")
    return {
        "ticker": ticker,
        "empresa": empresa,
        "precio": last_price,
        "sector": sector,
        "r1m": r1,
        "r3m": r3,
        "r6m": r6,
    }

async def build_top_n(tickers: list[str], n: int = 3) -> list[dict]:
    """Top N por rendimiento 3M."""
    cache_key = f"top:{','.join(tickers)}"
    c = _cache_get(cache_key, TTL_YF)
    if c:
        return c[:n]
    out = []
    # concurrencia limitada
    sem = asyncio.Semaphore(6)
    async def one(t):
        async with sem:
            try:
                card = await build_perf_card(t)
                if card:
                    out.append(card)
            except Exception as e:
                log.warning(f"perf {t} error: {e}")
    await asyncio.gather(*[one(t) for t in tickers])
    out.sort(key=lambda x: (x["r3m"] if x["r3m"] is not None else -9999), reverse=True)
    _cache_set(cache_key, out)
    return out[:n]

async def build_ranking(tickers: list[str], n: int = 5) -> list[dict]:
    """Score = 0.1*1m + 0.3*3m + 0.6*6m."""
    cache_key = f"rank:{','.join(tickers)}"
    c = _cache_get(cache_key, TTL_YF)
    if c:
        return c[:n]
    out = []
    sem = asyncio.Semaphore(6)
    async def one(t):
        async with sem:
            try:
                card = await build_perf_card(t)
                if card:
                    r1 = card["r1m"] or 0
                    r3 = card["r3m"] or 0
                    r6 = card["r6m"] or 0
                    score = 0.1*r1 + 0.3*r3 + 0.6*r6
                    card["score"] = score
                    out.append(card)
            except Exception as e:
                log.warning(f"rank {t} error: {e}")
    await asyncio.gather(*[one(t) for t in tickers])
    out.sort(key=lambda x: x.get("score",-9999), reverse=True)
    _cache_set(cache_key, out)
    return out[:n]

# ------------ Noticias (RSS) ---------------

def _rss_parse(xml_text: str) -> list[dict]:
    items = []
    try:
        root = ET.fromstring(xml_text)
        # RSS 2.0
        for item in root.findall(".//item"):
            title = item.findtext("title") or ""
            link = item.findtext("link") or ""
            if title and link:
                items.append({"title": title.strip(), "link": link.strip()})
    except Exception as e:
        log.warning(f"RSS parse error: {e}")
    return items

async def get_news() -> list[dict]:
    cache_key = "news"
    c = _cache_get(cache_key, TTL_NEWS)
    if c:
        return c
    out = []
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        for url in RSS_SOURCES:
            try:
                xml_text = await fetch_text(client, url)
                items = _rss_parse(xml_text)
                # filtrar por keywords económicas
                for it in items:
                    t = it["title"].lower()
                    if any(k in t for k in NEWS_KEYWORDS):
                        out.append(it)
            except Exception as e:
                log.warning(f"RSS fallo {url}: {e}")

    # de-dup por título
    seen = set()
    uniq = []
    for it in out:
        k = it["title"]
        if k not in seen:
            uniq.append(it)
            seen.add(k)
    uniq = uniq[:10]  # recorte
    _cache_set(cache_key, uniq)
    return uniq

# =========================
# ALERTAS (simple en memoria)
# =========================

_alerts = {}  # chat_id -> list[{"id":int,"tipo":str,"umbral":float}]

def _new_alert_id(chat_id: int) -> int:
    lst = _alerts.get(chat_id, [])
    return (max([x["id"] for x in lst], default=0) + 1) if lst else 1

async def job_check_alerts(context: ContextTypes.DEFAULT_TYPE):
    try:
        rates = await get_dolares()
        for chat_id, lst in list(_alerts.items()):
            send_msgs = []
            for a in lst:
                tipo = a["tipo"]
                umbral = a["umbral"]
                val = None
                slot = rates.get(tipo)
                if slot:
                    val = slot.get("venta") or slot.get("compra")
                if isinstance(val, (int,float)) and val >= umbral:
                    send_msgs.append(f"🔔 Alerta {tipo.upper()} cruzó {fmt_num(umbral)} → {fmt_num(val)}")
            if send_msgs:
                try:
                    await context.bot.send_message(chat_id=chat_id, text="\n".join(send_msgs))
                except Exception as e:
                    log.warning(f"alert send fail {chat_id}: {e}")
    except Exception as e:
        log.warning(f"job_check_alerts error: {e}")

# =========================
# FORMATO MENSAJES
# =========================

def _card_line(card: dict) -> str:
    t = html.escape(card["ticker"])
    emp = html.escape(card["empresa"])
    price = fmt_num(card["precio"])
    sec = html.escape(card.get("sector","—"))
    r1 = fmt_pct(card["r1m"]) if card["r1m"] is not None else "—"
    r3 = fmt_pct(card["r3m"]) if card["r3m"] is not None else "—"
    r6 = fmt_pct(card["r6m"]) if card["r6m"] is not None else "—"
    return (
        f"<b>{t}</b> ({emp}) — {price}\n"
        f"• {sec}\n"
        f"Rendimientos: {r1} · {r3} · {r6}"
    )

# =========================
# HANDLERS
# =========================

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = (
        "¡Hola! Soy el <b>Bot Económico AR</b> 🇦🇷\n\n"
        "Comandos:\n"
        "/dolar — Cotizaciones (Blue, MEP, CCL, Cripto, Oficial, Mayorista)\n"
        "/reservas — Reservas BCRA\n"
        "/inflacion — Inflación (variación mensual)\n"
        "/riesgo — Riesgo país\n"
        "/acciones — Top 3 BYMA (1m/3m/6m)\n"
        "/cedears — Top 3 CEDEARs (1m/3m/6m)\n"
        "/ranking_acciones — Top 5 por proyección 6M\n"
        "/ranking_cedears — Top 5 por proyección 6M\n"
        "/alerta_dolar &lt;tipo&gt; &lt;umbral&gt;\n"
        "/alertas — Ver alertas\n"
        "/alerta_borrar &lt;id&gt;\n"
        "/resumen_diario — Dólares + Reservas + Inflación + Riesgo + Noticias\n"
    )
    await update.message.reply_text(txt)

async def cmd_dolar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    rates = await get_dolares()
    lines = ["💵 <b>Dólares</b>"]
    order = ["blue","mep","ccl","cripto","oficial","mayorista"]
    for k in order:
        r = rates.get(k)
        if not r:
            lines.append(html_code(f"{k.upper():10s}  Dato no disponible"))
            continue
        compra = r.get("compra")
        venta = r.get("venta")
        if compra and venta:
            s = f"{k.upper():10s}  {fmt_num(compra)} / {fmt_num(venta)}"
        elif venta:
            s = f"{k.upper():10s}  — / {fmt_num(venta)}"
        else:
            s = f"{k.upper():10s}  Dato no disponible"
        lines.append(html_code(s))
    await update.message.reply_text("\n".join(lines))

async def cmd_reservas(update: Update, context: ContextTypes.DEFAULT_TYPE):
    r = await get_reservas()
    if not r:
        await update.message.reply_text("🏦 <b>Reservas BCRA</b>\n" + html_code("Dato no disponible"))
        return
    lines = [
        "🏦 <b>Reservas BCRA</b>",
        html_code(f"Serie: {r.get('serie','—')}"),
        html_code(f"Fecha: {r.get('fecha','—')}"),
        html_code(f"Valor: USD {fmt_num(r.get('valor'))}"),
    ]
    await update.message.reply_text("\n".join(lines))

async def cmd_inflacion(update: Update, context: ContextTypes.DEFAULT_TYPE):
    r = await get_inflacion()
    if not r:
        await update.message.reply_text("📈 <b>Inflación</b>\n" + html_code("Dato no disponible"))
        return
    lines = [
        "📈 <b>Inflación (variación mensual)</b>",
        html_code(f"Período: {r.get('fecha','—')}"),
        html_code(f"Variación: {fmt_pct(r.get('variacion_mensual'))}"),
    ]
    await update.message.reply_text("\n".join(lines))

async def cmd_riesgo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    r = await get_riesgo_pais()
    if not r:
        await update.message.reply_text("📉 <b>Riesgo País</b>\n" + html_code("Dato no disponible"))
        return
    val = r.get("valor") or r.get("riesgo") or r.get("ultimo") or r.get("value")
    fecha = r.get("fecha") or r.get("date")
    lines = [
        "📉 <b>Riesgo País</b>",
        html_code(f"Último: {fmt_num(val)} pb" if val is not None else "Dato no disponible"),
        html_code(f"Fecha: {fecha or '—'}")
    ]
    await update.message.reply_text("\n".join(lines))

async def cmd_acciones(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cards = await build_top_n(BYMA_TICKERS, n=3)
    if not cards:
        await update.message.reply_text("📊 <b>Acciones BYMA</b>\n" + html_code("Datos no disponibles"))
        return
    lines = ["📊 <b>Top 3 Acciones BYMA (3M)</b>"]
    for c in cards:
        lines.append(_card_line(c))
    await update.message.reply_text("\n\n".join(lines))

async def cmd_cedears(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cards = await build_top_n(CEDEAR_TICKERS, n=3)
    if not cards:
        await update.message.reply_text("🌎 <b>CEDEARs</b>\n" + html_code("Datos no disponibles"))
        return
    lines = ["🌎 <b>Top 3 CEDEARs (3M)</b>"]
    for c in cards:
        lines.append(_card_line(c))
    await update.message.reply_text("\n\n".join(lines))

async def cmd_ranking_acc(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cards = await build_ranking(BYMA_TICKERS, n=5)
    if not cards:
        await update.message.reply_text("🏁 <b>Ranking Acciones</b>\n" + html_code("Datos no disponibles"))
        return
    lines = ["🏁 <b>Ranking Acciones BYMA (score 6M)</b>"]
    for i, c in enumerate(cards, 1):
        score = c.get("score")
        score_s = f"{score:+.2f}" if score is not None else "—"
        lines.append(f"<b>{i}.</b> " + _card_line(c) + f"\nScore: {html_code(score_s)}")
    await update.message.reply_text("\n\n".join(lines))

async def cmd_ranking_ced(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cards = await build_ranking(CEDEAR_TICKERS, n=5)
    if not cards:
        await update.message.reply_text("🏁 <b>Ranking CEDEARs</b>\n" + html_code("Datos no disponibles"))
        return
    lines = ["🏁 <b>Ranking CEDEARs (score 6M)</b>"]
    for i, c in enumerate(cards, 1):
        score = c.get("score")
        score_s = f"{score:+.2f}" if score is not None else "—"
        lines.append(f"<b>{i}.</b> " + _card_line(c) + f"\nScore: {html_code(score_s)}")
    await update.message.reply_text("\n\n".join(lines))

async def cmd_alerta_dolar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # /alerta_dolar <tipo> <umbral>
    try:
        args = context.args
        if len(args) != 2:
            raise ValueError
        tipo = args[0].lower()
        if tipo not in ["blue","mep","ccl","cripto","oficial","mayorista"]:
            await update.message.reply_text("Tipos válidos: blue|mep|ccl|cripto|oficial|mayorista")
            return
        umbral = float(args[1].replace(",","."))
        chat_id = update.effective_chat.id
        lst = _alerts.get(chat_id, [])
        new_id = _new_alert_id(chat_id)
        lst.append({"id": new_id, "tipo": tipo, "umbral": umbral})
        _alerts[chat_id] = lst
        await update.message.reply_text(f"✅ Alerta creada #{new_id} — {tipo.upper()} ≥ {fmt_num(umbral)}")
    except Exception:
        await update.message.reply_text("Uso: /alerta_dolar <tipo> <umbral>")

async def cmd_alertas(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    lst = _alerts.get(chat_id, [])
    if not lst:
        await update.message.reply_text("🔕 No tenés alertas activas.")
        return
    lines = ["🔔 <b>Alertas</b>"]
    for a in lst:
        lines.append(html_code(f"#{a['id']}  {a['tipo'].upper():10s}  ≥ {fmt_num(a['umbral'])}"))
    await update.message.reply_text("\n".join(lines))

async def cmd_alerta_borrar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if len(context.args) != 1:
            raise ValueError
        delete_id = int(context.args[0])
        chat_id = update.effective_chat.id
        lst = _alerts.get(chat_id, [])
        lst = [x for x in lst if x["id"] != delete_id]
        _alerts[chat_id] = lst
        await update.message.reply_text(f"🗑️ Alerta #{delete_id} eliminada.")
    except Exception:
        await update.message.reply_text("Uso: /alerta_borrar <id>")

async def cmd_resumen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Dólares
    rates = await get_dolares()
    r_lines = ["💵 <b>Dólares</b>"]
    for k in ["blue","mep","ccl","cripto","oficial","mayorista"]:
        slot = rates.get(k)
        if not slot:
            r_lines.append(html_code(f"{k.upper():10s}  Dato no disponible"))
            continue
        compra = slot.get("compra")
        venta = slot.get("venta")
        if compra and venta:
            s = f"{k.upper():10s}  {fmt_num(compra)} / {fmt_num(venta)}"
        elif venta:
            s = f"{k.upper():10s}  — / {fmt_num(venta)}"
        else:
            s = f"{k.upper():10s}  Dato no disponible"
        r_lines.append(html_code(s))

    # Reservas
    rv = await get_reservas()
    if rv:
        reservas_s = "\n".join([
            "🏦 <b>Reservas</b>",
            html_code(f"Fecha: {rv.get('fecha','—')}"),
            html_code(f"USD {fmt_num(rv.get('valor'))}")
        ])
    else:
        reservas_s = "🏦 <b>Reservas</b>\n" + html_code("Dato no disponible")

    # Inflación
    inf = await get_inflacion()
    if inf:
        inflacion_s = "\n".join([
            "📈 <b>Inflación</b>",
            html_code(f"Período: {inf.get('fecha','—')}"),
            html_code(f"Variación: {fmt_pct(inf.get('variacion_mensual'))}")
        ])
    else:
        inflacion_s = "📈 <b>Inflación</b>\n" + html_code("Dato no disponible")

    # Riesgo
    rz = await get_riesgo_pais()
    if rz:
        val = rz.get("valor") or rz.get("riesgo") or rz.get("ultimo") or rz.get("value")
        fecha = rz.get("fecha") or rz.get("date")
        riesgo_s = "\n".join([
            "📉 <b>Riesgo País</b>",
            html_code(f"Último: {fmt_num(val)} pb" if val is not None else "Dato no disponible"),
            html_code(f"Fecha: {fecha or '—'}")
        ])
    else:
        riesgo_s = "📉 <b>Riesgo País</b>\n" + html_code("Dato no disponible")

    # Noticias
    news = await get_news()
    if news:
        n_lines = ["🗞️ <b>Noticias</b>"]
        for it in news[:5]:
            n_lines.append("• " + as_link(it["title"], it["link"]))
        news_s = "\n".join(n_lines)
    else:
        news_s = "🗞️ <b>Noticias</b>\n" + html_code("Sin novedades filtradas")

    parts = [
        "\n".join(r_lines),
        reservas_s,
        inflacion_s,
        riesgo_s,
        news_s,
    ]
    await update.message.reply_text("\n\n".join(parts), disable_web_page_preview=True)

async def fallback_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("No entendí. Probá /help o /start.")

# =========================
# APP
# =========================

def build_app() -> Application:
    defaults = Defaults(parse_mode=ParseMode.HTML)  # HTML: links clickeables
    app = ApplicationBuilder().token(BOT_TOKEN).defaults(defaults).build()

    # Handlers
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_start))

    app.add_handler(CommandHandler("dolar", cmd_dolar))
    app.add_handler(CommandHandler("reservas", cmd_reservas))
    app.add_handler(CommandHandler("inflacion", cmd_inflacion))
    app.add_handler(CommandHandler("riesgo", cmd_riesgo))
    app.add_handler(CommandHandler("acciones", cmd_acciones))
    app.add_handler(CommandHandler("cedears", cmd_cedears))
    app.add_handler(CommandHandler("ranking_acciones", cmd_ranking_acc))
    app.add_handler(CommandHandler("ranking_cedears", cmd_ranking_ced))
    app.add_handler(CommandHandler("alerta_dolar", cmd_alerta_dolar))
    app.add_handler(CommandHandler("alertas", cmd_alertas))
    app.add_handler(CommandHandler("alerta_borrar", cmd_alerta_borrar))
    app.add_handler(CommandHandler("resumen_diario", cmd_resumen))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, fallback_text))

    # Jobs de precarga (aceleran respuestas)
    jq: JobQueue = app.job_queue
    jq.run_repeating(lambda c: asyncio.create_task(get_dolares()), interval=120, first=5)
    jq.run_repeating(lambda c: asyncio.create_task(get_riesgo_pais()), interval=300, first=10)
    jq.run_repeating(lambda c: asyncio.create_task(get_news()), interval=600, first=15)
    # YF lo calentamos cada 10-15 min
    jq.run_repeating(lambda c: asyncio.create_task(build_top_n(BYMA_TICKERS, 3)), interval=900, first=20)
    jq.run_repeating(lambda c: asyncio.create_task(build_top_n(CEDEAR_TICKERS, 3)), interval=900, first=30)

    # Chequeo alertas cada 90s
    jq.run_repeating(job_check_alerts, interval=90, first=20)

    return app

def main():
    app = build_app()

    port = int(os.getenv("PORT", "10000"))  # Render expone PORT
    url_path = WEBHOOK_PATH or "tgwebhook"
    webhook_url = f"{PUBLIC_URL}/{url_path}"

    log.info(f"Levantando webhook en 0.0.0.0:{port} path=/{url_path}")
    log.info(f"Webhook URL = {webhook_url}")

    # IMPORTANTE: Sólo webhooks (NO run_polling)
    # PTB arranca un servidor Tornado interno (extra [webhooks] en requirements).
    app.run_webhook(
        listen="0.0.0.0",
        port=port,
        url_path=url_path,
        webhook_url=webhook_url,
        drop_pending_updates=False,
        stop_signals=None,  # evita cerrar event loop de Render
    )

if __name__ == "__main__":
    main()
