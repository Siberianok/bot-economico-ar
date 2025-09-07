#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ================================================================
# Bot Económico AR — Webhook only (PTB v20)
# - Noticias nacionales (títulos con links HTML)
# - Dólar: Blue/Oficial/Mayorista/Cripto (DolarAPI) + MEP/CCL (CriptoYa)
# - Riesgo País e Inflación (ArgentinaDatos, con follow_redirects)
# - Reservas BCRA (parseo de LaMacro)
# - Acciones / CEDEARs / Ranking (Yahoo Finance quote v7/v7-alt con headers y fallback)
# - Cache + JobQueue (prefetch) para respuestas rápidas
# - Keep-alive interno (golpea tu URL; 404 es OK en Free)
# ================================================================

import os
import sys
import re
import html
import time
import json
import logging
from datetime import datetime, timedelta, timezone
import asyncio
from typing import Dict, Any, List, Tuple, Optional

import httpx

from telegram import Update, LinkPreviewOptions
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    Defaults,
)

# ------------------------- CONFIG -------------------------------

BOT_TOKEN = os.environ.get("BOT_TOKEN", "").strip()
if not BOT_TOKEN:
    print("ERROR: Falta BOT_TOKEN en variables de entorno.", file=sys.stderr)
    sys.exit(1)

BASE_URL = os.environ.get("RENDER_EXTERNAL_URL") or os.environ.get("BASE_URL") or "https://bot-economico-ar.onrender.com"
WEBHOOK_PATH = os.environ.get("WEBHOOK_PATH", "/tgwebhook")
PORT = int(os.environ.get("PORT", "10000"))
LISTEN = os.environ.get("LISTEN", "0.0.0.0")

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
COMMON_HEADERS = {
    "User-Agent": UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "es-AR,es;q=0.9,en-US;q=0.8,en;q=0.7",
    "Connection": "keep-alive",
    "Referer": "https://finance.yahoo.com/",
}

HTTP_TIMEOUT = 8.0
HTTP = httpx.Client(timeout=HTTP_TIMEOUT, headers=COMMON_HEADERS, follow_redirects=True)
AHTTP = httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers=COMMON_HEADERS, follow_redirects=True)

# ------------------------- LOGGING ------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("bot")

# ------------------------- CACHE --------------------------------

_cache: Dict[str, Tuple[datetime, Any]] = {}

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def cache_get(key: str):
    item = _cache.get(key)
    if not item:
        return None
    exp, val = item
    if now_utc() >= exp:
        _cache.pop(key, None)
        return None
    return val

def cache_set(key: str, val: Any, ttl_sec: int):
    _cache[key] = (now_utc() + timedelta(seconds=ttl_sec), val)

def html_bold(s: str) -> str:
    return f"<b>{html.escape(s)}</b>"

def html_code(s: str) -> str:
    return f"<code>{html.escape(s)}</code>"

def pct(x: float) -> str:
    sign = "+" if x > 0 else ""
    return f"{sign}{x:.2f}%"

# --------------------- DÓLARES ----------------------------------

def get_dolares() -> Dict[str, Any]:
    ck = "dolares_v1"
    c = cache_get(ck)
    if c:
        return c

    data = {}
    base = "https://dolarapi.com/v1/dolares/"
    for tipo in ["blue", "oficial", "mayorista", "cripto"]:
        try:
            r = HTTP.get(base + tipo)
            if r.status_code == 200:
                j = r.json()
                data[tipo] = {
                    "compra": j.get("compra"),
                    "venta": j.get("venta"),
                    "fecha": j.get("fechaActualizacion") or j.get("fecha_actualizacion"),
                }
            else:
                log.warning(f"DolarAPI fallo {tipo}: {r.status_code}")
        except Exception as e:
            log.warning(f"DolarAPI error {tipo}: {e}")

    try:
        r = HTTP.get("https://criptoya.com/api/dolar")
        if r.status_code == 200:
            j = r.json()
            if "mep" in j:
                data["mep"] = {"venta": j["mep"], "fecha": None}
            if "ccl" in j:
                data["ccl"] = {"venta": j["ccl"], "fecha": None}
        else:
            log.warning(f"CriptoYa fallo: {r.status_code}")
    except Exception as e:
        log.warning(f"CriptoYa error: {e}")

    cache_set(ck, data, 120)
    return data

# --------------------- RIESGO / INFLACION -----------------------

def get_riesgo_pais() -> Optional[Dict[str, Any]]:
    ck = "riesgo_v1"
    c = cache_get(ck)
    if c:
        return c
    # el 301 se sigue por follow_redirects=True
    url = "https://api.argentinadatos.com/v1/finanzas/indices/riesgo-pais/ultimo"
    try:
        r = HTTP.get(url)
        if r.status_code == 200:
            j = r.json()
            cache_set(ck, j, 300)
            return j
        log.warning(f"ArgentinaDatos riesgo fallo: {r.status_code}")
    except Exception as e:
        log.warning(f"ArgentinaDatos riesgo error: {e}")
    return None

def get_inflacion() -> Optional[Dict[str, Any]]:
    ck = "inflacion_v1"
    c = cache_get(ck)
    if c:
        return c
    url = "https://api.argentinadatos.com/v1/finanzas/indices/inflacion"
    try:
        r = HTTP.get(url)
        if r.status_code == 200:
            arr = r.json()
            if isinstance(arr, list) and arr:
                ultimo = arr[-1]
                cache_set(ck, ultimo, 6 * 3600)
                return ultimo
        log.warning(f"ArgentinaDatos inflacion fallo: {r.status_code}")
    except Exception as e:
        log.warning(f"ArgentinaDatos inflacion error: {e}")
    return None

# --------------------- RESERVAS BCRA -----------------------------

def get_reservas() -> Optional[Dict[str, Any]]:
    ck = "reservas_v1"
    c = cache_get(ck)
    if c:
        return c
    url = "https://www.lamacro.ar/variables"
    try:
        r = HTTP.get(url)
        if r.status_code == 200:
            htmltxt = r.text
            m = re.search(r"Reservas Internacionales del BCRA.*?(\d{1,3}(?:\.\d{3})+)", htmltxt, re.I | re.S)
            f = re.search(r"Reservas Internacionales.*?Últ\. act:\s*([0-9]{2}/[0-9]{2}/[0-9]{4})", htmltxt, re.I | re.S)
            if m:
                valor_musd = float(m.group(1).replace(".", ""))
                data = {"valor_musd": valor_musd, "fecha": f.group(1) if f else None}
                cache_set(ck, data, 6 * 3600)
                return data
            else:
                log.warning("No se pudo parsear Reservas en LaMacro")
        else:
            log.warning(f"LaMacro status {r.status_code}")
    except Exception as e:
        log.warning(f"Reservas error: {e}")
    return None

# --------------------- NOTICIAS (RSS) ----------------------------

NEWS_SOURCES = [
    "https://www.iprofesional.com/rss/economia",
    "https://www.cronista.com/rss/economia/",
    "https://www.clarin.com/rss/economia/",
    "https://www.pagina12.com.ar/rss/sections/economia",  # su feed cambia; este suele redirigir
    "https://www.lanacion.com.ar/arc/outboundfeeds/rss/?outputType=xml&section=economia",
]

def parse_rss_items(xml_text: str) -> List[Tuple[str, str]]:
    items = re.findall(r"<item>.*?<title>(.*?)</title>.*?<link>(.*?)</link>.*?</item>", xml_text, re.S | re.I)
    out: List[Tuple[str, str]] = []
    for t, l in items:
        t = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", t)
        l = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", l)
        t = html.unescape(re.sub("<.*?>", "", t)).strip()
        l = html.unescape(re.sub("<.*?>", "", l)).strip()
        if t and l:
            out.append((t, l))
    return out

def get_noticias(n=5) -> List[Tuple[str, str]]:
    ck = "news_v2"
    c = cache_get(ck)
    if c:
        return c[:n]

    seen = set()
    acc: List[Tuple[str, str]] = []
    for url in NEWS_SOURCES:
        try:
            r = HTTP.get(url)
            if r.status_code == 200:
                items = parse_rss_items(r.text)
                for t, l in items:
                    if t not in seen:
                        seen.add(t)
                        acc.append((t, l))
                        if len(acc) >= 30:
                            break
            else:
                log.warning(f"RSS fallo {url}: {r.status_code}")
        except Exception as e:
            log.warning(f"RSS error {url}: {e}")

    acc = acc[:max(n, 5)]
    cache_set(ck, acc, 30 * 60)
    return acc[:n]

# --------------------- QUOTES (ACCIONES / CEDEARs) ---------------

ACCIONES_BA = [
    "GGAL.BA","BMA.BA","YPFD.BA","PAMP.BA","CEPU.BA","TGSU2.BA","TGNO4.BA",
    "ALUA.BA","TXAR.BA","LOMA.BA","BYMA.BA","BBAR.BA","VALO.BA","MIRG.BA",
    "SUPV.BA","COME.BA","EDN.BA","TRAN.BA",
]

CEDEARS_BA = [
    "AAPL.BA","MSFT.BA","NVDA.BA","TSLA.BA","AMZN.BA","META.BA","GOOGL.BA",
    "KO.BA","JPM.BA","WMT.BA","DIS.BA","XOM.BA","PFE.BA",
]

def yahoo_quote_batch(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Intenta v7/finance/quote (query1 y query2) con headers 'de navegador'.
    Devuelve symbol -> {price, change, changePercent, time}
    """
    out: Dict[str, Dict[str, Any]] = {}
    if not symbols:
        return out

    def _fetch(base_host: str, group: List[str]) -> int:
        url = f"https://{base_host}/v7/finance/quote"
        r = HTTP.get(url, params={"symbols": ",".join(group), "lang": "es-AR", "region": "AR"})
        if r.status_code == 200:
            j = r.json()
            quotes = j.get("quoteResponse", {}).get("result", [])
            for q in quotes:
                sym = q.get("symbol")
                if not sym:
                    continue
                out[sym] = {
                    "price": q.get("regularMarketPrice"),
                    "change": q.get("regularMarketChange"),
                    "changePercent": q.get("regularMarketChangePercent"),
                    "time": q.get("regularMarketTime"),
                }
        return r.status_code

    chunk = 40
    for i in range(0, len(symbols), chunk):
        group = symbols[i:i+chunk]
        status = _fetch("query1.finance.yahoo.com", group)
        if status == 401 or status == 403:
            # intentar host alternativo
            status2 = _fetch("query2.finance.yahoo.com", group)
            if status2 != 200:
                log.warning(f"Yahoo quote status {status} / alt {status2} para {','.join(group)}")
        elif status != 200:
            log.warning(f"Yahoo quote status {status} para {','.join(group)}")
        time.sleep(0.35)  # pausa corta para evitar rate-limit
    return out

def get_quotes_acciones() -> Dict[str, Dict[str, Any]]:
    ck = "quotes_acc_v3"
    c = cache_get(ck)
    if c:
        return c
    data = yahoo_quote_batch(ACCIONES_BA)
    cache_set(ck, data, 15 * 60)
    return data

def get_quotes_cedears() -> Dict[str, Dict[str, Any]]:
    ck = "quotes_ced_v3"
    c = cache_get(ck)
    if c:
        return c
    data = yahoo_quote_batch(CEDEARS_BA)
    cache_set(ck, data, 15 * 60)
    return data

def ranking_from_quotes(quotes: Dict[str, Dict[str, Any]], topn=10) -> Tuple[List[Tuple[str,float]], List[Tuple[str,float]]]:
    arr = []
    for sym, q in quotes.items():
        chp = q.get("changePercent")
        if isinstance(chp, (int, float)):
            arr.append((sym, float(chp)))
    arr.sort(key=lambda x: x[1], reverse=True)
    return arr[:topn], arr[-topn:][::-1]

# --------------------- COMMANDS ---------------------------------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = (
        html_bold("Observatorio Económico 🇦🇷") + "\n"
        "Comandos:\n"
        "• /resumen – panorama rápido (datos + 5 titulares)\n"
        "• /dolar – cotizaciones (blue, mep, ccl, etc.)\n"
        "• /reservas – reservas BCRA (MUSD)\n"
        "• /inflacion – último dato INDEC (mensual)\n"
        "• /riesgo – riesgo país (EMBI AR)\n"
        "• /acciones – precios/variaciones BYMA\n"
        "• /cedears – precios/variaciones CEDEARs\n"
        "• /ranking – top/bottom del día\n"
    )
    await update.message.reply_text(txt, disable_web_page_preview=True)

async def cmd_resumen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    d = get_dolares()
    mep = d.get("mep", {}).get("venta")
    ccl = d.get("ccl", {}).get("venta")
    blue = d.get("blue", {}).get("venta")
    ofi = d.get("oficial", {}).get("venta")
    may = d.get("mayorista", {}).get("venta")

    r = get_riesgo_pais()
    riesgo_txt = f"{int(r['valor'])} pb" if r and r.get("valor") is not None else "N/D"

    inf = get_inflacion()
    infl_txt = f"{float(inf['valor']):.2f}% ({inf.get('fecha')})" if inf and inf.get("valor") is not None else "N/D"

    res = get_reservas()
    res_txt = f"{int(res['valor_musd']):,} MUS$".replace(",", ".") if res and res.get("valor_musd") else "N/D"

    linea = (
        f"{html_bold('USD')} blue: {blue or 'N/D'} | mep: {mep or 'N/D'} | ccl: {ccl or 'N/D'} | oficial: {ofi or 'N/D'} | mayorista: {may or 'N/D'}\n"
        f"{html_bold('Riesgo País')}: {riesgo_txt}  ·  {html_bold('Inflación')}: {infl_txt}  ·  {html_bold('Reservas')}: {res_txt}\n"
    )

    items = get_noticias(5)
    if items:
        news_lines = [f"• <a href=\"{html.escape(u)}\">{html.escape(t)}</a>" for (t, u) in items]
        news_txt = "\n".join(news_lines)
    else:
        news_txt = "Sin noticias por el momento."

    await update.message.reply_text(linea + "\n" + html_bold("Últimos titulares:") + "\n" + news_txt, disable_web_page_preview=True)

async def cmd_dolar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    d = get_dolares()
    def v(name, k="venta"):
        x = d.get(name, {}).get(k)
        return f"{x}" if x is not None else "N/D"

    txt = (
        html_bold("Dólares") + "\n"
        f"• Blue: {v('blue')}\n"
        f"• MEP: {v('mep')}\n"
        f"• CCL: {v('ccl')}\n"
        f"• Oficial: {v('oficial')}\n"
        f"• Mayorista: {v('mayorista')}\n"
        f"• Cripto: {v('cripto')}\n"
    )
    await update.message.reply_text(txt, disable_web_page_preview=True)

async def cmd_riesgo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    r = get_riesgo_pais()
    if not r:
        await update.message.reply_text("Riesgo País: dato no disponible.")
        return
    txt = f"{html_bold('Riesgo País (EMBI AR)')}: {int(r['valor'])} pb · {r.get('fecha','')}"
    await update.message.reply_text(txt, disable_web_page_preview=True)

async def cmd_inflacion(update: Update, context: ContextTypes.DEFAULT_TYPE):
    inf = get_inflacion()
    if not inf:
        await update.message.reply_text("Inflación: dato no disponible.")
        return
    txt = f"{html_bold('Inflación mensual INDEC')}: {float(inf['valor']):.2f}% · {inf.get('fecha','')}"
    await update.message.reply_text(txt, disable_web_page_preview=True)

async def cmd_reservas(update: Update, context: ContextTypes.DEFAULT_TYPE):
    res = get_reservas()
    if not res:
        await update.message.reply_text("Reservas BCRA: dato no disponible.")
        return
    txt = f"{html_bold('Reservas Internacionales BCRA')}: {int(res['valor_musd']):,} MUS$".replace(",", ".")
    if res.get("fecha"):
        txt += f" · {res['fecha']}"
    await update.message.reply_text(txt, disable_web_page_preview=True)

def fmt_quotes_table(quotes: Dict[str, Dict[str, Any]]) -> str:
    if not quotes:
        return "Sin datos."
    lines = [html_code(f"{'TICKER':<10} {'PRECIO':>10} {'VAR%':>8}")]
    for sym in sorted(quotes.keys()):
        q = quotes[sym]
        price = q.get("price")
        chp = q.get("changePercent")
        price_s = f"{price:.2f}" if isinstance(price, (int, float)) else "-"
        chp_s = f"{chp:+.2f}%" if isinstance(chp, (int, float)) else "-"
        lines.append(html_code(f"{sym:<10} {price_s:>10} {chp_s:>8}"))
    return "\n".join(lines)

async def cmd_acciones(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = get_quotes_acciones()
    txt = html_bold("Acciones BYMA") + "\n" + fmt_quotes_table(q)
    await update.message.reply_text(txt, disable_web_page_preview=True)

async def cmd_cedears(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = get_quotes_cedears()
    txt = html_bold("CEDEARs") + "\n" + fmt_quotes_table(q)
    await update.message.reply_text(txt, disable_web_page_preview=True)

async def cmd_ranking(update: Update, context: ContextTypes.DEFAULT_TYPE):
    allq = {}
    allq.update(get_quotes_acciones())
    allq.update(get_quotes_cedears())
    top, bot = ranking_from_quotes(allq, topn=10)
    def block(title, arr):
        lines = [html_bold(title)]
        for sym, pch in arr:
            lines.append(f"• {html_code(sym)}  {pct(pch)}")
        return "\n".join(lines)
    txt = block("Top 10", top) + "\n\n" + block("Bottom 10", bot)
    await update.message.reply_text(txt, disable_web_page_preview=True)

# --------------------- JOBS (PREFETCH + KEEPALIVE) ---------------

async def job_prefetch(context: ContextTypes.DEFAULT_TYPE):
    try:
        get_dolares()
        get_riesgo_pais()
        get_inflacion()
        get_reservas()
        get_quotes_acciones()
        get_quotes_cedears()
    except Exception as e:
        log.warning(f"Prefetch job error: {e}")

async def job_news(context: ContextTypes.DEFAULT_TYPE):
    try:
        get_noticias(5)
    except Exception as e:
        log.warning(f"News job error: {e}")

async def job_keepalive(context: ContextTypes.DEFAULT_TYPE):
    # Pegarle a tu servicio mantiene tráfico entrante; que devuelva 404 está bien.
    url = BASE_URL.rstrip("/") + "/ping"
    try:
        await AHTTP.get(url)
    except Exception:
        pass

# Estos dos reemplazan los lambda (evita TypeError 'await dict'):
async def job_refresh_dolares(context: ContextTypes.DEFAULT_TYPE):
    try:
        get_dolares()
    except Exception:
        pass

async def job_refresh_riesgo(context: ContextTypes.DEFAULT_TYPE):
    try:
        get_riesgo_pais()
    except Exception:
        pass

# --------------------- APP / WEBHOOK -----------------------------

def build_app() -> Application:
    defaults = Defaults(parse_mode=ParseMode.HTML, link_preview_options=LinkPreviewOptions(is_disabled=True))
    app = ApplicationBuilder().token(BOT_TOKEN).defaults(defaults).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_start))
    app.add_handler(CommandHandler("resumen", cmd_resumen))
    app.add_handler(CommandHandler("dolar", cmd_dolar))
    app.add_handler(CommandHandler("reservas", cmd_reservas))
    app.add_handler(CommandHandler("inflacion", cmd_inflacion))
    app.add_handler(CommandHandler("riesgo", cmd_riesgo))
    app.add_handler(CommandHandler("acciones", cmd_acciones))
    app.add_handler(CommandHandler("cedears", cmd_cedears))
    app.add_handler(CommandHandler("ranking", cmd_ranking))

    jq = app.job_queue
    jq.run_repeating(job_prefetch, interval=15 * 60, first=5)
    jq.run_repeating(job_news, interval=10 * 60, first=10)
    jq.run_repeating(job_refresh_dolares, interval=2 * 60, first=3)  # antes lambda -> ahora async
    jq.run_repeating(job_refresh_riesgo, interval=5 * 60, first=7)   # idem
    jq.run_repeating(job_keepalive, interval=14 * 60, first=20)

    return app

def main():
    app = build_app()
    webhook_url = BASE_URL.rstrip("/") + WEBHOOK_PATH
    log.info(f"Levantando webhook en {LISTEN}:{PORT} path={WEBHOOK_PATH}")
    log.info(f"Webhook URL = {webhook_url}")
    app.run_webhook(
        listen=LISTEN,
        port=PORT,
        url_path=WEBHOOK_PATH.lstrip("/"),
        webhook_url=webhook_url,
        drop_pending_updates=True,
    )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.exception(e)
        raise
