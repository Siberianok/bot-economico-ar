#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

import httpx
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    ApplicationBuilder, CommandHandler, ContextTypes, Defaults
)

# =========================
# CONFIG
# =========================

TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
if not TOKEN:
    raise SystemExit("Falta TELEGRAM_TOKEN en variables de entorno.")

# URL pública (Render la expone como RENDER_EXTERNAL_URL). Si no, ponla fija.
PUBLIC_URL = (
    os.getenv("RENDER_EXTERNAL_URL")
    or os.getenv("PUBLIC_URL")
    or "https://bot-economico-ar.onrender.com"
).rstrip("/")

WEBHOOK_PATH = "/tgwebhook"
WEBHOOK_URL = f"{PUBLIC_URL}{WEBHOOK_PATH}"
PORT = int(os.getenv("PORT", "10000"))
LISTEN_ADDRESS = "0.0.0.0"

# =========================
# LOGGING
# =========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("bot")

# =========================
# UTILIDADES
# =========================

AR_TZ = timezone.utc  # Render está en UTC; dejamos UTC en los sellos.

def ars(n: float) -> str:
    """Formatea $ con separador de miles (.) y coma decimal (,)."""
    try:
        entero, dec = f"{n:,.2f}".split(".")
        entero = entero.replace(",", ".")
        return f"$ {entero},{dec}"
    except Exception:
        return f"$ {n}"

def pct(n: float) -> str:
    s = "▲" if n >= 0 else "▼"
    return f"{s} {n:+.2f}%"

def now_str() -> str:
    return datetime.now(AR_TZ).strftime("%Y-%m-%d %H:%M")

# =========================
# CACHES EN MEMORIA
# =========================

state: Dict[str, dict] = {
    "dolares": {},        # {"blue": {"compra":..,"venta":..}, "mep": {...}, ...}
    "reservas": None,     # {"valor": float, "fecha": "YYYY-MM-DD"}
    "inflacion": None,    # {"valor": float, "fecha": "YYYY-MM-DD"}
    "riesgo": None,       # {"valor": int, "fecha": "YYYY-MM-DD"}
    "acciones": {},       # {ticker: (precio, var_pct)}
    "cedears": {},        # {ticker: (precio, var_pct)}
    "news": [],           # [(titulo, link)]
}

# Listas de tickers
TICKERS_BYMA = [
    "ALUA.BA","BBAR.BA","BMA.BA","BYMA.BA","CEPU.BA","COME.BA","EDN.BA","GGAL.BA",
    "LOMA.BA","MIRG.BA","PAMP.BA","SUPV.BA","TGNO4.BA","TGSU2.BA","TRAN.BA",
    "TXAR.BA","VALO.BA","YPFD.BA",
]
TICKERS_CEDEAR = [
    "AAPL.BA","AMZN.BA","GOOGL.BA","JPM.BA","KO.BA","META.BA","MSFT.BA","NVDA.BA",
    "PFE.BA","TSLA.BA","WMT.BA","XOM.BA","DIS.BA",
]

# =========================
# FETCHERS
# =========================

HEADERS_JSON = {"Accept": "application/json"}
TIMEOUT = httpx.Timeout(15, connect=10)

async def fetch_json(client: httpx.AsyncClient, url: str) -> Optional[dict]:
    try:
        r = await client.get(url, headers=HEADERS_JSON)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        log.warning("%s fallo: %s", url, e)
    except Exception as e:
        log.warning("error %s: %s", url, e)
    return None

async def fetch_text(client: httpx.AsyncClient, url: str) -> Optional[str]:
    try:
        r = await client.get(url)
        r.raise_for_status()
        return r.text
    except httpx.HTTPStatusError as e:
        log.warning("RSS fallo %s: %s", url, e.response.status_code)
    except Exception as e:
        log.warning("RSS error %s: %s", url, e)
    return None

async def refresh_dolares():
    """Blue, Oficial, Mayorista, Cripto (DolarAPI) + MEP/CCL (CriptoYa)"""
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        res = {}
        # DolarAPI
        for k in ("blue","oficial","mayorista","cripto"):
            data = await fetch_json(client, f"https://dolarapi.com/v1/dolares/{k}")
            if data and "compra" in data and "venta" in data:
                res[k.upper()] = {
                    "compra": float(data["compra"]),
                    "venta": float(data["venta"]),
                }

        # CriptoYa para MEP & CCL (devuelve llaves directas)
        dj = await fetch_json(client, "https://criptoya.com/api/dolar")
        mep_val = None
        ccl_val = None
        if isinstance(dj, dict):
            # Rápido: claves directas
            if "mep" in dj:  # puede ser float o dict con 'price'
                mep_val = dj["mep"]["price"] if isinstance(dj["mep"], dict) and "price" in dj["mep"] else dj["mep"]
            if "ccl" in dj:
                ccl_val = dj["ccl"]["price"] if isinstance(dj["ccl"], dict) and "price" in dj["ccl"] else dj["ccl"]
            # Fallback por si viene por instrumentos (al30/gd30)
            if mep_val is None:
                try:
                    mep_val = float(dj["al30"]["ci"]["price"])
                except Exception:
                    pass
            if ccl_val is None:
                try:
                    ccl_val = float(dj["gd30"]["ci"]["price"])
                except Exception:
                    pass

        if mep_val:
            res["MEP"] = {"compra": float(mep_val), "venta": float(mep_val)}
        if ccl_val:
            res["CCL"] = {"compra": float(ccl_val), "venta": float(ccl_val)}

        if res:
            state["dolares"] = res
            log.info("Dólares actualizados: %s", ",".join(res.keys()))

async def refresh_reservas_inflacion_riesgo():
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        # Reservas (LaMacro scrapeo simple)
        try:
            html = await fetch_text(client, "https://www.lamacro.ar/variables")
            # Buscamos "Reservas internacionales" y el número más reciente (en MUS$)
            valor = None
            fecha = None
            if html:
                # Muy simple, robusto ante cambios pequeños
                import re
                m = re.search(r"Reservas\s+internacionales.*?([\d\.,]+)\s*MUS\$", html, flags=re.I|re.S)
                if m:
                    valor = float(m.group(1).replace(".","").replace(",","."))
                m2 = re.search(r"Reservas\s+internacionales.*?(\d{4}-\d{2}-\d{2})", html, flags=re.I|re.S)
                if m2:
                    fecha = m2.group(1)
            if valor:
                state["reservas"] = {"valor": valor, "fecha": fecha or datetime.now().date().isoformat()}
        except Exception as e:
            log.warning("Reservas error: %s", e)

        # Inflación (ArgentinaDatos)
        infl = await fetch_json(client, "https://api.argentinadatos.com/v1/finanzas/indices/inflacion/")
        try:
            if isinstance(infl, list) and infl:
                ultimo = infl[-1]
                state["inflacion"] = {
                    "valor": float(ultimo.get("valor", 0.0)),
                    "fecha": str(ultimo.get("fecha",""))[:10],
                }
        except Exception:
            pass

        # Riesgo país (ArgentinaDatos)
        rp = await fetch_json(client, "https://api.argentinadatos.com/v1/finanzas/indices/riesgo-pais/ultimo/")
        try:
            if isinstance(rp, dict) and "valor" in rp:
                state["riesgo"] = {
                    "valor": int(rp["valor"]),
                    "fecha": str(rp.get("fecha",""))[:10],
                }
        except Exception:
            pass

# Yahoo chart v8 por ticker -> (precio, var%)
async def fetch_yahoo_ticker(client: httpx.AsyncClient, ticker: str) -> Optional[Tuple[float,float]]:
    base = "https://query1.finance.yahoo.com/v8/finance/chart"
    # 2 días para calcular variación diaria
    url = f"{base}/{ticker}?range=2d&interval=1d&lang=es-AR&region=AR"
    try:
        r = await client.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        j = r.json()
        rs = j.get("chart", {}).get("result", [])
        if not rs:
            return None
        series = rs[0]
        closes = series.get("indicators",{}).get("quote",[{}])[0].get("close",[])
        # El último puede venir None cuando está en curso; filtramos
        closes = [c for c in closes if isinstance(c,(int,float))]
        if len(closes) == 0:
            return None
        last = float(closes[-1])
        prev = float(closes[-2]) if len(closes) > 1 else last
        var = ((last - prev) / prev * 100.0) if prev else 0.0
        return (last, var)
    except httpx.HTTPStatusError as e:
        log.warning("chart %s status %s", ticker, e.response.status_code)
    except Exception as e:
        log.warning("chart %s error: %s", ticker, e)
    return None

async def refresh_quotes():
    """Prefetch de precios para /acciones y /cedears"""
    async with httpx.AsyncClient() as client:
        acc = {}
        for tk in TICKERS_BYMA:
            data = await fetch_yahoo_ticker(client, tk)
            if data: acc[tk] = data
            await asyncio.sleep(0.15)
        ced = {}
        for tk in TICKERS_CEDEAR:
            data = await fetch_yahoo_ticker(client, tk)
            if data: ced[tk] = data
            await asyncio.sleep(0.15)
        state["acciones"] = acc
        state["cedears"] = ced

async def refresh_news():
    """Noticias económicas (solo fuentes que responden 200)"""
    feeds = [
        "https://www.iprofesional.com/rss/economia",
        "https://www.clarin.com/rss/economia/",
        "https://www.cronista.com/files/rss/news.xml",
        "https://www.lanacion.com.ar/arc/outboundfeeds/rss/?outputType=xml&section=economia",
    ]
    items: List[Tuple[str,str]] = []
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        import xml.etree.ElementTree as ET
        for url in feeds:
            txt = await fetch_text(client, url)
            if not txt: continue
            try:
                root = ET.fromstring(txt)
                for it in root.iterfind(".//item")[:6]:
                    title = (it.findtext("title") or "").strip()
                    link = (it.findtext("link") or "").strip()
                    if title and link:
                        items.append((title, link))
            except Exception as e:
                log.warning("RSS parse %s: %s", url, e)
    state["news"] = items[:12]

# =========================
# RENDER HEALTHCHECK (200 OK)
# =========================

# Pequeño servidor http para /health (200) y / (200)
# No interfiere con el webhook porque usamos rutas distintas.
from aiohttp import web

async def health_handler(request):
    return web.Response(text="ok", content_type="text/plain")

async def root_handler(request):
    return web.Response(text="Observatorio Económico bot", content_type="text/plain")

# =========================
# RESPUESTAS (HTML)
# =========================

def html_box_lines(lines: List[str]) -> str:
    return "<pre>" + "\n".join(lines) + "</pre>"

def render_dolares() -> str:
    d = state.get("dolares", {})
    orden = ["BLUE","MEP","CCL","CRIPTO","OFICIAL","MAYORISTA"]
    lines = []
    # encabezado
    lines.append(f"{'Dólares':<8}")
    for k in orden:
        v = d.get(k)
        if not v: continue
        lines.append(
            f"• {k.title()}:  Compra: {ars(v['compra'])} · Venta: {ars(v['venta'])}"
        )
    if not lines or len(lines)==1:
        return "Sin datos disponibles aún."
    return "\n".join(lines)

def render_tabla(tuples: List[Tuple[str,float,float]], titulo: str) -> str:
    # tuples: [(ticker, precio, var)]
    w_tk, w_px, w_v = 9, 10, 7
    header = f"{'TICKER':<{w_tk}}{'PRECIO':>{w_px}}{'VAR%':>{w_v}}"
    sep = "-"*(w_tk+w_px+w_v)
    rows = [header, sep]
    for tk, px, vr in tuples:
        rows.append(f"{tk:<{w_tk}}{ars(px):>{w_px}}{('+' if vr>=0 else '')+f'{vr:.2f}%':>{w_v}}")
    return f"<b>{titulo}</b>\n" + html_box_lines(rows)

def render_resumen() -> str:
    d = state.get("dolares", {})
    reservas = state.get("reservas")
    infl = state.get("inflacion")
    ris = state.get("riesgo")
    news = state.get("news", [])[:6]

    # Dólares en columnas ascii
    cols = ["KIND","Compra","Venta"]
    w1,w2,w3 = 8,9,9
    lines = [f"{'Dólares':<8}", html_box_lines([
        f"{'':<{w1}}{'Compra':>{w2}}{'Venta':>{w3}}",
        f"{'-'*(w1+w2+w3)}",
        f"{'BLUE':<{w1}}{ars(d.get('BLUE',{}).get('compra',0)):>{w2}}{ars(d.get('BLUE',{}).get('venta',0)):>{w3}}",
        f"{'MEP':<{w1}}{ars(d.get('MEP',{}).get('compra',0)):>{w2}}{ars(d.get('MEP',{}).get('venta',0)):>{w3}}",
        f"{'CCL':<{w1}}{ars(d.get('CCL',{}).get('compra',0)):>{w2}}{ars(d.get('CCL',{}).get('venta',0)):>{w3}}",
        f"{'CRIPTO':<{w1}}{ars(d.get('CRIPTO',{}).get('compra',0)):>{w2}}{ars(d.get('CRIPTO',{}).get('venta',0)):>{w3}}",
        f"{'OFICIAL':<{w1}}{ars(d.get('OFICIAL',{}).get('compra',0)):>{w2}}{ars(d.get('OFICIAL',{}).get('venta',0)):>{w3}}",
        f"{'MAYORISTA':<{w1}}{ars(d.get('MAYORISTA',{}).get('compra',0)):>{w2}}{ars(d.get('MAYORISTA',{}).get('venta',0)):>{w3}}",
    ])]

    # Macros
    lines.append(f"🏦 <b>Reservas</b>: {('%.3f' % reservas['valor']).replace('.',',')} M&nbsp;USD · {reservas['fecha']}" if reservas else "🏦 <b>Reservas</b>: dato no disponible")
    lines.append(f"📈 <b>Inflación</b>: {infl['valor']:.2f}% · {infl['fecha']}" if infl else "📈 <b>Inflación</b>: dato no disponible")
    lines.append(f"🧯 <b>Riesgo País</b>: {ris['valor']} pb · {ris['fecha']}" if ris else "🧯 <b>Riesgo País</b>: dato no disponible")

    # Noticias
    if news:
        lines.append("\n📰 <b>Noticias</b>")
        for t,l in news[:6]:
            lines.append(f"• {t}")
    return f"<b>Resumen {now_str()}</b>\n\n" + "\n".join(lines)

# =========================
# HANDLERS
# =========================

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = (
        "<b>Observatorio Económico</b>\n\n"
        "Comandos:\n"
        "• /dolar – Cotizaciones (Blue, MEP, CCL, etc.)\n"
        "• /reservas – Reservas internacionales BCRA\n"
        "• /inflacion – Último dato INDEC\n"
        "• /riesgo – Riesgo país EMBI AR\n"
        "• /acciones – Precios BYMA\n"
        "• /cedears – Precios CEDEARs\n"
        "• /ranking_acciones – TOP ± del día BYMA\n"
        "• /ranking_cedears – TOP ± del día CEDEARs\n"
        "• /resumen_diario – Dólares + macros + noticias\n"
    )
    await update.message.reply_html(txt)

async def cmd_dolar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_html(render_dolares())

async def cmd_reservas(update: Update, context: ContextTypes.DEFAULT_TYPE):
    r = state.get("reservas")
    if r:
        await update.message.reply_html(
            f"Reservas Internacionales BCRA: {('%.3f' % r['valor']).replace('.',',')} M&nbsp;USD"
        )
    else:
        await update.message.reply_html("Reservas: dato no disponible")

async def cmd_inflacion(update: Update, context: ContextTypes.DEFAULT_TYPE):
    i = state.get("inflacion")
    if i:
        await update.message.reply_html(f"Inflación INDEC: {i['valor']:.2f}% · {i['fecha']}")
    else:
        await update.message.reply_html("Inflación: dato no disponible")

async def cmd_riesgo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    rp = state.get("riesgo")
    if rp:
        await update.message.reply_html(f"Riesgo País (EMBI AR): {rp['valor']} pb · {rp['fecha']}")
    else:
        await update.message.reply_html("Riesgo País: dato no disponible")

async def cmd_acciones(update: Update, context: ContextTypes.DEFAULT_TYPE):
    d = state.get("acciones", {})
    if not d:
        await update.message.reply_html("Sin datos aún. Probá de nuevo en unos segundos.")
        return
    tuples = [(k, d[k][0], d[k][1]) for k in sorted(d.keys())]
    await update.message.reply_html(render_tabla(tuples, "Acciones BYMA"))

async def cmd_cedears(update: Update, context: ContextTypes.DEFAULT_TYPE):
    d = state.get("cedears", {})
    if not d:
        await update.message.reply_html("Sin datos aún. Probá de nuevo en unos segundos.")
        return
    tuples = [(k, d[k][0], d[k][1]) for k in sorted(d.keys())]
    await update.message.reply_html(render_tabla(tuples, "CEDEARs"))

def ordenar_por_var(d: Dict[str, Tuple[float,float]], top: int = 8) -> Tuple[List[Tuple[str,float,float]], List[Tuple[str,float,float]]]:
    arr = [(k,v[0],v[1]) for k,v in d.items()]
    gan = sorted(arr, key=lambda x: x[2], reverse=True)[:top]
    per = sorted(arr, key=lambda x: x[2])[:top]
    return gan, per

async def cmd_rank_acc(update: Update, context: ContextTypes.DEFAULT_TYPE):
    d = state.get("acciones", {})
    if not d:
        await update.message.reply_html("Sin datos aún.")
        return
    gan, per = ordenar_por_var(d)
    txt = render_tabla(gan, "Acciones – Mejores del día") + "\n\n" + render_tabla(per, "Acciones – Peores del día")
    await update.message.reply_html(txt)

async def cmd_rank_ced(update: Update, context: ContextTypes.DEFAULT_TYPE):
    d = state.get("cedears", {})
    if not d:
        await update.message.reply_html("Sin datos aún.")
        return
    gan, per = ordenar_por_var(d)
    txt = render_tabla(gan, "CEDEARs – Mejores del día") + "\n\n" + render_tabla(per, "CEDEARs – Peores del día")
    await update.message.reply_html(txt)

async def cmd_resumen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_html(render_resumen())

# =========================
# JOBS
# =========================

async def job_refresh_dolares(context: ContextTypes.DEFAULT_TYPE):
    await refresh_dolares()

async def job_refresh_riesgo(context: ContextTypes.DEFAULT_TYPE):
    # sólo riesgo (las otras dos en prefetch para espaciar)
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        rp = await fetch_json(client, "https://api.argentinadatos.com/v1/finanzas/indices/riesgo-pais/ultimo/")
        if isinstance(rp, dict) and "valor" in rp:
            state["riesgo"] = {"valor": int(rp["valor"]), "fecha": str(rp.get("fecha",""))[:10]}

async def job_prefetch(context: ContextTypes.DEFAULT_TYPE):
    await asyncio.gather(
        refresh_reservas_inflacion_riesgo(),
        refresh_quotes(),
    )

async def job_news(context: ContextTypes.DEFAULT_TYPE):
    await refresh_news()

async def job_keepalive(context: ContextTypes.DEFAULT_TYPE):
    # Hace una request saliente a /health para que Render vea tráfico.
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            await client.get(f"{PUBLIC_URL}/health")
    except Exception:
        pass

# =========================
# MAIN
# =========================

async def main():
    defaults = Defaults(parse_mode=ParseMode.HTML, disable_web_page_preview=True)

    app = ApplicationBuilder().token(TOKEN).defaults(defaults).build()

    # Handlers
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_start))
    app.add_handler(CommandHandler("dolar", cmd_dolar))
    app.add_handler(CommandHandler("reservas", cmd_reservas))
    app.add_handler(CommandHandler("inflacion", cmd_inflacion))
    app.add_handler(CommandHandler("riesgo", cmd_riesgo))
    app.add_handler(CommandHandler("acciones", cmd_acciones))
    app.add_handler(CommandHandler("cedears", cmd_cedears))
    app.add_handler(CommandHandler("ranking_acciones", cmd_rank_acc))
    app.add_handler(CommandHandler("ranking_cedears", cmd_rank_ced))
    app.add_handler(CommandHandler("resumen_diario", cmd_resumen))

    # Jobs
    jq = app.job_queue
    jq.run_repeating(job_refresh_dolares, interval=120, first=3)
    jq.run_repeating(job_prefetch, interval=15*60, first=5)
    jq.run_repeating(job_refresh_riesgo, interval=5*60, first=7)
    jq.run_repeating(job_news, interval=10*60, first=9)
    jq.run_repeating(job_keepalive, interval=14*60, first=11)

    # Servidor aiohttp para /health y /
    runner = web.AppRunner(web.Application())
    await runner.setup()
    site = web.TCPSite(runner, LISTEN_ADDRESS, PORT)
    app_http = runner.app
    app_http.router.add_get("/health", health_handler)
    app_http.router.add_get("/", root_handler)

    # IMPORTANTE: arrancamos ambos servidores; PTB usa su propio servidor para webhook.
    # 1) sitio auxiliar
    await site.start()
    log.info("HTTP auxiliar en %s:%d (/health listo)", LISTEN_ADDRESS, PORT)

    # 2) webhook PTB (mismo puerto pero ruta distinta no es posible en el mismo socket),
    #    por eso PTB abre su servidor interno *en el mismo puerto*. Para evitar conflicto,
    #    usamos el servidor interno de PTB y dejamos el auxiliar arriba con /health.
    #    Render detecta el puerto en uso y está OK.
    await app.bot.delete_webhook(drop_pending_updates=True)
    log.info("Levantando webhook en %s:%d path=%s", LISTEN_ADDRESS, PORT, WEBHOOK_PATH)
    log.info("Webhook URL = %s", WEBHOOK_URL)

    await app.run_webhook(
        listen=LISTEN_ADDRESS,
        port=PORT,
        url_path=WEBHOOK_PATH.lstrip("/"),
        webhook_url=WEBHOOK_URL,
        drop_pending_updates=True,
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
