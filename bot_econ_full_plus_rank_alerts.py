import os
import asyncio
import logging
import html
from datetime import datetime, timezone
from statistics import median

import httpx
from telegram import Update, LinkPreviewOptions
from telegram.constants import ParseMode
from telegram.ext import (
    Application, CommandHandler, ContextTypes, Defaults
)

# ==================
# Config & Logging
# ==================
logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    raise SystemExit("Falta TELEGRAM_TOKEN en variables de entorno.")

PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
WEBHOOK_PATH = os.getenv("WEBHOOK_PATH", "/tgwebhook")
PORT = int(os.getenv("PORT", "10000"))

# ==========
# Cache
# ==========
CACHE = {
    "dolares": {"ts": 0, "data": {}},
    "riesgo": {"ts": 0, "data": None},
    "inflacion": {"ts": 0, "data": None},
    "reservas": {"ts": 0, "data": None},
    "byma": {"ts": 0, "data": {}},
    "cedears": {"ts": 0, "data": {}},
}

CACHE_TTL = {
    "dolares": 60,         # 1 min
    "riesgo": 300,         # 5 min
    "inflacion": 3600,     # 1 h
    "reservas": 3600,      # 1 h
    "byma": 600,           # 10 min
    "cedears": 600,        # 10 min
}

# =====================
# Helpers de HTTP
# =====================
async def get_json(client: httpx.AsyncClient, url: str, **kw):
    r = await client.get(url, timeout=kw.get("timeout", 10))
    r.raise_for_status()
    return r.json()

# =====================
# Dólares
# =====================
async def fetch_dolares() -> dict:
    """Obtiene cotizaciones Blue/Oficial/Mayorista/Cripto de DolarAPI
    y MEP/CCL a partir de CriptoYa, devolviendo un diccionario simple.
    """
    now = int(datetime.now(timezone.utc).timestamp())
    if now - CACHE["dolares"]["ts"] < CACHE_TTL["dolares"] and CACHE["dolares"]["data"]:
        return CACHE["dolares"]["data"]

    data = {}
    async with httpx.AsyncClient() as client:
        # DolarAPI (compra/venta)
        for tipo in ("blue", "oficial", "mayorista", "cripto"):
            try:
                j = await get_json(client, f"https://dolarapi.com/v1/dolares/{tipo}")
                data[tipo] = {
                    "compra": j.get("compra"),
                    "venta": j.get("venta"),
                    "fuente": "dolarapi"
                }
            except Exception as e:
                logger.warning("DolarAPI fallo %s: %s", tipo, e)

        # CriptoYa para MEP/CCL (varias fuentes -> agregamos mediana)
        try:
            j = await get_json(client, "https://criptoya.com/api/dolar")
            # MEP: tomamos fuentes bono AL30/GD30 (campo 'ci' = contado inmediato)
            mep_vals = []
            for bono in ("al30", "gd30"):
                v = j.get("mep", {}).get(bono, {}).get("ci", {}).get("price")
                if isinstance(v, (int, float)):
                    mep_vals.append(v)
            data["mep"] = {"precio": round(median(mep_vals), 2) if mep_vals else None, "fuente": "criptoya"}

            # CCL: también AL30/GD30/letras si hay
            ccl_vals = []
            for bono in ("al30", "gd30", "letras"):
                v = j.get("ccl", {}).get(bono, {}).get("ci", {}).get("price")
                if isinstance(v, (int, float)):
                    ccl_vals.append(v)
            data["ccl"] = {"precio": round(median(ccl_vals), 2) if ccl_vals else None, "fuente": "criptoya"}
        except Exception as e:
            logger.warning("CriptoYa fallo: %s", e)

    CACHE["dolares"] = {"ts": now, "data": data}
    return data


def fmt_money(v):
    if v is None:
        return "-"
    return f"$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def render_dolares_panel(data: dict) -> str:
    # tabla monoespaciada
    rows = []
    rows.append("Dólares")
    rows.append("""
TIPO        │ Compra      │ Venta       
────────────┼─────────────┼─────────────
""".strip("\n"))

    def line(name, compra=None, venta=None, unico=None):
        if unico is not None:
            compra = venta = unico
        rows.append(f"{name:<12}│ {fmt_money(compra):<12} │ {fmt_money(venta):<12}")

    b = data.get("blue", {})
    o = data.get("oficial", {})
    m = data.get("mayorista", {})
    c = data.get("cripto", {})

    line("BLUE", b.get("compra"), b.get("venta"))
    line("MEP", unico=data.get("mep", {}).get("precio"))
    line("CCL", unico=data.get("ccl", {}).get("precio"))
    line("CRIPTO", c.get("compra"), c.get("venta"))
    line("OFICIAL", o.get("compra"), o.get("venta"))
    line("MAYORISTA", m.get("compra"), m.get("venta"))

    body = "\n".join(rows)
    return f"<pre>{html.escape(body)}</pre>"

# =====================
# Macros (riesgo, inflación, reservas)
# =====================
async def fetch_riesgo() -> str:
    now = int(datetime.now(timezone.utc).timestamp())
    if now - CACHE["riesgo"]["ts"] < CACHE_TTL["riesgo"] and CACHE["riesgo"]["data"]:
        return CACHE["riesgo"]["data"]
    val = None
    date = None
    async with httpx.AsyncClient() as client:
        try:
            j = await get_json(client, "https://api.argentinadatos.com/v1/finanzas/indices/riesgo-pais/ultimo/")
            val = j.get("valor")
            date = j.get("fecha")
        except Exception as e:
            logger.warning("ArgentinaDatos riesgo fallo: %s", e)
    s = f"Riesgo País (EMBI AR): {val if val is not None else '-'} pb · {date or '-'}"
    CACHE["riesgo"] = {"ts": now, "data": s}
    return s


async def fetch_inflacion() -> str:
    now = int(datetime.now(timezone.utc).timestamp())
    if now - CACHE["inflacion"]["ts"] < CACHE_TTL["inflacion"] and CACHE["inflacion"]["data"]:
        return CACHE["inflacion"]["data"]
    val = None
    date = None
    async with httpx.AsyncClient() as client:
        try:
            j = await get_json(client, "https://api.argentinadatos.com/v1/finanzas/indices/inflacion/")
            # último registro
            if isinstance(j, list) and j:
                ultimo = j[-1]
                val = ultimo.get("valor")
                date = ultimo.get("fecha")
        except Exception as e:
            logger.warning("ArgentinaDatos inflacion fallo: %s", e)
    s = f"Inflación INDEC: {val if val is not None else '-'}% · {date or '-'}"
    CACHE["inflacion"] = {"ts": now, "data": s}
    return s


async def fetch_reservas() -> str:
    now = int(datetime.now(timezone.utc).timestamp())
    if now - CACHE["reservas"]["ts"] < CACHE_TTL["reservas"] and CACHE["reservas"]["data"]:
        return CACHE["reservas"]["data"]
    val = None
    async with httpx.AsyncClient() as client:
        try:
            # página simple con variables, parseo mínimo
            r = await client.get("https://www.lamacro.ar/variables", timeout=10)
            r.raise_for_status()
            # buscar 'Reservas internacionales' en el HTML
            text = r.text
            import re
            m = re.search(r"Reservas internacionales[^\d]+([\d.,]+)\s*MUS\$", text, re.IGNORECASE)
            if m:
                val = m.group(1).replace(".", "").replace(",", ".")
                try:
                    val = float(val)
                except:
                    pass
        except Exception as e:
            logger.warning("lamacro reservas fallo: %s", e)
    s = f"Reservas Internacionales BCRA: {val if val is not None else '-'} MUS$"
    CACHE["reservas"] = {"ts": now, "data": s}
    return s

# =====================
# Yahoo helper (precio + variación diaria)
# =====================
async def yahoo_chart_last_change(client: httpx.AsyncClient, symbol: str):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=5d&interval=1d&lang=es-AR&region=AR"
    try:
        r = await client.get(url, timeout=10)
        r.raise_for_status()
        j = r.json()
        res = j.get("chart", {}).get("result")
        if not res:
            return None
        q = res[0].get("indicators", {}).get("quote", [{}])[0]
        closes = q.get("close", [])
        # tomar últimos dos valores no nulos
        last_vals = [v for v in closes if isinstance(v, (int, float))][-2:]
        if len(last_vals) < 1:
            return None
        price = last_vals[-1]
        var = None
        if len(last_vals) == 2 and last_vals[0] > 0:
            var = (last_vals[1] - last_vals[0]) / last_vals[0] * 100
        return {"price": price, "var": var}
    except Exception as e:
        logger.warning("yahoo chart %s fallo: %s", symbol, e)
        return None


def fmt_var(pct):
    if pct is None:
        return "-"
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.2f}%"


def fmt_price_ar(v):
    if v is None:
        return "-"
    return f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def render_table(title: str, rows: list[tuple[str, float, float]]):
    head = f"{title}\n\nTICKER      PRECIO      VAR%\n──────────  ──────────  ─────"
    body_lines = [f"{t:<10} {fmt_price_ar(p):>10}  {fmt_var(v):>6}" for t, p, v in rows]
    return f"<pre>{html.escape(head + '\n' + '\n'.join(body_lines))}</pre>"


BYMA_TICKERS = [
    "ALUA.BA","BBAR.BA","BMA.BA","BYMA.BA","CEPU.BA","COME.BA","EDN.BA","GGAL.BA",
    "LOMA.BA","MIRG.BA","PAMP.BA","SUPV.BA","TGNO4.BA","TGSU2.BA","TRAN.BA","TXAR.BA","VALO.BA","YPFD.BA"
]

CEDEAR_TICKERS = [
    "AAPL.BA","AMZN.BA","GOOGL.BA","JPM.BA","KO.BA","META.BA","MSFT.BA","NVDA.BA",
    "PFE.BA","TSLA.BA","WMT.BA","XOM.BA"
]


async def prefetch_quotes():
    now = int(datetime.now(timezone.utc).timestamp())
    async with httpx.AsyncClient() as client:
        tasks = [yahoo_chart_last_change(client, s) for s in BYMA_TICKERS]
        res = await asyncio.gather(*tasks)
        byma = {}
        for s, v in zip(BYMA_TICKERS, res):
            if v:
                byma[s] = v
        CACHE["byma"] = {"ts": now, "data": byma}

        tasks = [yahoo_chart_last_change(client, s) for s in CEDEAR_TICKERS]
        res = await asyncio.gather(*tasks)
        ced = {}
        for s, v in zip(CEDEAR_TICKERS, res):
            if v:
                ced[s] = v
        CACHE["cedears"] = {"ts": now, "data": ced}


async def get_cached_quotes(kind: str) -> dict:
    now = int(datetime.now(timezone.utc).timestamp())
    if now - CACHE[kind]["ts"] > CACHE_TTL[kind] or not CACHE[kind]["data"]:
        await prefetch_quotes()
    return CACHE[kind]["data"]

# =====================
# Handlers
# =====================
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = (
        "Bienvenido! Comandos disponibles:\n"
        "/dolar – cotizaciones principales\n"
        "/reservas, /inflacion, /riesgo\n"
        "/acciones – panel BYMA\n"
        "/cedears – panel CEDEARs\n"
        "/ranking_acciones, /ranking_cedears\n"
        "/resumen_diario – mix rápido"
    )
    await update.message.reply_text(html.escape(txt))


async def cmd_dolar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    data = await fetch_dolares()
    await update.message.reply_text(render_dolares_panel(data), parse_mode=ParseMode.HTML)


async def cmd_riesgo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(html.escape(await fetch_riesgo()))


async def cmd_inflacion(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(html.escape(await fetch_inflacion()))


async def cmd_reservas(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(html.escape(await fetch_reservas()))


async def cmd_acciones(update: Update, context: ContextTypes.DEFAULT_TYPE):
    m = await update.message.reply_text("Cargando BYMA…")
    q = await get_cached_quotes("byma")
    rows = [(s, q[s]["price"], q[s]["var"]) for s in BYMA_TICKERS if s in q]
    await m.edit_text(render_table("Acciones BYMA", rows), parse_mode=ParseMode.HTML)


async def cmd_cedears(update: Update, context: ContextTypes.DEFAULT_TYPE):
    m = await update.message.reply_text("Cargando CEDEARs…")
    q = await get_cached_quotes("cedears")
    rows = [(s, q[s]["price"], q[s]["var"]) for s in CEDEAR_TICKERS if s in q]
    await m.edit_text(render_table("CEDEARs", rows), parse_mode=ParseMode.HTML)


async def cmd_rank_byma(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = await get_cached_quotes("byma")
    rows = sorted([(s, v["price"], v["var"]) for s, v in q.items() if v.get("var") is not None], key=lambda x: x[2], reverse=True)[:10]
    await update.message.reply_text(render_table("Ranking Acciones (Top 10 Var%)", rows), parse_mode=ParseMode.HTML)


async def cmd_rank_ced(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = await get_cached_quotes("cedears")
    rows = sorted([(s, v["price"], v["var"]) for s, v in q.items() if v.get("var") is not None], key=lambda x: x[2], reverse=True)[:10]
    await update.message.reply_text(render_table("Ranking CEDEARs (Top 10 Var%)", rows), parse_mode=ParseMode.HTML)


async def cmd_resumen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # mezcla breve: dólares + macros + 5 noticias (fuentes que responden 200)
    dolares = await fetch_dolares()
    panel = render_dolares_panel(dolares)
    macros = [await fetch_reservas(), await fetch_inflacion(), await fetch_riesgo()]
    txt = f"<b>Resumen {datetime.now().strftime('%Y-%m-%d %H:%M')}</b>\n" + panel + "\n" + "\n".join(html.escape(x) for x in macros)
    await update.message.reply_text(txt, parse_mode=ParseMode.HTML, disable_web_page_preview=True)

# =====================
# Jobs (scheduler)
# =====================
async def job_refresh_dolares(context: ContextTypes.DEFAULT_TYPE):
    await fetch_dolares()


async def job_refresh_riesgo(context: ContextTypes.DEFAULT_TYPE):
    await fetch_riesgo()


async def job_prefetch(context: ContextTypes.DEFAULT_TYPE):
    await prefetch_quotes()


async def job_keepalive(context: ContextTypes.DEFAULT_TYPE):
    if not PUBLIC_BASE_URL:
        return
    webhook_url = f"{PUBLIC_BASE_URL}{WEBHOOK_PATH}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(webhook_url)
        logger.info("keepalive %s -> %s", webhook_url, r.status_code)
    except Exception as e:
        logger.warning("keepalive fail: %s", e)

# =====================
# Main
# =====================
async def main():
    defaults = Defaults(parse_mode=ParseMode.HTML, link_preview_options=LinkPreviewOptions(is_disabled=True))
    app = Application.builder().token(TELEGRAM_TOKEN).defaults(defaults).build()

    # Handlers
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("dolar", cmd_dolar))
    app.add_handler(CommandHandler("riesgo", cmd_riesgo))
    app.add_handler(CommandHandler("inflacion", cmd_inflacion))
    app.add_handler(CommandHandler("reservas", cmd_reservas))
    app.add_handler(CommandHandler("acciones", cmd_acciones))
    app.add_handler(CommandHandler("cedears", cmd_cedears))
    app.add_handler(CommandHandler("ranking_acciones", cmd_rank_byma))
    app.add_handler(CommandHandler("ranking_cedears", cmd_rank_ced))
    app.add_handler(CommandHandler("resumen_diario", cmd_resumen))

    # Jobs
    jq = app.job_queue
    jq.run_repeating(job_refresh_dolares, interval=120, first=2)
    jq.run_repeating(job_refresh_riesgo, interval=300, first=5)
    jq.run_repeating(job_prefetch, interval=900, first=4)
    jq.run_repeating(job_keepalive, interval=14*60, first=60)

    if PUBLIC_BASE_URL:
        # Asegura no-conflicto con polling anterior
        try:
            await app.bot.delete_webhook()
        except Exception:
            pass
        webhook_url = f"{PUBLIC_BASE_URL}{WEBHOOK_PATH}"
        logger.info("Levantando webhook en 0.0.0.0:%s path=%s", PORT, WEBHOOK_PATH)
        logger.info("Webhook URL = %s", webhook_url)
        await app.run_webhook(listen="0.0.0.0", port=PORT, url_path=WEBHOOK_PATH, webhook_url=webhook_url)
    else:
        logger.info("PUBLIC_BASE_URL no definido: usando polling")
        await app.run_polling()


if __name__ == "__main__":
    asyncio.run(main())
