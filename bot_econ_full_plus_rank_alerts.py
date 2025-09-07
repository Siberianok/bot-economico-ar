# bot_econ_full_plus_rank_alerts.py
# -*- coding: utf-8 -*-
#
# Telegram bot Económico AR para Render (webhook only, sin polling)
# - Endpoint raíz "/" para health/keepalive (200 OK)
# - Webhook en "/<WEBHOOK_SECRET>" (POST)
# - Comandos: /start /dolar /acciones /cedears /rankings /reservas /resumen_diario
#             /alertas /alertas_add /alertas_clear
# - Fuentes sin credenciales: DolarAPI, CriptoYa, ArgentinaDatos, Yahoo Finance v8, RSS (iProfesional, Clarín, La Nación, Ámbito)
#
# Requisitos (requirements.txt):
#   python-telegram-bot>=21.5
#   aiohttp>=3.9
#
# Comando de arranque (Render): python bot_econ_full_plus_rank_alerts.py
#
import os
import asyncio
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict, List, Tuple, Any, Optional

import re
from aiohttp import web, ClientSession, ClientTimeout

from telegram import Update, LinkPreviewOptions
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    Defaults,
)

# ------------------------------ Configuración ------------------------------

TZ = ZoneInfo("America/Argentina/Buenos_Aires")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "tgwebhook").strip().strip("/")
PORT = int(os.getenv("PORT", "10000"))
BASE_URL = os.getenv("BASE_URL", os.getenv("RENDER_EXTERNAL_URL", "https://bot-economico-ar.onrender.com")).rstrip("/")

if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN no configurado en variables de entorno.")

WEBHOOK_PATH = f"/{WEBHOOK_SECRET}"
WEBHOOK_URL = f"{BASE_URL}{WEBHOOK_PATH}"

# Endpoints externos
CRYPTOYA_DOLAR_URL = "https://criptoya.com/api/dolar"
DOLARAPI_BASE = "https://dolarapi.com/v1"
ARG_DATOS_BASES = [
    "https://api.argentinadatos.com/v1/finanzas/indices",  # principal (corregido)
    "https://argentinadatos.com/v1/finanzas/indices",       # fallback por si redirige
]
LAMACRO_RESERVAS_URL = "https://www.lamacro.ar/variables/1"

# Yahoo Finance v8 chart (por símbolo)
YF_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
YF_HEADERS = {"User-Agent": "Mozilla/5.0"}

# Noticias (RSS/XML)
RSS_FEEDS = [
    "https://www.iprofesional.com/rss",
    "https://www.clarin.com/rss/economia/",
    "https://www.lanacion.com.ar/economia/rss/",
    "https://www.ambito.com/contenidos/economia.xml",
]

# Listas base
ACCIONES_BA = [
    "GGAL.BA", "YPFD.BA", "PAMP.BA", "CEPU.BA", "ALUA.BA",
    "TXAR.BA", "TGSU2.BA", "BYMA.BA", "SUPV.BA", "BMA.BA",
]
CEDEARS_BA = [
    "AAPL.BA", "MSFT.BA", "NVDA.BA", "AMZN.BA", "GOOGL.BA",
    "TSLA.BA", "META.BA", "JNJ.BA", "KO.BA", "NFLX.BA",
]

# ------------------------------ Logging ------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("bot-econ-ar")

# ------------------------------ Utilidades ---------------------------------

def now_ar() -> datetime:
    return datetime.now(TZ)

def fmt_dt(dt: Optional[datetime]) -> str:
    if not dt:
        dt = now_ar()
    return dt.strftime("%d/%m/%Y %H:%M")

def fmt_number(n: Optional[float], nd=2) -> str:
    try:
        if n is None:
            return "—"
        s = f"{n:,.{nd}f}"
        # Convertir US->AR: 1,234,567.89 -> 1.234.567,89
        s = s.replace(",", "X").replace(".", ",").replace("X", ".")
        return s
    except Exception:
        return str(n)

def fmt_money_ars(n: Optional[float]) -> str:
    return f"$ {fmt_number(n, 2)}"

def pct(n: Optional[float]) -> str:
    try:
        return f"{n:+.2f}%"
    except Exception:
        return "—"

async def fetch_json(session: ClientSession, url: str, **kwargs) -> Optional[Dict[str, Any]]:
    try:
        timeout = kwargs.pop("timeout", ClientTimeout(total=12))
        async with session.get(url, timeout=timeout, **kwargs) as resp:
            if resp.status == 200:
                return await resp.json(content_type=None)
            else:
                log.warning("GET %s -> %s", url, resp.status)
                return None
    except Exception as e:
        log.warning("fetch_json error %s: %s", url, e)
        return None

async def fetch_text(session: ClientSession, url: str, **kwargs) -> Optional[str]:
    try:
        timeout = kwargs.pop("timeout", ClientTimeout(total=12))
        async with session.get(url, timeout=timeout, **kwargs) as resp:
            if resp.status == 200:
                return await resp.text()
            else:
                log.warning("GET %s -> %s", url, resp.status)
                return None
    except Exception as e:
        log.warning("fetch_text error %s: %s", url, e)
        return None

def anchor(href: str, text: str) -> str:
    return f'<a href="{href}">{text}</a>'

# ------------------------------ Dólares ------------------------------------

async def get_dolares(session: ClientSession) -> Dict[str, Dict[str, Any]]:
    """CriptoYa primero; DolarAPI como refuerzo/fallback."""
    data: Dict[str, Dict[str, Any]] = {}

    # 1) CriptoYa (single call)
    cj = await fetch_json(session, CRYPTOYA_DOLAR_URL)
    if cj:
        def _safe_get(block: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
            if not isinstance(block, dict):
                return (None, None)
            compra = block.get("compra") or block.get("buy")
            venta  = block.get("venta")  or block.get("sell")
            try:
                return (float(compra) if compra is not None else None,
                        float(venta)  if venta  is not None else None)
            except Exception:
                return (None, None)
        for k in ["oficial", "mayorista", "blue", "mep", "ccl", "cripto", "tarjeta"]:
            c, v = _safe_get(cj.get(k, {}))
            if c or v:
                data[k] = {"compra": c, "venta": v, "fuente": "CriptoYa"}

    # 2) DolarAPI (completar o fallback)
    async def dolarapi(path: str) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        url = f"{DOLARAPI_BASE}{path}"
        j = await fetch_json(session, url)
        if not j:
            return (None, None, None)
        c = j.get("compra")
        v = j.get("venta")
        fecha = j.get("fechaActualizacion")
        try:
            return (float(c) if c is not None else None, float(v) if v is not None else None, fecha)
        except Exception:
            return (None, None, fecha)

    mapping = {
        "oficial": "/dolares/oficial",
        "mayorista": "/ambito/dolares/mayorista",
        "blue": "/dolares/blue",
        "mep": "/dolares/bolsa",
        "ccl": "/dolares/contadoconliqui",
        "tarjeta": "/dolares/tarjeta",
        "cripto": "/ambito/dolares/cripto",
    }
    for k, path in mapping.items():
        if k not in data or (data[k].get("venta") is None and data[k].get("compra") is None):
            c, v, fecha = await dolarapi(path)
            if c is not None or v is not None:
                data[k] = {"compra": c, "venta": v, "fuente": "DolarAPI", "fecha": fecha}

    return data

# ------------------------------ ArgentinaDatos -----------------------------

async def _arg_datos_first_ok(session: ClientSession, path_suffix: str) -> Optional[Dict[str, Any]]:
    for base in ARG_DATOS_BASES:
        for variant in (f"{base}{path_suffix}", f"{base}{path_suffix}/"):
            j = await fetch_json(session, variant)
            if j:
                return j
    return None

async def get_riesgo_pais(session: ClientSession) -> Optional[int]:
    j = await _arg_datos_first_ok(session, "/riesgo-pais/ultimo")
    if not j:
        j = await _arg_datos_first_ok(session, "/riesgo-pais")
    if j:
        if isinstance(j, dict) and "valor" in j:
            try:
                return int(float(j["valor"]))
            except Exception:
                return None
        if isinstance(j, list) and j:
            try:
                return int(float(j[-1].get("valor")))
            except Exception:
                return None
    return None

async def get_inflacion_mensual(session: ClientSession) -> Optional[float]:
    j = await _arg_datos_first_ok(session, "/inflacion/mensual/ultimo")
    if not j:
        j = await _arg_datos_first_ok(session, "/inflacion/mensual")
    if j:
        if isinstance(j, dict) and "valor" in j:
            try:
                return float(j["valor"])
            except Exception:
                return None
        if isinstance(j, list) and j:
            try:
                return float(j[-1].get("valor"))
            except Exception:
                return None
    return None

# ------------------------------ Reservas (LaMacro) -------------------------

async def get_reservas_lamacro(session: ClientSession) -> Optional[Tuple[float, Optional[str]]]:
    """Devuelve (valor_en_MUS$, fecha_str). MUS$ = millones de USD."""
    html = await fetch_text(session, LAMACRO_RESERVAS_URL)
    if not html:
        return None
    m = re.search(r"Último dato:\s*([0-9\.\,]+)", html)
    fecha = None
    if not m:
        m = re.search(r"Valor actual:\s*([0-9\.\,]+)", html)
    mdate = re.search(r"Últ\. act:\s*([0-9]{2}/[0-9]{2}/[0-9]{4})", html)
    if mdate:
        fecha = mdate.group(1)
    if m:
        s = m.group(1).replace('.', '').replace(',', '.')
        try:
            val = float(s)
            return (val, fecha)
        except Exception:
            return None
    return None

# ------------------------------ Yahoo: Retornos (6m/3m/1m) -----------------

async def _yf_ret_pct(session: ClientSession, symbol: str, range_: str) -> Optional[float]:
    """Variación % usando chart v8 (adjclose), p.ej. range=6mo/3mo/1mo, interval=1d."""
    params = {"range": range_, "interval": "1d", "events": "div,split"}
    j = await fetch_json(session, YF_CHART_URL.format(symbol=symbol), headers=YF_HEADERS, params=params)
    try:
        res = j.get("chart", {}).get("result", [])[0]
        closes = res["indicators"]["adjclose"][0]["adjclose"]
        series = [c for c in closes if c is not None]
        if len(series) >= 2:
            first, last = series[0], series[-1]
            if first and last:
                return (last - first) / first * 100.0
    except Exception:
        return None
    return None

async def returns_for_symbols(session: ClientSession, symbols: List[str]) -> Dict[str, Dict[str, Optional[float]]]:
    """Retornos por símbolo para 6m/3m/1m con concurrencia limitada (rápido y estable)."""
    out: Dict[str, Dict[str, Optional[float]]] = {s: {"6m": None, "3m": None, "1m": None} for s in symbols}
    sem = asyncio.Semaphore(6)  # limitar concurrencia

    async def work(sym: str):
        async with sem:
            out[sym]["6m"] = await _yf_ret_pct(session, sym, "6mo")
        async with sem:
            out[sym]["3m"] = await _yf_ret_pct(session, sym, "3mo")
        async with sem:
            out[sym]["1m"] = await _yf_ret_pct(session, sym, "1mo")

    await asyncio.gather(*(work(s) for s in symbols))
    return out

def top_n_by_window(retmap: Dict[str, Dict[str, Optional[float]]], window: str, n=3) -> List[Tuple[str, float]]:
    pairs: List[Tuple[str, float]] = []
    for sym, d in retmap.items():
        v = d.get(window)
        if v is not None:
            pairs.append((sym, float(v)))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:n]

def rank_projection(retmap: Dict[str, Dict[str, Optional[float]]], n=5) -> List[Tuple[str, float, float, float]]:
    """Ranking 'proyección 6 meses' ~ momentum 6m (desempate 3m, luego 1m)."""
    syms = list(retmap.keys())
    syms.sort(key=lambda s: (
        -(retmap[s]["6m"] if retmap[s]["6m"] is not None else -1e9),
        -(retmap[s]["3m"] if retmap[s]["3m"] is not None else -1e9),
        -(retmap[s]["1m"] if retmap[s]["1m"] is not None else -1e9),
    ))
    out = []
    for s in syms:
        if retmap[s]["6m"] is None:
            continue
        out.append((s, retmap[s]["6m"], retmap[s]["3m"], retmap[s]["1m"]))
        if len(out) >= n:
            break
    return out

# ------------------------------ Noticias -----------------------------------

from xml.etree import ElementTree

async def fetch_rss_entries(session: ClientSession, limit: int = 5) -> List[Tuple[str, str]]:
    entries: List[Tuple[str, str]] = []
    for url in RSS_FEEDS:
        xml = await fetch_text(session, url)
        if not xml:
            continue
        try:
            root = ElementTree.fromstring(xml)
            for item in root.findall(".//item"):
                title_el = item.find("title")
                link_el = item.find("link")
                title = title_el.text.strip() if title_el is not None and title_el.text else None
                link = link_el.text.strip() if link_el is not None and link_el.text else None
                if title and link:
                    entries.append((title, link))
        except Exception as e:
            log.warning("RSS parse error %s: %s", url, e)
    seen = set()
    uniq = []
    for t, l in entries:
        if l not in seen:
            uniq.append((t, l))
            seen.add(l)
    return uniq[:limit]

# ------------------------------ Alertas ------------------------------------

ALERTS: Dict[int, List[Dict[str, Any]]] = {}

def parse_alert_add(args: List[str]) -> Optional[Tuple[str, str, float]]:
    if len(args) != 3:
        return None
    tipo = args[0].lower()
    op = args[1]
    try:
        val = float(args[2].replace(",", "."))
    except Exception:
        return None
    if tipo not in {"blue", "mep", "ccl", "riesgo", "inflacion", "reservas"}:
        return None
    if op not in {">", "<"}:
        return None
    return (tipo, op, val)

async def read_metrics_for_alerts(session: ClientSession) -> Dict[str, Optional[float]]:
    out = {"blue": None, "mep": None, "ccl": None, "riesgo": None, "inflacion": None, "reservas": None}
    d = await get_dolares(session)
    for k in ["blue", "mep", "ccl"]:
        if d.get(k) and d[k].get("venta") is not None:
            out[k] = float(d[k]["venta"])
    rp = await get_riesgo_pais(session)
    out["riesgo"] = float(rp) if rp is not None else None
    out["inflacion"] = await get_inflacion_mensual(session)
    rv = await get_reservas_lamacro(session)
    if rv:
        out["reservas"] = rv[0]
    return out

async def alerts_loop(app: Application):
    await asyncio.sleep(5)
    timeout = ClientTimeout(total=10)
    while True:
        try:
            if any(ALERTS.values()):
                async with ClientSession(timeout=timeout) as session:
                    vals = await read_metrics_for_alerts(session)
                for chat_id, rules in list(ALERTS.items()):
                    if not rules:
                        continue
                    triggered = []
                    for r in rules:
                        t, op, v = r["type"], r["op"], float(r["value"])
                        cur = vals.get(t)
                        if cur is None:
                            continue
                        ok = (cur > v) if op == ">" else (cur < v)
                        if ok:
                            triggered.append((t, op, v, cur))
                    if triggered:
                        lines = [f"<b>🔔 Alertas activadas</b>  <i>{fmt_dt(None)} hs</i>"]
                        for (t, op, v, cur) in triggered:
                            if t in {"blue", "mep", "ccl"}:
                                lines.append(f"{t.upper()}: {fmt_money_ars(cur)} ({op} {fmt_money_ars(v)})")
                            elif t == "riesgo":
                                lines.append(f"Riesgo país: {cur:.0f} pb ({op} {v:.0f} pb)")
                            elif t == "inflacion":
                                lines.append(f"Inflación mensual: {cur:.2f}% ({op} {v:.2f}%)")
                            elif t == "reservas":
                                lines.append(f"Reservas: {fmt_number(cur, 0)} MUS$ ({op} {fmt_number(v, 0)} MUS$)")
                        try:
                            await app.bot.send_message(chat_id=chat_id, text="\n".join(lines), parse_mode=ParseMode.HTML, link_preview_options=LinkPreviewOptions(is_disabled=True))
                        except Exception as e:
                            log.warning("send alert failed chat %s: %s", chat_id, e)
            await asyncio.sleep(600)
        except Exception as e:
            log.warning("alerts_loop error: %s", e)
            await asyncio.sleep(30)

# ------------------------------ Formateadores ------------------------------

def format_dolar_message(d: Dict[str, Dict[str, Any]]) -> str:
    parts = [f"<b>💵 Dólar (AR)</b>  <i>{fmt_dt(None)} hs</i>"]
    order = [
        ("oficial",   "Oficial"),
        ("mayorista", "Mayorista"),
        ("blue",      "Blue"),
        ("mep",       "MEP"),
        ("ccl",       "CCL"),
        ("cripto",    "Cripto"),
        ("tarjeta",   "Tarjeta"),
    ]
    for key, label in order:
        row = d.get(key)
        if not row:
            continue
        compra_s = fmt_money_ars(row.get("compra")) if row.get("compra") is not None else "—"
        venta_s  = fmt_money_ars(row.get("venta"))  if row.get("venta")  is not None else "—"
        parts.append(f"• <b>{label}:</b> compra {compra_s} | venta {venta_s}")
    return "\n".join(parts)

def format_top_block(title: str, items: List[Tuple[str, float]]) -> str:
    lines = [f"<u>{title}</u>"]
    if not items:
        lines.append("—")
        return "\n".join(lines)
    for sym, val in items:
        lines.append(f"• {anchor(f'https://finance.yahoo.com/quote/{sym}', sym)} {pct(val)}")
    return "\n".join(lines)

def format_proj_block(title: str, rows: List[Tuple[str, float, float, float]]) -> str:
    lines = [f"<u>{title}</u>"]
    if not rows:
        lines.append("—")
        return "\n".join(lines)
    for sym, r6, r3, r1 in rows:
        lines.append(f"• {anchor(f'https://finance.yahoo.com/quote/{sym}', sym)} 6m {pct(r6)} · 3m {pct(r3)} · 1m {pct(r1)}")
    return "\n".join(lines)

# ------------------------------ Handlers -----------------------------------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = (
        "¡Hola! Soy tu bot económico 🇦🇷 en Render (webhook).\n\n"
        "Comandos:\n"
        "• /dolar – Cotizaciones compra/venta (Oficial, Mayorista, Blue, MEP, CCL, Cripto, Tarjeta)\n"
        "• /acciones – Top 3 por ventana (6m, 3m, 1m) en BYMA (.BA)\n"
        "• /cedears – Top 3 por ventana (6m, 3m, 1m) en CEDEARs (.BA)\n"
        "• /rankings – Top 5 con mayor proyección (momentum 6m) en Acciones y CEDEARs\n"
        "• /reservas – Reservas internacionales BCRA (MUS$ = millones de USD)\n"
        "• /resumen_diario – Dólares + riesgo + reservas + inflación + 5 noticias con links\n"
        "• /alertas – Ver alertas | /alertas_add <tipo> <op> <valor> | /alertas_clear [tipo]\n"
    )
    ALERTS.setdefault(update.effective_chat.id, [])
    await update.effective_message.reply_text(
        txt,
        parse_mode=ParseMode.HTML,
        link_preview_options=LinkPreviewOptions(is_disabled=True),
    )

async def cmd_dolar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    async with ClientSession() as session:
        data = await get_dolares(session)
    msg = format_dolar_message(data) if data else "No pude obtener cotizaciones ahora."
    await update.effective_message.reply_text(
        msg,
        parse_mode=ParseMode.HTML,
        link_preview_options=LinkPreviewOptions(is_disabled=True),
    )

async def _top_blocks_for(update: Update, symbols: List[str], title: str):
    async with ClientSession() as session:
        rets = await returns_for_symbols(session, symbols)
    top6 = top_n_by_window(rets, "6m", n=3)
    top3 = top_n_by_window(rets, "3m", n=3)
    top1 = top_n_by_window(rets, "1m", n=3)
    lines = [f"<b>{title}</b>  <i>{fmt_dt(None)} hs</i>",
             format_top_block("Mejores 6M (Top 3)", top6),
             format_top_block("Mejores 3M (Top 3)", top3),
             format_top_block("Mejores 1M (Top 3)", top1)]
    return "\n".join(lines)

async def cmd_acciones(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await _top_blocks_for(update, ACCIONES_BA, "📈 Acciones BYMA (.BA)")
    await update.effective_message.reply_text(
        msg, parse_mode=ParseMode.HTML, link_preview_options=LinkPreviewOptions(is_disabled=True)
    )

async def cmd_cedears(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await _top_blocks_for(update, CEDEARS_BA, "🌎 CEDEARs (.BA)")
    await update.effective_message.reply_text(
        msg, parse_mode=ParseMode.HTML, link_preview_options=LinkPreviewOptions(is_disabled=True)
    )

async def cmd_rankings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbols = ACCIONES_BA + CEDEARS_BA
    async with ClientSession() as session:
        rets = await returns_for_symbols(session, symbols)
    acc = {k: v for k, v in rets.items() if k in ACCIONES_BA}
    ced = {k: v for k, v in rets.items() if k in CEDEARS_BA}
    acc_top = rank_projection(acc, n=5)
    ced_top = rank_projection(ced, n=5)
    lines = [f"<b>🏁 Rankings (proyección ~ momentum 6m)</b>  <i>{fmt_dt(None)} hs</i>"]
    if acc_top:
        lines.append(format_proj_block("Acciones – Top 5", acc_top))
    if ced_top:
        lines.append(format_proj_block("CEDEARs – Top 5", ced_top))
    if not (acc_top or ced_top):
        lines.append("No hay datos suficientes para el ranking ahora.")
    await update.effective_message.reply_text(
        "\n".join(lines),
        parse_mode=ParseMode.HTML,
        link_preview_options=LinkPreviewOptions(is_disabled=True),
    )

async def cmd_reservas(update: Update, context: ContextTypes.DEFAULT_TYPE):
    async with ClientSession() as session:
        res = await get_reservas_lamacro(session)
    if not res:
        txt = "No pude obtener reservas ahora."
    else:
        val, fecha = res
        fecha_str = f" (Últ. act: {fecha})" if fecha else ""
        txt = f"<b>🏦 Reservas BCRA</b>{fecha_str}\nValor: <b>{fmt_number(val, 0)} MUS$</b>  <i>(MUS$ = millones de USD)</i>"
    await update.effective_message.reply_text(
        txt, parse_mode=ParseMode.HTML, link_preview_options=LinkPreviewOptions(is_disabled=True)
    )

async def cmd_resumen_diario(update: Update, context: ContextTypes.DEFAULT_TYPE):
    async with ClientSession() as session:
        dolares = await get_dolares(session)
        riesgo = await get_riesgo_pais(session)
        reservas = await get_reservas_lamacro(session)
        inflacion = await get_inflacion_mensual(session)
        news = await fetch_rss_entries(session, limit=5)

    lines = [f"<b>🗞️ Resumen diario</b>  <i>{fmt_dt(None)} hs</i>"]

    if dolares:
        def one(key, label):
            row = dolares.get(key)
            if not row:
                return None
            venta = row.get("venta")
            return f"{label}: {fmt_money_ars(venta) if venta else '—'}"
        got = [one("blue", "Blue"), one("oficial", "Oficial"), one("mayorista", "May."), one("mep", "MEP"), one("ccl", "CCL")]
        got = [g for g in got if g]
        if got:
            lines.append("💵 " + " • ".join(got))

    rr = []
    if riesgo is not None:
        rr.append(f"Riesgo país: <b>{riesgo} pb</b>")
    if reservas:
        rv, rf = reservas
        rr.append(f"Reservas: <b>{fmt_number(rv, 0)} MUS$</b>{f' (act: {rf})' if rf else ''}")
    if inflacion is not None:
        rr.append(f"Inflación mensual: <b>{inflacion:.2f}%</b>")
    if rr:
        lines.append("📊 " + " • ".join(rr))

    if news:
        lines.append("<u>Top 5 noticias</u>")
        for t, l in news:
            lines.append(f"• {anchor(l, t)}")

    await update.effective_message.reply_text(
        "\n".join(lines),
        parse_mode=ParseMode.HTML,
        link_preview_options=LinkPreviewOptions(is_disabled=True),
    )

# ---------- Alert commands ----------

ALERTS: Dict[int, List[Dict[str, Any]]] = ALERTS  # mantener referencia

async def cmd_alertas_list(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    rules = ALERTS.get(chat_id, [])
    if not rules:
        msg = "No tenés alertas configuradas. Usá: /alertas_add <tipo> <op> <valor>"
    else:
        lines = ["<b>🔔 Alertas configuradas</b>"]
        for r in rules:
            t, op, v = r["type"], r["op"], r["value"]
            if t in {"blue", "mep", "ccl"}:
                val = fmt_money_ars(v)
            elif t == "riesgo":
                val = f"{v:.0f} pb"
            elif t == "reservas":
                val = f"{fmt_number(v,0)} MUS$"
            else:
                val = f"{v:.2f}%"
            lines.append(f"• {t.upper()} {op} {val}")
        msg = "\n".join(lines)
    await update.effective_message.reply_text(msg, parse_mode=ParseMode.HTML, link_preview_options=LinkPreviewOptions(is_disabled=True))

async def cmd_alertas_add(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    def parse_alert_add(args: List[str]) -> Optional[Tuple[str, str, float]]:
        if len(args) != 3:
            return None
        tipo = args[0].lower()
        op = args[1]
        try:
            val = float(args[2].replace(",", "."))
        except Exception:
            return None
        if tipo not in {"blue", "mep", "ccl", "riesgo", "inflacion", "reservas"}:
            return None
        if op not in {">", "<"}:
            return None
        return (tipo, op, val)

    parsed = parse_alert_add(args)
    if not parsed:
        await update.effective_message.reply_text(
            "Formato: /alertas_add <tipo> <op> <valor>\n"
            "  tipos: blue, mep, ccl, riesgo, inflacion, reservas\n"
            "  op: > o <\n"
            "Ejemplos:\n"
            "  /alertas_add blue > 1350\n"
            "  /alertas_add riesgo > 850\n"
            "  /alertas_add inflacion > 4",
            parse_mode=ParseMode.HTML,
            link_preview_options=LinkPreviewOptions(is_disabled=True),
        )
        return
    tipo, op, val = parsed
    chat_id = update.effective_chat.id
    ALERTS.setdefault(chat_id, []).append({"type": tipo, "op": op, "value": val})
    await update.effective_message.reply_text("Listo. Alerta agregada ✅", parse_mode=ParseMode.HTML, link_preview_options=LinkPreviewOptions(is_disabled=True))

async def cmd_alertas_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    tipo = (context.args[0].lower() if context.args else None)
    if not ALERTS.get(chat_id):
        await update.effective_message.reply_text("No hay alertas para borrar.", parse_mode=ParseMode.HTML, link_preview_options=LinkPreviewOptions(is_disabled=True))
        return
    if tipo:
        before = len(ALERTS[chat_id])
        ALERTS[chat_id] = [r for r in ALERTS[chat_id] if r["type"] != tipo]
        after = len(ALERTS[chat_id])
        await update.effective_message.reply_text(f"Eliminadas {before - after} alertas de tipo {tipo.upper()}.", parse_mode=ParseMode.HTML, link_preview_options=LinkPreviewOptions(is_disabled=True))
    else:
        ALERTS[chat_id] = []
        await update.effective_message.reply_text("Todas las alertas fueron eliminadas.", parse_mode=ParseMode.HTML, link_preview_options=LinkPreviewOptions(is_disabled=True))

# ------------------------------ AIOHTTP + Webhook --------------------------

async def keepalive_loop(app: Application):
    await asyncio.sleep(5)
    url = f"{BASE_URL}/"
    timeout = ClientTimeout(total=5)
    async with ClientSession(timeout=timeout) as session:
        while True:
            try:
                async with session.get(url) as resp:
                    log.info("Keepalive %s -> %s", url, resp.status)
            except Exception as e:
                log.warning("Keepalive error: %s", e)
            await asyncio.sleep(300)  # 5 min

async def on_startup(app: web.Application):
    await application.initialize()
    await application.start()
    await application.bot.set_webhook(url=WEBHOOK_URL, allowed_updates=["message"], drop_pending_updates=True)
    log.info("Webhook set: %s", WEBHOOK_URL)
    asyncio.create_task(keepalive_loop(application))
    asyncio.create_task(alerts_loop(application))

async def on_cleanup(app: web.Application):
    await application.stop()
    await application.shutdown()

async def handle_root(request: web.Request):
    return web.Response(text="ok", status=200)

async def handle_webhook(request: web.Request):
    if request.method != "POST":
        return web.Response(text="Method Not Allowed", status=405)
    try:
        data = await request.json()
    except Exception:
        return web.Response(text="Bad Request", status=400)
    update = Update.de_json(data, application.bot)
    await application.process_update(update)
    return web.Response(text="OK", status=200)

def build_web_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/", handle_root)
    app.router.add_post(WEBHOOK_PATH, handle_webhook)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app

# ------------------------------ PTB Application ----------------------------

defaults = Defaults(parse_mode=ParseMode.HTML, link_preview_options=LinkPreviewOptions(is_disabled=True), tzinfo=TZ)
application = Application.builder().token(TELEGRAM_TOKEN).defaults(defaults).updater(None).build()

# Handlers
application.add_handler(CommandHandler("start", cmd_start))
application.add_handler(CommandHandler("dolar", cmd_dolar))
application.add_handler(CommandHandler("acciones", cmd_acciones))
application.add_handler(CommandHandler("cedears", cmd_cedears))
application.add_handler(CommandHandler("rankings", cmd_rankings))
application.add_handler(CommandHandler("reservas", cmd_reservas))
application.add_handler(CommandHandler("resumen_diario", cmd_resumen_diario))
application.add_handler(CommandHandler("alertas", cmd_alertas_list))
application.add_handler(CommandHandler("alertas_add", cmd_alertas_add))
application.add_handler(CommandHandler("alertas_clear", cmd_alertas_clear))

# ------------------------------ Main ---------------------------------------

if __name__ == "__main__":
    log.info("Iniciando bot Económico AR (Render webhook)")
    app = build_web_app()
    web.run_app(app, host="0.0.0.0", port=PORT)
