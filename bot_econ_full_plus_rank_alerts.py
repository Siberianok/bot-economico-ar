# bot_econ_full_plus_rank_alerts.py
# -*- coding: utf-8 -*-
#
# Telegram bot Económico AR para Render (webhook only, sin polling)
# - Endpoint raíz "/" para health/keepalive (200 OK)
# - Webhook en "/<WEBHOOK_SECRET>" (POST)
# - Comandos:
#   /dolar /acciones /cedears
#   /rankings /rankings_acciones /rankings_cedears
#   /reservas /inflacion /riesgo /resumen_diario
#   /alertas /alertas_add /alertas_clear
#
# Fuentes sin credenciales: CriptoYa, DolarAPI, ArgentinaDatos, Yahoo Finance v8, RSS (iProfesional/Clarín/La Nación/Ámbito)
#
# requirements.txt:
#   python-telegram-bot>=21.5
#   aiohttp>=3.9
#
# Start (Render): python bot_econ_full_plus_rank_alerts.py
#
import os
import asyncio
import logging
import re
from time import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, List, Tuple, Any, Optional

from aiohttp import web, ClientSession, ClientTimeout
from telegram import Update, LinkPreviewOptions
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes, Defaults

# ------------------------------ Config ------------------------------

TZ = ZoneInfo("America/Argentina/Buenos_Aires")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "tgwebhook").strip().strip("/")
PORT = int(os.getenv("PORT", "10000"))
BASE_URL = os.getenv("BASE_URL", os.getenv("RENDER_EXTERNAL_URL", "https://bot-economico-ar.onrender.com")).rstrip("/")

if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN no configurado.")

WEBHOOK_PATH = f"/{WEBHOOK_SECRET}"
WEBHOOK_URL = f"{BASE_URL}{WEBHOOK_PATH}"

# Endpoints
CRYPTOYA_DOLAR_URL = "https://criptoya.com/api/dolar"
DOLARAPI_BASE = "https://dolarapi.com/v1"
ARG_DATOS_BASES = [
    "https://api.argentinadatos.com/v1/finanzas/indices",  # principal
    "https://argentinadatos.com/v1/finanzas/indices",       # fallback
]
LAMACRO_RESERVAS_URL = "https://www.lamacro.ar/variables/1"

# Yahoo Finance v8 chart
YF_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
YF_HEADERS = {"User-Agent": "Mozilla/5.0"}

# Noticias (RSS)
RSS_FEEDS = [
    "https://www.iprofesional.com/rss",
    "https://www.clarin.com/rss/economia/",
    "https://www.lanacion.com.ar/economia/rss/",
    "https://www.ambito.com/contenidos/economia.xml",
]

# Listas base
ACCIONES_BA = ["GGAL.BA","YPFD.BA","PAMP.BA","CEPU.BA","ALUA.BA","TXAR.BA","TGSU2.BA","BYMA.BA","SUPV.BA","BMA.BA"]
CEDEARS_BA  = ["AAPL.BA","MSFT.BA","NVDA.BA","AMZN.BA","GOOGL.BA","TSLA.BA","META.BA","JNJ.BA","KO.BA","NFLX.BA"]

# ------------------------------ Logging ------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("bot-econ-ar")

# ------------------------------ Utils --------------------------------------

def now_ar() -> datetime:
    return datetime.now(TZ)

def fmt_dt(dt: Optional[datetime] = None) -> str:
    if dt is None:
        dt = now_ar()
    return dt.strftime("%d/%m/%Y %H:%M")

def fmt_number(n: Optional[float], nd=2) -> str:
    try:
        if n is None: return "—"
        s = f"{n:,.{nd}f}"
        return s.replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return str(n)

def fmt_money_ars(n: Optional[float]) -> str:
    return f"$ {fmt_number(n, 2)}"

def pct(n: Optional[float]) -> str:
    try:
        return f"{n:+.2f}%"
    except Exception:
        return "—"

def anchor(href: str, text: str) -> str:
    return f'<a href="{href}">{text}</a>'

MES_ES = {"01":"ene","02":"feb","03":"mar","04":"abr","05":"may","06":"jun","07":"jul","08":"ago","09":"sep","10":"oct","11":"nov","12":"dic"}
def fmt_periodo(fecha: Optional[str]) -> Optional[str]:
    """Acepta 'YYYY-MM' o 'YYYY-MM-DD' y devuelve 'mmm-YYYY'."""
    if not fecha or len(fecha) < 7: return None
    y = fecha[0:4]; m = fecha[5:7]
    return f"{MES_ES.get(m, m)}-{y}"

# ------------------------------ HTTP helpers --------------------------------

async def fetch_json(session: ClientSession, url: str, **kwargs) -> Optional[Dict[str, Any]]:
    try:
        timeout = kwargs.pop("timeout", ClientTimeout(total=15))
        async with session.get(url, timeout=timeout, **kwargs) as resp:
            if resp.status == 200:
                return await resp.json(content_type=None)
            log.warning("GET %s -> %s", url, resp.status)
    except Exception as e:
        log.warning("fetch_json error %s: %s", url, e)
    return None

async def fetch_text(session: ClientSession, url: str, **kwargs) -> Optional[str]:
    try:
        timeout = kwargs.pop("timeout", ClientTimeout(total=15))
        async with session.get(url, timeout=timeout, **kwargs) as resp:
            if resp.status == 200:
                return await resp.text()
            log.warning("GET %s -> %s", url, resp.status)
    except Exception as e:
        log.warning("fetch_text error %s: %s", url, e)
    return None

# ------------------------------ Dólares ------------------------------------

async def get_dolares(session: ClientSession) -> Dict[str, Dict[str, Any]]:
    """CriptoYa primero; DolarAPI fallback/complemento."""
    data: Dict[str, Dict[str, Any]] = {}

    # CriptoYa
    cj = await fetch_json(session, CRYPTOYA_DOLAR_URL)
    if cj:
        def _safe(block: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
            if not isinstance(block, dict): return (None, None)
            c, v = block.get("compra") or block.get("buy"), block.get("venta") or block.get("sell")
            try:
                return (float(c) if c is not None else None, float(v) if v is not None else None)
            except Exception:
                return (None, None)
        for k in ["oficial","mayorista","blue","mep","ccl","cripto","tarjeta"]:
            c,v = _safe(cj.get(k,{}))
            if c is not None or v is not None:
                data[k] = {"compra": c, "venta": v, "fuente": "CriptoYa"}

    # DolarAPI
    async def dolarapi(path: str):
        j = await fetch_json(session, f"{DOLARAPI_BASE}{path}")
        if not j: return (None, None, None)
        c,v,fecha = j.get("compra"), j.get("venta"), j.get("fechaActualizacion") or j.get("fecha")
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
        if k not in data or (data[k].get("compra") is None and data[k].get("venta") is None):
            c,v,fecha = await dolarapi(path)
            if c is not None or v is not None:
                data[k] = {"compra": c, "venta": v, "fuente": "DolarAPI", "fecha": fecha}
    return data

# ------------------------------ ArgentinaDatos -----------------------------

async def arg_datos_get(session: ClientSession, suffix: str) -> Optional[Dict[str, Any]]:
    for base in ARG_DATOS_BASES:
        for u in (f"{base}{suffix}", f"{base}{suffix}/"):
            j = await fetch_json(session, u)
            if j: return j
    return None

async def get_riesgo_pais(session: ClientSession) -> Optional[Tuple[int, Optional[str]]]:
    """Devuelve (valor_pb, fecha 'YYYY-MM-DD' o similar)."""
    j = await arg_datos_get(session, "/riesgo-pais/ultimo")
    if not j:
        j = await arg_datos_get(session, "/riesgo-pais")
        if isinstance(j, list) and j:
            last = j[-1]
            val = last.get("valor"); f = last.get("fecha") or last.get("periodo")
            try: return (int(float(val)), f) if val is not None else None
            except Exception: return None
    if isinstance(j, dict):
        val = j.get("valor"); f = j.get("fecha") or j.get("periodo")
        try: return (int(float(val)), f) if val is not None else None
        except Exception: return None
    return None

async def get_inflacion_mensual(session: ClientSession) -> Optional[Tuple[float, Optional[str]]]:
    """Usa /v1/finanzas/indices/inflacion (lista) y toma el último."""
    j = await arg_datos_get(session, "/inflacion")
    # si el doc cambia de path, probar también índice plural (compat)
    if not j:
        j = await arg_datos_get(session, "/inflacion/mensual")  # compat
    if isinstance(j, list) and j:
        last = j[-1]
        val = last.get("valor")
        per = last.get("periodo") or last.get("fecha")
        try:
            return (float(val), per) if val is not None else None
        except Exception:
            return None
    if isinstance(j, dict):
        val = j.get("valor")
        per = j.get("periodo") or j.get("fecha")
        try:
            return (float(val), per) if val is not None else None
        except Exception:
            return None
    return None

# ------------------------------ Reservas (LaMacro) -------------------------

async def get_reservas_lamacro(session: ClientSession) -> Optional[Tuple[float, Optional[str]]]:
    """Devuelve (valor_en_MUS$, fecha_str). MUS$ = millones de USD."""
    html = await fetch_text(session, LAMACRO_RESERVAS_URL)
    if not html: return None
    # Valor
    m_val = re.search(r"(?:Último dato|Valor actual)\s*:\s*([0-9\.\,]+)", html)
    # Fecha: capturá el primer dd/mm/aaaa que aparezca
    m_date = re.search(r"([0-3]\d/[0-1]\d/\d{4})", html)
    fecha = m_date.group(1) if m_date else None
    if m_val:
        s = m_val.group(1).replace('.', '').replace(',', '.')
        try: return (float(s), fecha)
        except Exception: return None
    return None

# ------------------------------ Yahoo retornos (ultra-rápido) --------------

# Caché
RET_CACHE_1Y: Dict[str, Tuple[float, Optional[Dict[str, Any]]]] = {}  # symbol -> (ts, payload_result)
RET_TTL = 600  # 10 minutos

async def _yf_chart_1y(session: ClientSession, symbol: str, interval: str) -> Optional[Dict[str, Any]]:
    """Trae 1 año de data con interval 1d o 1wk (según parámetro)."""
    now_ts = time()
    cache_key = f"{symbol}|{interval}"
    ts_val = RET_CACHE_1Y.get(cache_key)
    if ts_val and now_ts - ts_val[0] < RET_TTL:
        return ts_val[1]
    params = {"range": "1y", "interval": interval, "events": "div,split"}
    j = await fetch_json(session, YF_CHART_URL.format(symbol=symbol), headers=YF_HEADERS, params=params)
    try:
        res = j.get("chart", {}).get("result", [])[0]
        RET_CACHE_1Y[cache_key] = (now_ts, res)
        return res
    except Exception:
        RET_CACHE_1Y[cache_key] = (now_ts, None)
        return None

def _returns_from_1y(res: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
    """Calcula 6m/3m/1m a partir de la serie 1y."""
    try:
        ts = res["timestamp"]
        closes = res["indicators"]["adjclose"][0]["adjclose"]
        # ult valor válido
        idx_last = max(i for i,v in enumerate(closes) if v is not None)
        last = closes[idx_last]
        t_last = ts[idx_last]
        # límites de tiempo
        t6 = t_last - 180*24*3600
        t3 = t_last - 90*24*3600
        t1 = t_last - 30*24*3600
        def first_on_or_after(tcut):
            for i, t in enumerate(ts):
                if t is None or closes[i] is None: 
                    continue
                if t >= tcut:
                    return closes[i]
            # si no hay, usar primer válido
            for i, v in enumerate(closes):
                if v is not None: return v
            return None
        c6 = first_on_or_after(t6)
        c3 = first_on_or_after(t3)
        c1 = first_on_or_after(t1)
        def r(base):
            return (last - base) / base * 100.0 if base else None
        return (r(c6), r(c3), r(c1))
    except Exception:
        return None

async def yf_returns_6_3_1(session: ClientSession, symbol: str) -> Dict[str, Optional[float]]:
    """1 llamada por símbolo (1y). Fallback a semanal si hace falta."""
    res = await _yf_chart_1y(session, symbol, "1d")
    out = {"6m": None, "3m": None, "1m": None}
    if res:
        trio = _returns_from_1y(res)
        if trio:
            out["6m"], out["3m"], out["1m"] = trio
            return out
    # fallback semanal
    res = await _yf_chart_1y(session, symbol, "1wk")
    if res:
        trio = _returns_from_1y(res)
        if trio:
            out["6m"], out["3m"], out["1m"] = trio
    return out

async def returns_for_symbols(session: ClientSession, symbols: List[str]) -> Dict[str, Dict[str, Optional[float]]]:
    out = {s: {"6m": None, "3m": None, "1m": None} for s in symbols}
    sem = asyncio.Semaphore(5)  # cuidadoso con 429
    async def work(sym: str):
        async with sem:
            out[sym] = await yf_returns_6_3_1(session, sym)
    await asyncio.gather(*(work(s) for s in symbols))
    return out

def top_n_by_window(retmap: Dict[str, Dict[str, Optional[float]]], window: str, n=3) -> List[Tuple[str, float]]:
    pairs = [(sym, float(v)) for sym,d in retmap.items() if (v:=d.get(window)) is not None]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:n]

def rank_projection(retmap: Dict[str, Dict[str, Optional[float]]], n=5) -> List[Tuple[str, float, float, float]]:
    syms = list(retmap.keys())
    syms.sort(key=lambda s: (
        -(retmap[s]["6m"] if retmap[s]["6m"] is not None else -1e9),
        -(retmap[s]["3m"] if retmap[s]["3m"] is not None else -1e9),
        -(retmap[s]["1m"] if retmap[s]["1m"] is not None else -1e9),
    ))
    out = []
    for s in syms:
        if retmap[s]["6m"] is None: continue
        out.append((s, retmap[s]["6m"], retmap[s]["3m"], retmap[s]["1m"]))
        if len(out) >= n: break
    return out

# ------------------------------ Noticias -----------------------------------

from xml.etree import ElementTree

async def fetch_rss_entries(session: ClientSession, limit: int = 5) -> List[Tuple[str, str]]:
    entries = []
    for url in RSS_FEEDS:
        xml = await fetch_text(session, url)
        if not xml: continue
        try:
            root = ElementTree.fromstring(xml)
            for item in root.findall(".//item"):
                t_el, l_el = item.find("title"), item.find("link")
                t = t_el.text.strip() if t_el is not None and t_el.text else None
                l = l_el.text.strip() if l_el is not None and l_el.text else None
                if t and l: entries.append((t, l))
        except Exception as e:
            log.warning("RSS parse %s: %s", url, e)
    seen, uniq = set(), []
    for t,l in entries:
        if l not in seen:
            uniq.append((t,l)); seen.add(l)
    return uniq[:limit]

# ------------------------------ Alertas ------------------------------------

ALERTS: Dict[int, List[Dict[str, Any]]] = {}

def parse_alert_add(args: List[str]) -> Optional[Tuple[str, str, float]]:
    if len(args) != 3: return None
    tipo, op = args[0].lower(), args[1]
    try: val = float(args[2].replace(",", "."))
    except Exception: return None
    if tipo not in {"blue","mep","ccl","riesgo","inflacion","reservas"}: return None
    if op not in {">","<"}: return None
    return (tipo, op, val)

async def read_metrics_for_alerts(session: ClientSession) -> Dict[str, Optional[float]]:
    out = {"blue":None,"mep":None,"ccl":None,"riesgo":None,"inflacion":None,"reservas":None}
    d = await get_dolares(session)
    for k in ["blue","mep","ccl"]:
        if d.get(k) and d[k].get("venta") is not None:
            out[k] = float(d[k]["venta"])
    rp = await get_riesgo_pais(session)
    out["riesgo"] = float(rp[0]) if rp is not None else None
    infl = await get_inflacion_mensual(session)
    out["inflacion"] = float(infl[0]) if infl is not None else None
    rv = await get_reservas_lamacro(session)
    if rv: out["reservas"] = rv[0]
    return out

async def alerts_loop(app: Application):
    await asyncio.sleep(5)
    timeout = ClientTimeout(total=12)
    while True:
        try:
            if any(ALERTS.values()):
                async with ClientSession(timeout=timeout) as session:
                    vals = await read_metrics_for_alerts(session)
                for chat_id, rules in list(ALERTS.items()):
                    if not rules: continue
                    trig = []
                    for r in rules:
                        cur = vals.get(r["type"])
                        if cur is None: continue
                        ok = (cur > r["value"]) if r["op"] == ">" else (cur < r["value"])
                        if ok: trig.append((r["type"], r["op"], r["value"], cur))
                    if trig:
                        lines = [f"<b>🔔 Alertas</b>  <i>{fmt_dt()}</i>"]
                        for t,op,v,cur in trig:
                            if t in {"blue","mep","ccl"}:
                                lines.append(f"{t.upper()}: {fmt_money_ars(cur)} ({op} {fmt_money_ars(v)})")
                            elif t=="riesgo":
                                lines.append(f"Riesgo país: {cur:.0f} pb ({op} {v:.0f} pb)")
                            elif t=="inflacion":
                                lines.append(f"Inflación mensual: {cur:.2f}% ({op} {v:.2f}%)")
                            elif t=="reservas":
                                lines.append(f"Reservas: {fmt_number(cur,0)} MUS$ ({op} {fmt_number(v,0)} MUS$)")
                        try:
                            await app.bot.send_message(chat_id, "\n".join(lines), parse_mode=ParseMode.HTML, link_preview_options=LinkPreviewOptions(is_disabled=True))
                        except Exception as e:
                            log.warning("send alert failed %s: %s", chat_id, e)
            await asyncio.sleep(600)
        except Exception as e:
            log.warning("alerts_loop error: %s", e)
            await asyncio.sleep(30)

# ------------------------------ Formatos -----------------------------------

def format_dolar_message(d: Dict[str, Dict[str, Any]]) -> str:
    lines = [f"<b>💵 Dólar (AR)</b>  <i>{fmt_dt()}</i>"]
    order = [("oficial","Oficial"),("mayorista","Mayorista"),("blue","Blue"),("mep","MEP"),("ccl","CCL"),("cripto","Cripto"),("tarjeta","Tarjeta")]
    for k, label in order:
        row = d.get(k)
        if not row: continue
        compra = fmt_money_ars(row.get("compra")) if row.get("compra") is not None else "—"
        venta  = fmt_money_ars(row.get("venta"))  if row.get("venta")  is not None else "—"
        lines.append(f"• <b>{label}:</b> compra {compra} | venta {venta}")
    return "\n".join(lines)

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

async def cmd_dolar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    async with ClientSession() as session:
        data = await get_dolares(session)
    msg = format_dolar_message(data) if data else "No pude obtener cotizaciones ahora."
    await update.effective_message.reply_text(msg, parse_mode=ParseMode.HTML, link_preview_options=LinkPreviewOptions(is_disabled=True))

async def _top_blocks_for(symbols: List[str], title: str) -> str:
    async with ClientSession() as session:
        # timeout corto para evitar “no muestra”
        try:
            rets = await asyncio.wait_for(returns_for_symbols(session, symbols), timeout=10)
        except asyncio.TimeoutError:
            rets = {s: {"6m": None, "3m": None, "1m": None} for s in symbols}
    top6 = top_n_by_window(rets, "6m", 3)
    top3 = top_n_by_window(rets, "3m", 3)
    top1 = top_n_by_window(rets, "1m", 3)
    return "\n".join([
        f"<b>{title}</b>  <i>{fmt_dt()}</i>",
        format_top_block("Mejores 6M (Top 3)", top6),
        format_top_block("Mejores 3M (Top 3)", top3),
        format_top_block("Mejores 1M (Top 3)", top1),
    ])

async def cmd_acciones(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(await _top_blocks_for(ACCIONES_BA, "📈 Acciones BYMA (.BA)"), parse_mode=ParseMode.HTML, link_preview_options=LinkPreviewOptions(is_disabled=True))

async def cmd_cedears(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(await _top_blocks_for(CEDEARS_BA, "🌎 CEDEARs (.BA)"), parse_mode=ParseMode.HTML, link_preview_options=LinkPreviewOptions(is_disabled=True))

async def cmd_rankings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    async with ClientSession() as session:
        try:
            acc, ced = await asyncio.wait_for(asyncio.gather(
                returns_for_symbols(session, ACCIONES_BA),
                returns_for_symbols(session, CEDEARS_BA)
            ), timeout=12)
        except asyncio.TimeoutError:
            acc, ced = {}, {}
    acc_top = rank_projection(acc, 5) if acc else []
    ced_top = rank_projection(ced, 5) if ced else []
    lines = [f"<b>🏁 Rankings (proyección ~ momentum 6m)</b>  <i>{fmt_dt()}</i>"]
    lines.append(format_proj_block("Acciones – Top 5", acc_top))
    lines.append(format_proj_block("CEDEARs – Top 5", ced_top))
    await update.effective_message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML, link_preview_options=LinkPreviewOptions(is_disabled=True))

async def cmd_rankings_acciones(update: Update, context: ContextTypes.DEFAULT_TYPE):
    async with ClientSession() as session:
        try:
            acc = await asyncio.wait_for(returns_for_symbols(session, ACCIONES_BA), timeout=10)
        except asyncio.TimeoutError:
            acc = {}
    acc_top = rank_projection(acc, 5) if acc else []
    lines = [f"<b>🏁 Acciones – Rankings</b>  <i>{fmt_dt()}</i>", format_proj_block("Top 5", acc_top)]
    await update.effective_message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML, link_preview_options=LinkPreviewOptions(is_disabled=True))

async def cmd_rankings_cedears(update: Update, context: ContextTypes.DEFAULT_TYPE):
    async with ClientSession() as session:
        try:
            ced = await asyncio.wait_for(returns_for_symbols(session, CEDEARS_BA), timeout=10)
        except asyncio.TimeoutError:
            ced = {}
    ced_top = rank_projection(ced, 5) if ced else []
    lines = [f"<b>🏁 CEDEARs – Rankings</b>  <i>{fmt_dt()}</i>", format_proj_block("Top 5", ced_top)]
    await update.effective_message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML, link_preview_options=LinkPreviewOptions(is_disabled=True))

async def cmd_reservas(update: Update, context: ContextTypes.DEFAULT_TYPE):
    async with ClientSession() as session:
        res = await get_reservas_lamacro(session)
    if not res:
        txt = "No pude obtener reservas ahora."
    else:
        val, fecha = res
        txt = f"<b>🏦 Reservas BCRA</b>{f'  <i>Últ. act: {fecha}</i>' if fecha else ''}\n<b>{fmt_number(val,0)} MUS$</b> (MUS$ = millones de USD)"
    await update.effective_message.reply_text(txt, parse_mode=ParseMode.HTML, link_preview_options=LinkPreviewOptions(is_disabled=True))

async def cmd_inflacion(update: Update, context: ContextTypes.DEFAULT_TYPE):
    async with ClientSession() as session:
        tup = await get_inflacion_mensual(session)
    if tup is None:
        txt = "No pude obtener inflación ahora."
    else:
        val, per = tup
        per_s = fmt_periodo(per) or per or ""
        txt = f"<b>📉 Inflación mensual</b>{f'  <i>{per_s}</i>' if per_s else ''}\n<b>{val:.2f}%</b>"
    await update.effective_message.reply_text(txt, parse_mode=ParseMode.HTML, link_preview_options=LinkPreviewOptions(is_disabled=True))

async def cmd_riesgo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    async with ClientSession() as session:
        tup = await get_riesgo_pais(session)
    if tup is None:
        txt = "No pude obtener riesgo país ahora."
    else:
        rp, f = tup
        f_str = None
        if f and len(f) >= 10 and f[4] == "-" and f[7] == "-":
            try:
                f_str = datetime.strptime(f[:10], "%Y-%m-%d").strftime("%d/%m/%Y")
            except Exception:
                f_str = f
        else:
            f_str = f
        txt = f"<b>📈 Riesgo país</b>{f'  <i>{f_str}</i>' if f_str else ''}\n<b>{rp} pb</b>"
    await update.effective_message.reply_text(txt, parse_mode=ParseMode.HTML, link_preview_options=LinkPreviewOptions(is_disabled=True))

async def cmd_resumen_diario(update: Update, context: ContextTypes.DEFAULT_TYPE):
    async with ClientSession() as session:
        dolares = await get_dolares(session)
        riesgo_t  = await get_riesgo_pais(session)       # (valor, fecha)
        reservas  = await get_reservas_lamacro(session)  # (valor, fecha dd/mm/aaaa)
        inflac_t  = await get_inflacion_mensual(session) # (valor, periodo)
        news      = await fetch_rss_entries(session, limit=5)

    blocks = [f"<b>🗞️ Resumen diario</b>  <i>{fmt_dt()}</i>"]

    if dolares:
        blocks.append(format_dolar_message(dolares))

    if riesgo_t:
        rp, f = riesgo_t
        f_str = None
        if f and len(f) >= 10 and f[4] == "-" and f[7] == "-":
            try: f_str = datetime.strptime(f[:10], "%Y-%m-%d").strftime("%d/%m/%Y")
            except Exception: f_str = f
        else:
            f_str = f
        blocks.append(f"<b>📈 Riesgo país</b>{f'  <i>{f_str}</i>' if f_str else ''}\n<b>{rp} pb</b>")
    else:
        blocks.append("<b>📈 Riesgo país</b>\n—")

    if reservas:
        rv, rf = reservas
        blocks.append(f"<b>🏦 Reservas BCRA</b>{f'  <i>Últ. act: {rf}</i>' if rf else ''}\n<b>{fmt_number(rv,0)} MUS$</b>")
    else:
        blocks.append("<b>🏦 Reservas BCRA</b>\n—")

    if inflac_t:
        iv, ip = inflac_t
        ip_s = fmt_periodo(ip) or ip or ""
        blocks.append(f"<b>📉 Inflación mensual</b>{f'  <i>{ip_s}</i>' if ip_s else ''}\n<b>{iv:.2f}%</b>")
    else:
        blocks.append("<b>📉 Inflación mensual</b>\n—")

    if news:
        lines = ["<u>Top 5 noticias</u>"]
        lines += [f"• {anchor(l, t)}" for t,l in news]
        blocks.append("\n".join(lines))

    await update.effective_message.reply_text("\n\n".join(blocks), parse_mode=ParseMode.HTML, link_preview_options=LinkPreviewOptions(is_disabled=True))

# ---------- Alert commands ----------

async def cmd_alertas_list(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    rules = ALERTS.get(chat_id, [])
    if not rules:
        txt = "No tenés alertas. Usá: /alertas_add <tipo> <op> <valor>"
    else:
        lines = ["<b>🔔 Alertas configuradas</b>"]
        for r in rules:
            t,op,v = r["type"], r["op"], r["value"]
            if t in {"blue","mep","ccl"}: val = fmt_money_ars(v)
            elif t=="riesgo": val = f"{v:.0f} pb"
            elif t=="reservas": val = f"{fmt_number(v,0)} MUS$"
            else: val = f"{v:.2f}%"
            lines.append(f"• {t.upper()} {op} {val}")
        txt = "\n".join(lines)
    await update.effective_message.reply_text(txt, parse_mode=ParseMode.HTML, link_preview_options=LinkPreviewOptions(is_disabled=True))

async def cmd_alertas_add(update: Update, context: ContextTypes.DEFAULT_TYPE):
    parsed = parse_alert_add(context.args)
    if not parsed:
        await update.effective_message.reply_text(
            "Formato: /alertas_add <tipo> <op> <valor>\n"
            "  tipos: blue, mep, ccl, riesgo, inflacion, reservas\n"
            "  op: > o <\n"
            "Ej.: /alertas_add blue > 1350",
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
        await update.effective_message.reply_text(f"Eliminadas {before-after} alertas de tipo {tipo.upper()}.", parse_mode=ParseMode.HTML, link_preview_options=LinkPreviewOptions(is_disabled=True))
    else:
        ALERTS[chat_id] = []
        await update.effective_message.reply_text("Todas las alertas fueron eliminadas.", parse_mode=ParseMode.HTML, link_preview_options=LinkPreviewOptions(is_disabled=True))

# ------------------------------ AIOHTTP + Webhook --------------------------

async def keepalive_loop(app: Application):
    await asyncio.sleep(5)
    url = f"{BASE_URL}/"
    timeout = ClientTimeout(total=6)
    async with ClientSession(timeout=timeout) as session:
        while True:
            try:
                async with session.get(url) as resp:
                    log.info("Keepalive %s -> %s", url, resp.status)
            except Exception as e:
                log.warning("Keepalive error: %s", e)
            await asyncio.sleep(300)

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

# ------------------------------ PTB App ------------------------------------

defaults = Defaults(parse_mode=ParseMode.HTML, link_preview_options=LinkPreviewOptions(is_disabled=True), tzinfo=TZ)
application = Application.builder().token(TELEGRAM_TOKEN).defaults(defaults).updater(None).build()

# Handlers (¡sin /start!)
application.add_handler(CommandHandler("dolar", cmd_dolar))
application.add_handler(CommandHandler("acciones", cmd_acciones))
application.add_handler(CommandHandler("cedears", cmd_cedears))
application.add_handler(CommandHandler("rankings", cmd_rankings))
application.add_handler(CommandHandler("rankings_acciones", cmd_rankings_acciones))
application.add_handler(CommandHandler("rankings_cedears", cmd_rankings_cedears))
application.add_handler(CommandHandler("reservas", cmd_reservas))
application.add_handler(CommandHandler("inflacion", cmd_inflacion))
application.add_handler(CommandHandler("riesgo", cmd_riesgo))
application.add_handler(CommandHandler("resumen_diario", cmd_resumen_diario))
application.add_handler(CommandHandler("alertas", cmd_alertas_list))
application.add_handler(CommandHandler("alertas_add", cmd_alertas_add))
application.add_handler(CommandHandler("alertas_clear", cmd_alertas_clear))

# ------------------------------ Main ---------------------------------------

if __name__ == "__main__":
    log.info("Iniciando bot Económico AR (Render webhook)")
    app = build_web_app()
    web.run_app(app, host="0.0.0.0", port=PORT)
