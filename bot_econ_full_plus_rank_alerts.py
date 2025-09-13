# bot_econ_full_plus_rank_alerts.py
# -*- coding: utf-8 -*-
#
# Bot Económico AR (Render webhook, sin polling)
#
# Comandos:
#   /dolar                 Tipos de Cambio
#   /acciones              Top 3 Acciones (6m, 3m, 1m)
#   /cedears               Top 3 Cedears (6m, 3m, 1m)
#   /rankings_acciones     Top 5 Acciones (Proy. 6–12M)
#   /rankings_cedears      Top 5 Cedears (Proy. 6–12M)
#   /reservas              Reservas BCRA
#   /inflacion             Inflación Mensual
#   /riesgo                Riesgo País
#   /resumen_diario        Resumen Diario + Noticias (5 links)
#   /alertas               Listar Alertas
#   /alertas_add           Agregar Alerta (menú con flechas, %, “Volver”)
#   /alertas_clear         Borrar Alertas (todas o por tipo/ticker)
#
import os, asyncio, logging, re, html as _html
from time import time
from math import sqrt
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, List, Tuple, Any, Optional
from urllib.parse import urlparse

from aiohttp import web, ClientSession, ClientTimeout
from telegram import (
    Update, LinkPreviewOptions, BotCommand,
    InlineKeyboardMarkup, InlineKeyboardButton,
)
from telegram.constants import ParseMode
from telegram.ext import (
    Application, CommandHandler, ContextTypes, Defaults,
    CallbackQueryHandler, MessageHandler, ConversationHandler, filters
)

# ---------------- Config ----------------
TZ = ZoneInfo("America/Argentina/Buenos_Aires")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "tgwebhook").strip().strip("/")
PORT = int(os.getenv("PORT", "10000"))
BASE_URL = os.getenv("BASE_URL", os.getenv("RENDER_EXTERNAL_URL", "https://bot-economico-ar.onrender.com")).rstrip("/")
if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN no configurado.")
WEBHOOK_PATH = f"/{WEBHOOK_SECRET}"
WEBHOOK_URL = f"{BASE_URL}{WEBHOOK_PATH}"

# APIs/Fuentes
CRYPTOYA_DOLAR_URL = "https://criptoya.com/api/dolar"
DOLARAPI_BASE = "https://dolarapi.com/v1"
ARG_DATOS_BASES = [
    "https://api.argentinadatos.com/v1/finanzas/indices",
    "https://argentinadatos.com/v1/finanzas/indices",
]
LAMACRO_RESERVAS_URL = "https://www.lamacro.ar/variables/1"

YF_URLS = [
    "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
    "https://query2.finance.yahoo.com/v8/finance/chart/{symbol}",
]
YF_HEADERS = {"User-Agent": "Mozilla/5.0"}

# RSS nacionales
RSS_FEEDS = [
    "https://www.ambito.com/contenidos/economia.xml",
    "https://www.iprofesional.com/rss",
    "https://www.infobae.com/economia/rss",
    "https://www.perfil.com/rss/economia.xml",
    "https://www.baenegocios.com/rss/economia.xml",
    "https://www.telam.com.ar/rss2/economia.xml",
    "https://www.cronista.com/files/rss/economia.xml",
    "https://www.cronista.com/files/rss/finanzas-mercados.xml",
    "https://www.clarin.com/rss/economia/",
    "https://www.lanacion.com.ar/economia/rss/",
    "https://www.pagina12.com.ar/rss/secciones/economia/notas",
]

ACCIONES_BA = ["GGAL.BA","YPFD.BA","PAMP.BA","CEPU.BA","ALUA.BA","TXAR.BA","TGSU2.BA","BYMA.BA","SUPV.BA","BMA.BA"]
CEDEARS_BA  = ["AAPL.BA","MSFT.BA","NVDA.BA","AMZN.BA","GOOGL.BA","TSLA.BA","META.BA","JNJ.BA","KO.BA","NFLX.BA"]

# ------------- Logging -------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("bot-econ-ar")

# ------------- Utils -------------
def fmt_number(n: Optional[float], nd=2) -> str:
    try:
        if n is None: return "—"
        s = f"{n:,.{nd}f}"
        return s.replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return str(n)

def fmt_money_ars(n: Optional[float]) -> str:
    return f"$ {fmt_number(n, 2)}"

def pct(n: Optional[float], nd: int = 2) -> str:
    try:
        return f"{n:+.{nd}f}%".replace(".", ",")
    except Exception:
        return "—"

def anchor(href: str, text: str) -> str:
    return f'<a href="{_html.escape(href, quote=True)}">{_html.escape(text)}</a>'

def html_op(op: str) -> str:
    return "&gt;" if op == ">" else "&lt;"

def fmt_fecha_ddmmyyyy_from_iso(s: Optional[str]) -> Optional[str]:
    if not s: return None
    try:
        if re.match(r"^\d{4}-\d{2}-\d{2}", s):
            return datetime.strptime(s[:10], "%Y-%m-%d").strftime("%d/%m/%Y")
    except Exception:
        pass
    return s

def last_day_of_month_str(periodo_yyyy_mm: str) -> Optional[str]:
    try:
        y = int(periodo_yyyy_mm[0:4]); m = int(periodo_yyyy_mm[5:7])
        d = (datetime(y, 12, 31) if m==12 else datetime(y, m+1, 1) - timedelta(days=1))
        return d.strftime("%d/%m/%Y")
    except Exception:
        return None

def parse_period_to_ddmmyyyy(per: Optional[str]) -> Optional[str]:
    if not per: return None
    if re.match(r"^\d{4}-\d{2}-\d{2}$", per): return fmt_fecha_ddmmyyyy_from_iso(per)
    if re.match(r"^\d{4}-\d{2}$", per): return last_day_of_month_str(per)
    return per

# ------------- HTTP helpers -------------
REQ_HEADERS = {"User-Agent":"Mozilla/5.0", "Accept":"*/*"}

async def fetch_json(session: ClientSession, url: str, **kwargs) -> Optional[Dict[str, Any]]:
    try:
        timeout = kwargs.pop("timeout", ClientTimeout(total=15))
        headers = kwargs.pop("headers", {})
        async with session.get(url, timeout=timeout, headers={**REQ_HEADERS, **headers}, **kwargs) as resp:
            if resp.status == 200:
                return await resp.json(content_type=None)
            log.warning("GET %s -> %s", url, resp.status)
    except Exception as e:
        log.warning("fetch_json error %s: %s", url, e)
    return None

async def fetch_text(session: ClientSession, url: str, **kwargs) -> Optional[str]:
    try:
        timeout = kwargs.pop("timeout", ClientTimeout(total=15))
        headers = kwargs.pop("headers", {})
        async with session.get(url, timeout=timeout, headers={**REQ_HEADERS, **headers}, **kwargs) as resp:
            if resp.status == 200:
                return await resp.text()
            log.warning("GET %s -> %s", url, resp.status)
    except Exception as e:
        log.warning("fetch_text error %s: %s", url, e)
    return None

# ------------- Dólares -------------
async def get_dolares(session: ClientSession) -> Dict[str, Dict[str, Any]]:
    data: Dict[str, Dict[str, Any]] = {}

    # CriptoYa (varios tipos)
    cj = await fetch_json(session, CRYPTOYA_DOLAR_URL)
    if cj:
        def _safe(block: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
            if not isinstance(block, dict): return (None, None)
            c, v = block.get("compra") or block.get("buy"), block.get("venta") or block.get("sell")
            try: return (float(c) if c is not None else None, float(v) if v is not None else None)
            except Exception: return (None, None)
        for k in ["oficial","mayorista","blue","mep","ccl","cripto","tarjeta"]:
            c,v = _safe(cj.get(k,{}))
            if c is not None or v is not None:
                data[k] = {"compra": c, "venta": v, "fuente": "CriptoYa"}

    # DolarAPI (complemento / fecha)
    async def dolarapi(path: str):
        j = await fetch_json(session, f"{DOLARAPI_BASE}{path}")
        if not j: return (None, None, None)
        c,v,fecha = j.get("compra"), j.get("venta"), j.get("fechaActualizacion") or j.get("fecha")
        try: return (float(c) if c is not None else None, float(v) if v is not None else None, fecha)
        except Exception: return (None, None, fecha)

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

def extract_latest_dolar_date(d: Dict[str, Dict[str, Any]]) -> Optional[str]:
    dates = []
    for row in d.values():
        f = row.get("fecha")
        if f:
            dd = fmt_fecha_ddmmyyyy_from_iso(f)
            if dd: dates.append(dd)
    if not dates: return None
    try:
        dates.sort(key=lambda s: datetime.strptime(s, "%d/%m/%Y"))
        return dates[-1]
    except Exception:
        return dates[-1]

# ------------- ArgentinaDatos -------------
async def arg_datos_get(session: ClientSession, suffix: str) -> Optional[Any]:
    for base in ARG_DATOS_BASES:
        for u in (f"{base}{suffix}", f"{base}{suffix}/"):
            j = await fetch_json(session, u)
            if j: return j
    return None

async def get_riesgo_pais(session: ClientSession) -> Optional[Tuple[int, Optional[str]]]:
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
    j = await arg_datos_get(session, "/inflacion")
    if not j: j = await arg_datos_get(session, "/inflacion/mensual/ultimo")
    if not j: j = await arg_datos_get(session, "/inflacion/mensual")
    if isinstance(j, dict) and "serie" in j and isinstance(j["serie"], list) and j["serie"]:
        j = j["serie"]
    per = None; val = None
    if isinstance(j, list) and j:
        last = j[-1]; val = last.get("valor"); per = last.get("fecha") or last.get("periodo")
    elif isinstance(j, dict):
        val = j.get("valor"); per = j.get("fecha") or j.get("periodo")
    if val is None: return None
    fecha = parse_period_to_ddmmyyyy(per)
    try: return (float(val), fecha)
    except Exception: return None

# ------------- Reservas -------------
async def get_reservas_lamacro(session: ClientSession) -> Optional[Tuple[float, Optional[str]]]:
    html = await fetch_text(session, LAMACRO_RESERVAS_URL)
    if not html: return None
    m_val = re.search(r"(?:Último dato|Valor actual)\s*:\s*([0-9\.\,]+)", html)
    m_date = re.search(r"([0-3]\d/[0-1]\d/\d{4})", html)
    fecha = m_date.group(1) if m_date else None
    if m_val:
        s = m_val.group(1).replace('.', '').replace(',', '.')
        try: return (float(s), fecha)
        except Exception: return None
    return None

# ------------- Yahoo métricas (acciones/cedears) -------------
RET_CACHE_1Y: Dict[str, Tuple[float, Optional[Dict[str, Any]]]] = {}
RET_TTL = 600

async def _yf_chart_1y(session: ClientSession, symbol: str, interval: str) -> Optional[Dict[str, Any]]:
    for base in YF_URLS:
        host = base.split("/")[2]
        cache_key = f"{host}|{symbol}|{interval}"
        now_ts = time()
        if cache_key in RET_CACHE_1Y:
            ts_cache, res_cache = RET_CACHE_1Y[cache_key]
            if now_ts - ts_cache < RET_TTL:
                return res_cache
        params = {"range": "1y", "interval": interval, "events": "div,split"}
        j = await fetch_json(session, base.format(symbol=symbol), headers=YF_HEADERS, params=params)
        try:
            res = j.get("chart", {}).get("result", [])[0]
            RET_CACHE_1Y[cache_key] = (now_ts, res)
            return res
        except Exception:
            RET_CACHE_1Y[cache_key] = (now_ts, None)
            continue
    return None

def _metrics_from_chart(res: Dict[str, Any]) -> Optional[Dict[str, Optional[float]]]:
    try:
        ts = res["timestamp"]
        closes_raw = res["indicators"]["adjclose"][0]["adjclose"]
        pairs = [(t,c) for t,c in zip(ts, closes_raw) if (t is not None and c is not None)]
        if len(pairs) < 30: return None
        ts = [p[0] for p in pairs]; closes = [p[1] for p in pairs]
        idx_last = len(closes) - 1; last = closes[idx_last]; t_last = ts[idx_last]
        def first_on_or_after(tcut):
            for i, t in enumerate(ts):
                if t >= tcut: return closes[i]
            return closes[0]
        t6 = t_last - 180*24*3600; t3 = t_last - 90*24*3600; t1 = t_last - 30*24*3600
        base6 = first_on_or_after(t6); base3 = first_on_or_after(t3); base1 = first_on_or_after(t1)
        ret6 = (last/base6 - 1.0)*100.0 if base6 else None
        ret3 = (last/base3 - 1.0)*100.0 if base3 else None
        ret1 = (last/base1 - 1.0)*100.0 if base1 else None
        # vol anualizada aprox (ventana 60d)
        rets_d = []
        for i in range(1, len(closes)):
            if closes[i-1] and closes[i]: rets_d.append(closes[i]/closes[i-1]-1.0)
        look = 60 if len(rets_d) >= 60 else max(10, len(rets_d)-1)
        vol_ann = None
        if len(rets_d) >= 10:
            mu = sum(rets_d[-look:]) / len(rets_d[-look:])
            var = sum((r-mu)**2 for r in rets_d[-look:])/(len(rets_d[-look:])-1) if len(rets_d[-look:])>1 else 0.0
            sd = sqrt(var); vol_ann = sd*sqrt(252)*100.0
        # drawdown 6m
        idx_cut = next((i for i,t in enumerate(ts) if t >= t6), 0)
        peak = closes[idx_cut]; dd_min = 0.0
        for v in closes[idx_cut:]:
            if v > peak: peak = v
            dd = v/peak - 1.0
            if dd < dd_min: dd_min = dd
        dd6 = abs(dd_min)*100.0 if dd_min < 0 else 0.0
        hi52 = (last/max(closes) - 1.0)*100.0
        # SMA50/SMA200
        def _sma(vals, w):
            out, s, q = [None]*len(vals), 0.0, []
            for i, v in enumerate(vals):
                q.append(v); s += v
                if len(q) > w: s -= q.pop(0)
                if len(q) == w: out[i] = s/w
            return out
        sma50  = _sma(closes, 50); sma200 = _sma(closes, 200)
        s50_last = sma50[idx_last] if idx_last < len(sma50) else None
        s50_prev = sma50[idx_last-20] if idx_last-20 >= 0 else None
        slope50 = ((s50_last/s50_prev - 1.0)*100.0) if (s50_last and s50_prev) else 0.0
        s200_last = sma200[idx_last] if idx_last < len(sma200) else None
        trend_flag = 1 if (s200_last and last > s200_last) else (-1 if s200_last else 0)
        return {"1m": ret1, "3m": ret3, "6m": ret6, "last_ts": int(t_last),
                "vol_ann": vol_ann, "dd6m": dd6, "hi52": hi52, "slope50": slope50,
                "trend_flag": float(trend_flag), "last_px": float(last)}
    except Exception:
        return None

async def _yf_metrics_1y(session: ClientSession, symbol: str) -> Dict[str, Optional[float]]:
    out = {"6m": None, "3m": None, "1m": None, "last_ts": None, "vol_ann": None,
           "dd6m": None, "hi52": None, "slope50": None, "trend_flag": None, "last_px": None}
    for interval in ("1d", "1wk"):
        res = await _yf_chart_1y(session, symbol, interval)
        if res:
            m = _metrics_from_chart(res)
            if m: out.update(m); break
    return out

async def metrics_for_symbols(session: ClientSession, symbols: List[str]) -> Tuple[Dict[str, Dict[str, Optional[float]]], Optional[int]]:
    out = {s: {"6m": None, "3m": None, "1m": None, "last_ts": None, "vol_ann": None,
               "dd6m": None, "hi52": None, "slope50": None, "trend_flag": None, "last_px": None} for s in symbols}
    sem = asyncio.Semaphore(4)  # amable con Yahoo (evita 429)
    async def work(sym: str):
        async with sem:
            out[sym] = await _yf_metrics_1y(session, sym)
    await asyncio.gather(*(work(s) for s in symbols))
    last_ts = None
    for d in out.values():
        ts = d.get("last_ts")
        if ts: last_ts = ts if last_ts is None else max(last_ts, ts)
    return out, last_ts

def _nz(x: Optional[float], fb: float) -> float:
    return float(x) if x is not None else fb

def projection_pct(m: Dict[str, Optional[float]]) -> float:
    ret6 = _nz(m.get("6m"), -100.0); ret3 = _nz(m.get("3m"), -50.0); ret1 = _nz(m.get("1m"), -20.0)
    momentum = 0.5*ret6 + 0.3*ret3 + 0.2*ret1
    hi52   = _nz(m.get("hi52"), 0.0); slope  = _nz(m.get("slope50"), 0.0); trend  = _nz(m.get("trend_flag"), 0.0)
    vol    = _nz(m.get("vol_ann"), 40.0); dd6    = _nz(m.get("dd6m"), 30.0)
    return 0.5*momentum + 0.2*hi52 + 0.2*slope - 0.05*vol - 0.05*dd6 + 3.0*trend

# ------------- Noticias -------------
from xml.etree import ElementTree as ET
KEYWORDS = ["inflación","ipc","índice de precios","devalu","dólar","ccl","mep","blue",
            "bcra","reservas","tasas","pases","fmi","deuda","riesgo país",
            "cepo","importaciones","exportaciones","merval","acciones","bonos","brecha",
            "subsidios","retenciones","tarifas","liquidez","recaudación","déficit"]

def domain_of(url: str) -> str:
    try: return urlparse(url).netloc.lower()
    except Exception: return ""

def _score_title(title: str) -> int:
    t = title.lower(); score = 0
    for kw in KEYWORDS:
        if kw in t: score += 3
    for kw in ("sube","baja","récord","acelera","cae","acuerdo","medida","ley","resolución","emergencia","reperfil","brecha","dólar","inflación"):
        if kw in t: score += 1
    return score

def _parse_feed_entries(xml: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    try: root = ET.fromstring(xml)
    except Exception: return out
    for item in root.findall(".//item"):
        t_el = item.find("title"); l_el = item.find("link")
        t = (t_el.text or "").strip() if (t_el is not None and t_el.text) else None
        l = (l_el.text or "").strip() if (l_el is not None and l_el.text) else None
        if t and l and l.startswith("http"): out.append((t, l))
    for entry in root.findall(".//{*}entry"):
        t_el = entry.find(".//{*}title")
        link_el = entry.find(".//{*}link[@rel='alternate']") or entry.find(".//{*}link")
        t = (t_el.text or "").strip() if (t_el is not None and t_el.text) else None
        l = link_el.get("href").strip() if (link_el is not None and link_el.get("href")) else None
        if (not l) and entry.find(".//{*}id") is not None:
            l = (entry.find(".//{*}id").text or "").strip()
        if t and l and l.startswith("http"): out.append((t, l))
    if not out:
        for m in re.finditer(r"<title>(.*?)</title>.*?<link>(https?://[^<]+)</link>", xml, flags=re.S|re.I):
            t = re.sub(r"<.*?>", "", m.group(1)).strip(); l = m.group(2).strip()
            if t and l: out.append((t, l))
    return out

async def fetch_rss_entries(session: ClientSession, limit: int = 5) -> List[Tuple[str, str]]:
    entries: List[Tuple[str, str]] = []
    for url in RSS_FEEDS:
        xml = await fetch_text(session, url, headers={"Accept":"application/rss+xml, application/atom+xml, */*"})
        if not xml: continue
        try: entries.extend(_parse_feed_entries(xml))
        except Exception as e: log.warning("RSS parse %s: %s", url, e)
    uniq: Dict[str, str] = {}
    for t, l in entries:
        if l not in uniq and l.startswith("http"): uniq[l] = t
    if not uniq:
        return []
    scored = sorted([(t,l,_score_title(t), domain_of(l)) for l,t in uniq.items()],
                    key=lambda x: x[2], reverse=True)
    picked: List[Tuple[str,str]] = []
    for t,l,_,_dom in scored:
        picked.append((t,l))
        if len(picked) >= limit: break
    return picked[:limit]

def format_news_block(news: List[Tuple[str, str]]) -> str:
    if not news:
        return "<b>📰 Noticias</b>\n—"
    body = "\n\n".join([f"{i}. {anchor(l, t)}" for i,(t,l) in enumerate(news, 1)])  # espaciado extra
    return "<b>📰 Noticias</b>\n" + body

# ------------- Alertas (memoria) -------------
ALERTS: Dict[int, List[Dict[str, Any]]] = {}

def _parse_float_user(s: str) -> Optional[float]:
    s = s.strip()
    s = s.replace("pb","").replace("%","").replace("MUS$","").replace("mus$","").replace("$","")
    s = s.replace(".", "").replace(",", ".")
    try: return float(s)
    except Exception: return None

# ------------- Formatos de salida -------------
def format_dolar_message(d: Dict[str, Dict[str, Any]]) -> str:
    fecha = extract_latest_dolar_date(d)
    header = "<b>💵 Dólares</b>" + (f"  <i>Actualizado: {fecha}</i>" if fecha else "")
    lines = [header, "<pre>Tipo          Venta         Compra</pre>"]
    rows = []
    order = [("oficial","Oficial"),("mayorista","Mayorista"),("blue","Blue"),("mep","MEP"),("ccl","CCL"),("cripto","Cripto"),("tarjeta","Tarjeta")]
    for k, label in order:
        row = d.get(k)
        if not row: continue
        venta_val  = row.get("compra")   # swap intencional (pedido del usuario)
        compra_val = row.get("venta")
        venta  = fmt_money_ars(venta_val)  if venta_val  is not None else "—"
        compra = fmt_money_ars(compra_val) if compra_val is not None else "—"
        l = f"{label:<12}{venta:>12}    {compra:>12}"
        rows.append(f"<pre>{l}</pre>")
    rows.append("<i>Fuentes: CriptoYa + DolarAPI</i>")
    return "\n".join([lines[0], lines[1]] + rows)

def format_top3_single_table(title: str, fecha: Optional[str], rows_syms: List[str],
                             retmap: Dict[str, Dict[str, Optional[float]]]) -> str:
    head = f"<b>{title}</b>" + (f"  <i>Últ. Dato: {fecha}</i>" if fecha else "")
    lines = [head, "<pre>Rank  Ticker         1M         3M         6M</pre>"]
    rows = []
    for idx, sym in enumerate(rows_syms[:3], start=1):
        d = retmap.get(sym, {})
        p1 = pct(d.get("1m"), 2) if d.get("1m") is not None else "—"
        p3 = pct(d.get("3m"), 2) if d.get("3m") is not None else "—"
        p6 = pct(d.get("6m"), 2) if d.get("6m") is not None else "—"
        l = f"{idx:<4} {sym:<12}{p1:>10}{p3:>11}{p6:>11}"
        rows.append(f"<pre>{l}</pre>")
    if not rows: rows.append("<pre>—</pre>")
    return "\n".join([lines[0], lines[1]] + rows)

def format_ranking_projection_table(title: str, fecha: Optional[str], rows: List[Tuple[str, float]]) -> str:
    head = f"<b>{title}</b>" + (f"  <i>Últ. Dato: {fecha}</i>" if fecha else "")
    sub  = "<i>Proy. 6–12M (momentum + tendencia − riesgo)</i>"
    lines = [head, sub, "<pre>Rank  Ticker            Proy. 6–12M</pre>"]
    out_rows = []
    if not rows:
        out_rows.append("<pre>—</pre>")
    else:
        for idx, (sym, proj) in enumerate(rows[:5], start=1):
            p = pct(proj, 1) if proj is not None else "—"
            l = f"{idx:<4} {sym:<14}{p:>12}"
            out_rows.append(f"<pre>{l}</pre>")
    return "\n".join(lines + out_rows)

# ------------- Handlers de datos -------------
async def cmd_dolar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    async with ClientSession() as session:
        data = await get_dolares(session)
    msg = format_dolar_message(data) if data else "No pude obtener cotizaciones ahora."
    await update.effective_message.reply_text(msg, parse_mode=ParseMode.HTML, link_preview_options=LinkPreviewOptions(is_disabled=True))

async def cmd_acciones(update: Update, context: ContextTypes.DEFAULT_TYPE):
    async with ClientSession() as session:
        try: mets, last_ts = await asyncio.wait_for(metrics_for_symbols(session, ACCIONES_BA), timeout=30)
        except asyncio.TimeoutError: mets, last_ts = ({s: {"6m": None, "3m": None, "1m": None} for s in ACCIONES_BA}, None)
    fecha = datetime.fromtimestamp(last_ts, TZ).strftime("%d/%m/%Y") if last_ts else None
    pairs = sorted([(sym, m["6m"]) for sym,m in mets.items() if m.get("6m") is not None], key=lambda x: x[1], reverse=True)
    top_syms = [sym for sym,_ in pairs[:3]]
    msg = format_top3_single_table("📈 Top 3 Acciones (BYMA .BA)", fecha, top_syms, mets)
    await update.effective_message.reply_text(msg, parse_mode=ParseMode.HTML, link_preview_options=LinkPreviewOptions(is_disabled=True))

async def cmd_cedears(update: Update, context: ContextTypes.DEFAULT_TYPE):
    async with ClientSession() as session:
        try: mets, last_ts = await asyncio.wait_for(metrics_for_symbols(session, CEDEARS_BA), timeout=30)
        except asyncio.TimeoutError: mets, last_ts = ({s: {"6m": None, "3m": None, "1m": None} for s in CEDEARS_BA}, None)
    fecha = datetime.fromtimestamp(last_ts, TZ).strftime("%d/%m/%Y") if last_ts else None
    pairs = sorted([(sym, m["6m"]) for sym,m in mets.items() if m.get("6m") is not None], key=lambda x: x[1], reverse=True)
    top_syms = [sym for sym,_ in pairs[:3]]
    msg = format_top3_single_table("🌎 Top 3 Cedears (.BA)", fecha, top_syms, mets)
    await update.effective_message.reply_text(msg, parse_mode=ParseMode.HTML, link_preview_options=LinkPreviewOptions(is_disabled=True))

def rank_projection_rows(metmap: Dict[str, Dict[str, Optional[float]]], n=5) -> List[Tuple[str, float]]:
    pairs = []
    for sym, m in metmap.items():
        if m.get("6m") is None: continue
        pairs.append((sym, projection_pct(m)))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:n]

async def cmd_rankings_acciones(update: Update, context: ContextTypes.DEFAULT_TYPE):
    async with ClientSession() as session:
        try: metmap, last_ts = await asyncio.wait_for(metrics_for_symbols(session, ACCIONES_BA), timeout=30)
        except asyncio.TimeoutError: metmap, last_ts = ({}, None)
    fecha = datetime.fromtimestamp(last_ts, TZ).strftime("%d/%m/%Y") if last_ts else None
    rows = rank_projection_rows(metmap, 5) if metmap else []
    msg = format_ranking_projection_table("🏁 Top 5 Acciones (Proyección 6–12M)", fecha, rows)
    await update.effective_message.reply_text(msg, parse_mode=ParseMode.HTML, link_preview_options=LinkPreviewOptions(is_disabled=True))

async def cmd_rankings_cedears(update: Update, context: ContextTypes.DEFAULT_TYPE):
    async with ClientSession() as session:
        try: metmap, last_ts = await asyncio.wait_for(metrics_for_symbols(session, CEDEARS_BA), timeout=30)
        except asyncio.TimeoutError: metmap, last_ts = ({}, None)
    fecha = datetime.fromtimestamp(last_ts, TZ).strftime("%d/%m/%Y") if last_ts else None
    rows = rank_projection_rows(metmap, 5) if metmap else []
    msg = format_ranking_projection_table("🏁 Top 5 Cedears (Proyección 6–12M)", fecha, rows)
    await update.effective_message.reply_text(msg, parse_mode=ParseMode.HTML, link_preview_options=LinkPreviewOptions(is_disabled=True))

async def cmd_reservas(update: Update, context: ContextTypes.DEFAULT_TYPE):
    async with ClientSession() as session:
        res = await get_reservas_lamacro(session)
    if not res:
        txt = "No pude obtener reservas ahora."
    else:
        val, fecha = res
        txt = (f"<b>🏦 Reservas BCRA</b>{f'  <i>Últ. Act.: {fecha}</i>' if fecha else ''}\n"
               f"<b>{fmt_number(val,0)} MUS$</b> (MUS$ = millones de USD)\n"
               f"<i>Fuente: LaMacro</i>")
    await update.effective_message.reply_text(txt, parse_mode=ParseMode.HTML, link_preview_options=LinkPreviewOptions(is_disabled=True))

async def cmd_inflacion(update: Update, context: ContextTypes.DEFAULT_TYPE):
    async with ClientSession() as session:
        tup = await get_inflacion_mensual(session)
    if tup is None:
        txt = "No pude obtener inflación ahora."
    else:
        val, fecha = tup
        val_str = str(round(val, 1)).replace(".", ",")
        txt = f"<b>📉 Inflación Mensual</b>{f'  <i>{fecha}</i>' if fecha else ''}\n<b>{val_str}%</b>\n<i>Fuente: ArgentinaDatos</i>"
    await update.effective_message.reply_text(txt, parse_mode=ParseMode.HTML, link_preview_options=LinkPreviewOptions(is_disabled=True))

async def cmd_riesgo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    async with ClientSession() as session:
        tup = await get_riesgo_pais(session)
    if tup is None:
        txt = "No pude obtener riesgo país ahora."
    else:
        rp, f = tup; f_str = fmt_fecha_ddmmyyyy_from_iso(f)
        txt = f"<b>📈 Riesgo País</b>{f'  <i>{f_str}</i>' if f_str else ''}\n<b>{rp} pb</b>\n<i>Fuente: ArgentinaDatos</i>"
    await update.effective_message.reply_text(txt, parse_mode=ParseMode.HTML, link_preview_options=LinkPreviewOptions(is_disabled=True))

async def cmd_resumen_diario(update: Update, context: ContextTypes.DEFAULT_TYPE):
    async with ClientSession() as session:
        dolares  = await get_dolares(session)
        riesgo_t = await get_riesgo_pais(session)
        reservas = await get_reservas_lamacro(session)
        inflac_t = await get_inflacion_mensual(session)
        news     = await fetch_rss_entries(session, limit=5)

    blocks = [f"<b>🗞️ Resumen Diario</b>"]
    if dolares: blocks.append(format_dolar_message(dolares))
    if riesgo_t:
        rp, f = riesgo_t; f_str = fmt_fecha_ddmmyyyy_from_iso(f)
        blocks.append(f"<b>📈 Riesgo País</b>{f'  <i>{f_str}</i>' if f_str else ''}\n<b>{rp} pb</b>\n<i>Fuente: ArgentinaDatos</i>")
    if reservas:
        rv, rf = reservas
        blocks.append(f"<b>🏦 Reservas BCRA</b>{f'  <i>Últ. Act.: {rf}</i>' if rf else ''}\n<b>{fmt_number(rv,0)} MUS$</b>\n<i>Fuente: LaMacro</i>")
    if inflac_t:
        iv, ip = inflac_t; iv_str = str(round(iv,1)).replace(".", ",")
        blocks.append(f"<b>📉 Inflación Mensual</b>{f'  <i>{ip}</i>' if ip else ''}\n<b>{iv_str}%</b>\n<i>Fuente: ArgentinaDatos</i>")

    try:
        news_block = format_news_block(news or [])
    except Exception as e:
        log.warning("format news error: %s", e)
        news_block = "<b>📰 Noticias</b>\n—"
    blocks.append(news_block)

    await update.effective_message.reply_text("\n\n".join(blocks), parse_mode=ParseMode.HTML, link_preview_options=LinkPreviewOptions(is_disabled=True))

# ------------- Alertas (memoria) -------------
ALERTS: Dict[int, List[Dict[str, Any]]] = ALERTS  # alias

# Estados conversación
AL_KIND, AL_FX_TYPE, AL_FX_SIDE, AL_OP, AL_MODE, AL_VALUE, AL_METRIC_TYPE, AL_TICKER, AL_PERIOD = range(9)

def kb(rows: List[List[Tuple[str,str]]]) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton(text, callback_data=data) for text, data in r] for r in rows])

def kb_tickers(symbols: List[str], back_target: str) -> InlineKeyboardMarkup:
    rows: List[List[Tuple[str,str]]] = []
    row: List[Tuple[str,str]] = []
    for s in symbols:
        row.append((s, f"TICK:{s}"))
        if len(row) == 3:
            rows.append(row); row = []
    if row: rows.append(row)
    rows.append([("Volver","BACK:"+back_target), ("Cancelar","CANCEL")])
    return kb(rows)

# ----- Listar / Clear -----
async def cmd_alertas_list(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    rules = ALERTS.get(chat_id, [])
    if not rules:
        txt = ("No tenés alertas configuradas.\n"
               "Usá /alertas_add para crear una con el menú.")
    else:
        lines = ["<b>🔔 Alertas Configuradas</b>"]
        for r in rules:
            if r.get("kind") == "fx":
                t, side, op, v = r["type"], r["side"], r["op"], r["value"]
                lines.append(f"• {t.upper()} ({side}) {html_op(op)} {fmt_money_ars(v)}")
            elif r.get("kind") == "metric":
                t, op, v = r["type"], r["op"], r["value"]
                if t=="riesgo": val = f"{v:.0f} pb"
                elif t=="reservas": val = f"{fmt_number(v,0)} MUS$"
                else: val = f"{str(round(v,1)).replace('.',',')}%"
                lines.append(f"• {t.upper()} {html_op(op)} {val}")
            else:
                sym, per, op, v = r["symbol"], r["period"], r["op"], r["value"]
                lines.append(f"• {sym} ({per.upper()}) {html_op(op)} {pct(v,1)}")
        lines.append("\nPara borrar: /alertas_clear [tipo|TICKER.BA] (sin parámetro borra todas)")
        txt = "\n".join(lines)
    await update.effective_message.reply_text(txt, parse_mode=ParseMode.HTML, link_preview_options=LinkPreviewOptions(is_disabled=True))

async def cmd_alertas_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    arg = (context.args[0].strip() if context.args else None)
    if not ALERTS.get(chat_id):
        await update.effective_message.reply_text("No había alertas guardadas.", parse_mode=ParseMode.HTML, link_preview_options=LinkPreviewOptions(is_disabled=True))
        return
    if not arg:
        cnt = len(ALERTS[chat_id]); ALERTS[chat_id] = []
        await update.effective_message.reply_text(f"Se eliminaron {cnt} alertas.", parse_mode=ParseMode.HTML, link_preview_options=LinkPreviewOptions(is_disabled=True))
        return
    arg_low = arg.lower(); before = len(ALERTS[chat_id])
    fx_types = {"oficial","mayorista","blue","mep","ccl","tarjeta","cripto"}
    if arg_low in fx_types:
        ALERTS[chat_id] = [r for r in ALERTS[chat_id] if not (r.get("kind")=="fx" and r.get("type")==arg_low)]
    elif arg_low in {"riesgo","inflacion","reservas"}:
        ALERTS[chat_id] = [r for r in ALERTS[chat_id] if not (r.get("kind")=="metric" and r.get("type")==arg_low)]
    elif re.match(r"^[A-Za-z0-9]+\.BA$", arg, flags=re.I):
        symu = arg.upper()
        ALERTS[chat_id] = [r for r in ALERTS[chat_id] if not (r.get("kind")=="ticker" and r.get("symbol")==symu)]
    else:
        await update.effective_message.reply_text("Indicá un tipo (blue/mep/ccl/oficial/mayorista/tarjeta/cripto/riesgo/inflacion/reservas) o un TICKER.BA. Sin parámetro borra todas.", parse_mode=ParseMode.HTML, link_preview_options=LinkPreviewOptions(is_disabled=True))
        return
    after = len(ALERTS[chat_id])
    await update.effective_message.reply_text(f"Eliminadas {before-after} alertas.", parse_mode=ParseMode.HTML, link_preview_options=LinkPreviewOptions(is_disabled=True))

# ----- Conversación /alertas_add (con “Volver” y valor actual) -----
async def alertas_add_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["al"] = {}
    k = kb([
        [("Dólares", "KIND:fx"), ("Macroeconomía", "KIND:metric")],
        [("Acciones", "KIND:acciones"), ("Cedears", "KIND:cedears")],
        [("Cancelar", "CANCEL")]
    ])
    await update.effective_message.reply_text("¿Qué querés alertar?", reply_markup=k)
    return AL_KIND

async def alertas_back(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    target = q.data.split(":",1)[1]
    al = context.user_data.get("al", {})
    if target == "KIND":
        k = kb([
            [("Dólares", "KIND:fx"), ("Macroeconomía", "KIND:metric")],
            [("Acciones", "KIND:acciones"), ("Cedears", "KIND:cedears")],
            [("Cancelar", "CANCEL")]
        ])
        await q.edit_message_text("¿Qué querés alertar?", reply_markup=k)
        return AL_KIND
    if target == "FXTYPE":
        k = kb([
            [("Oficial","FXTYPE:oficial"),("Mayorista","FXTYPE:mayorista")],
            [("Blue","FXTYPE:blue"),("MEP","FXTYPE:mep"),("CCL","FXTYPE:ccl")],
            [("Tarjeta","FXTYPE:tarjeta"),("Cripto","FXTYPE:cripto")],
            [("Volver","BACK:KIND"),("Cancelar","CANCEL")]
        ])
        await q.edit_message_text("Elegí el tipo de dólar:", reply_markup=k)
        return AL_FX_TYPE
    if target == "FXSIDE":
        t = al.get("type","?")
        k = kb([
            [("Venta","SIDE:venta"),("Compra","SIDE:compra")],
            [("Volver","BACK:FXTYPE"),("Cancelar","CANCEL")]
        ])
        await q.edit_message_text(f"Tipo: {t.upper()}\nElegí lado:", reply_markup=k)
        return AL_FX_SIDE
    if target == "METRIC":
        k = kb([
            [("Riesgo País","METRIC:riesgo")],
            [("Inflación Mensual","METRIC:inflacion")],
            [("Reservas BCRA","METRIC:reservas")],
            [("Volver","BACK:KIND"),("Cancelar","CANCEL")]
        ])
        await q.edit_message_text("Elegí la métrica:", reply_markup=k)
        return AL_METRIC_TYPE
    if target == "TICKERS_ACC":
        await q.edit_message_text("Elegí el ticker (Acciones .BA):", reply_markup=kb_tickers(ACCIONES_BA, "KIND"))
        return AL_TICKER
    if target == "TICKERS_CEDEARS":
        await q.edit_message_text("Elegí el ticker (Cedears .BA):", reply_markup=kb_tickers(CEDEARS_BA, "KIND"))
        return AL_TICKER
    if target == "PERIOD":
        per_kb = kb([
            [("1m", "PERIOD:1m"), ("3m", "PERIOD:3m"), ("6m", "PERIOD:6m")],
            [("Volver","BACK:" + ("TICKERS_ACC" if al.get("segment")=="acciones" else "TICKERS_CEDEARS")),
             ("Cancelar","CANCEL")]
        ])
        await q.edit_message_text(f"Ticker: {al.get('symbol','?')}\nElegí período:", reply_markup=per_kb)
        return AL_PERIOD
    if target == "OP":
        kind = al.get("kind")
        if kind == "ticker":
            per = al.get("period","?")
            kb_op = kb([
                [("↑ Sube", "OP:>"), ("↓ Baja", "OP:<")],
                [("Volver","BACK:PERIOD"),("Cancelar","CANCEL")]
            ])
            await q.edit_message_text(f"Período: {per}\nElegí condición:", reply_markup=kb_op)
        elif kind == "fx":
            kb_op = kb([
                [("↑ Sube", "OP:>"), ("↓ Baja", "OP:<")],
                [("Volver","BACK:FXSIDE"),("Cancelar","CANCEL")]
            ])
            await q.edit_message_text("Elegí condición:", reply_markup=kb_op)
        else:  # metric
            kb_op = kb([
                [("↑ Sube", "OP:>"), ("↓ Baja", "OP:<")],
                [("Volver","BACK:METRIC"),("Cancelar","CANCEL")]
            ])
            await q.edit_message_text("Elegí condición:", reply_markup=kb_op)
        return AL_OP
    if target == "MODE":
        kb_mode = kb([
            [("Ingresar Monto", "MODE:absolute")],
            [("Ingresar % vs Valor Actual", "MODE:percent")],
            [("Volver","BACK:OP"),("Cancelar","CANCEL")]
        ])
        await q.edit_message_text("¿Cómo querés definir el umbral?", reply_markup=kb_mode)
        return AL_MODE
    await q.edit_message_text("Operación cancelada.")
    return ConversationHandler.END

async def alertas_add_kind(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    data = q.data
    if data == "CANCEL":
        await q.edit_message_text("Operación cancelada.")
        return ConversationHandler.END
    kind = data.split(":",1)[1]
    context.user_data["al"] = {}
    al = context.user_data["al"]
    if kind == "fx":
        k = kb([
            [("Oficial","FXTYPE:oficial"),("Mayorista","FXTYPE:mayorista")],
            [("Blue","FXTYPE:blue"),("MEP","FXTYPE:mep"),("CCL","FXTYPE:ccl")],
            [("Tarjeta","FXTYPE:tarjeta"),("Cripto","FXTYPE:cripto")],
            [("Volver","BACK:KIND"),("Cancelar","CANCEL")]
        ])
        al["kind"] = "fx"
        await q.edit_message_text("Elegí el tipo de dólar:", reply_markup=k)
        return AL_FX_TYPE
    if kind == "metric":
        k = kb([
            [("Riesgo País","METRIC:riesgo")],
            [("Inflación Mensual","METRIC:inflacion")],
            [("Reservas BCRA","METRIC:reservas")],
            [("Volver","BACK:KIND"),("Cancelar","CANCEL")]
        ])
        al["kind"] = "metric"
        await q.edit_message_text("Elegí la métrica:", reply_markup=k)
        return AL_METRIC_TYPE
    if kind == "acciones":
        al["kind"] = "ticker"; al["segment"] = "acciones"
        await q.edit_message_text("Elegí el ticker (Acciones .BA):", reply_markup=kb_tickers(ACCIONES_BA, "KIND"))
        return AL_TICKER
    if kind == "cedears":
        al["kind"] = "ticker"; al["segment"] = "cedears"
        await q.edit_message_text("Elegí el ticker (Cedears .BA):", reply_markup=kb_tickers(CEDEARS_BA, "KIND"))
        return AL_TICKER
    await q.edit_message_text("Operación cancelada.")
    return ConversationHandler.END

async def alertas_add_fx_type(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    if q.data == "CANCEL": await q.edit_message_text("Operación cancelada."); return ConversationHandler.END
    if q.data.startswith("BACK:"): return await alertas_back(update, context)
    t = q.data.split(":",1)[1]
    context.user_data["al"]["type"] = t
    k = kb([
        [("Venta","SIDE:venta"),("Compra","SIDE:compra")],
        [("Volver","BACK:FXTYPE"),("Cancelar","CANCEL")]
    ])
    await q.edit_message_text(f"Tipo: {t.upper()}\nElegí lado:", reply_markup=k)
    return AL_FX_SIDE

async def alertas_add_fx_side(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    if q.data == "CANCEL": await q.edit_message_text("Operación cancelada."); return ConversationHandler.END
    if q.data.startswith("BACK:"): return await alertas_back(update, context)
    side = q.data.split(":",1)[1]
    context.user_data["al"]["side"] = side
    k = kb([
        [("↑ Sube", "OP:>"), ("↓ Baja", "OP:<")],
        [("Volver","BACK:FXSIDE"),("Cancelar","CANCEL")]
    ])
    await q.edit_message_text(f"Lado: {side}\nElegí condición:", reply_markup=k)
    return AL_OP

async def alertas_add_metric_type(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    if q.data == "CANCEL": await q.edit_message_text("Operación cancelada."); return ConversationHandler.END
    if q.data.startswith("BACK:"): return await alertas_back(update, context)
    m = q.data.split(":",1)[1]
    context.user_data["al"]["type"] = m
    k = kb([
        [("↑ Sube", "OP:>"), ("↓ Baja", "OP:<")],
        [("Volver","BACK:METRIC"),("Cancelar","CANCEL")]
    ])
    await q.edit_message_text(f"Métrica: {m.upper()}\nElegí condición:", reply_markup=k)
    return AL_OP

async def alertas_add_op(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    if q.data == "CANCEL": await q.edit_message_text("Operación cancelada."); return ConversationHandler.END
    if q.data.startswith("BACK:"): return await alertas_back(update, context)
    op = q.data.split(":",1)[1]
    context.user_data["al"]["op"] = op
    kind = context.user_data["al"].get("kind")

    # Si es ticker: mostrar precio actual y rendimiento del período antes de pedir el valor
    if kind == "ticker":
        sym = context.user_data["al"].get("symbol")
        per = context.user_data["al"].get("period")
        if not sym or not per:
            await q.edit_message_text("Faltan datos del ticker. Volvé a intentarlo con /alertas_add.")
            return ConversationHandler.END
        async with ClientSession() as session:
            metmap, _ = await metrics_for_symbols(session, [sym])
            cur = metmap.get(sym, {}).get(per)
            last_px = metmap.get(sym, {}).get("last_px")
        cur_s = pct(cur,1) if cur is not None else "—"
        px_s  = fmt_money_ars(last_px) if last_px is not None else "—"
        await q.edit_message_text(
            f"Ticker: {sym}\nPeríodo: {per}\nCondición: {'↑ Sube' if op=='>' else '↓ Baja'}\n"
            f"Actual: Precio {px_s} | Rendimiento {cur_s}\n\n"
            "Ingresá el valor objetivo en % (puntos). Ej: 12  |  -8.5"
        )
        return AL_VALUE

    # Si es fx o metric: ir a elegir modo (monto / %), y mostrar valor actual al pedir el número (en AL_MODE)
    k = kb([
        [("Ingresar Monto", "MODE:absolute")],
        [("Ingresar % vs Valor Actual", "MODE:percent")],
        [("Volver","BACK:OP"),("Cancelar","CANCEL")]
    ])
    await q.edit_message_text("¿Cómo querés definir el umbral?", reply_markup=k)
    return AL_MODE

async def alertas_add_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    if q.data == "CANCEL": await q.edit_message_text("Operación cancelada."); return ConversationHandler.END
    if q.data.startswith("BACK:"): return await alertas_back(update, context)
    mode = q.data.split(":",1)[1]  # absolute | percent
    context.user_data["al"]["mode"] = mode

    al = context.user_data.get("al", {})
    kind = al.get("kind")
    msg_help = ""

    async with ClientSession() as session:
        if kind == "fx":
            fx = await get_dolares(session); row = fx.get(al.get("type",""), {})
            cur = row.get(al.get("side",""))
            cur_s = fmt_money_ars(cur) if cur is not None else "—"
            if mode == "percent":
                msg_help = (f"Ahora: {al.get('type','?').upper()} ({al.get('side','?')}) = {cur_s}\n"
                            "Ingresá el porcentaje (solo número). Ej: 10  |  7,5\n"
                            "Se calculará sobre el valor actual (p. ej. 10% ↑ suma 10% al valor de hoy).")
            else:
                msg_help = (f"Ahora: {al.get('type','?').upper()} ({al.get('side','?')}) = {cur_s}\n"
                            "Ingresá el monto (solo número). Ej: 1580  |  25500")
        elif kind == "metric":
            rp = await get_riesgo_pais(session)
            infl = await get_inflacion_mensual(session)
            rv  = await get_reservas_lamacro(session)
            vals = {
                "riesgo": (f"{rp[0]:.0f} pb" if rp else "—", rp[0] if rp else None),
                "inflacion": ((str(round(infl[0],1)).replace('.',','))+"%" if infl else "—", infl[0] if infl else None),
                "reservas": (f"{fmt_number(rv[0],0)} MUS$" if rv else "—", rv[0] if rv else None)
            }
            label, cur = vals.get(al.get("type",""), ("—", None))
            if mode == "percent":
                msg_help = (f"Ahora: {al.get('type','?').upper()} = {label}\n"
                            "Ingresá el porcentaje (solo número). Ej: 10  |  7,5\n"
                            "Se calculará sobre el valor actual (p. ej. 10% ↑ suma 10% al valor de hoy).")
            else:
                unidad = "pb" if al.get("type")=="riesgo" else ("MUS$" if al.get("type")=="reservas" else "%")
                msg_help = (f"Ahora: {al.get('type','?').upper()} = {label}\n"
                            f"Ingresá el monto (solo número, en {unidad}). Ej: 25000")
        else:
            msg_help = "Ingresá el valor."

    await q.edit_message_text(msg_help)
    return AL_VALUE

async def alertas_add_ticker_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    if q.data == "CANCEL": await q.edit_message_text("Operación cancelada."); return ConversationHandler.END
    if q.data.startswith("BACK:"): return await alertas_back(update, context)
    sym = q.data.split(":",1)[1].upper()
    context.user_data["al"]["symbol"] = sym
    per_kb = kb([
        [("1m", "PERIOD:1m"), ("3m", "PERIOD:3m"), ("6m", "PERIOD:6m")],
        [("Volver","BACK:" + ("TICKERS_ACC" if context.user_data["al"].get("segment")=="acciones" else "TICKERS_CEDEARS")),
         ("Cancelar","CANCEL")]
    ])
    await q.edit_message_text(f"Ticker: {sym}\nElegí período:", reply_markup=per_kb)
    return AL_PERIOD

async def alertas_add_ticker_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip().upper()
    if not re.match(r"^[A-Z0-9]+\.BA$", text):
        await update.message.reply_text("Formato inválido. Ejemplo: TSLA.BA\nIngresá el TICKER .BA:")
        return AL_TICKER
    context.user_data["al"]["symbol"] = text
    per_kb = kb([
        [("1m", "PERIOD:1m"), ("3m", "PERIOD:3m"), ("6m", "PERIOD:6m")],
        [("Cancelar","CANCEL")]
    ])
    await update.message.reply_text(f"Ticker: {text}\nElegí período:", reply_markup=per_kb)
    return AL_PERIOD

async def alertas_add_period(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    if q.data == "CANCEL": await q.edit_message_text("Operación cancelada."); return ConversationHandler.END
    if q.data.startswith("BACK:"): return await alertas_back(update, context)
    per = q.data.split(":",1)[1]
    context.user_data["al"]["period"] = per
    k = kb([
        [("↑ Sube", "OP:>"), ("↓ Baja", "OP:<")],
        [("Volver","BACK:PERIOD"),("Cancelar","CANCEL")]
    ])
    await q.edit_message_text(f"Período: {per}\nElegí condición:", reply_markup=k)
    return AL_OP

# --- Valor final e inserción de regla ---
def _symbols_from_alerts() -> List[str]:
    syms = set()
    for rules in ALERTS.values():
        for r in rules:
            if r.get("kind") == "ticker" and r.get("symbol"): syms.add(r["symbol"])
    return sorted(syms)

async def alertas_add_value(update: Update, context: ContextTypes.DEFAULT_TYPE):
    val = _parse_float_user(update.message.text or "")
    if val is None:
        await update.message.reply_text("No entendí el número. Probá de nuevo (usá coma o punto).")
        return AL_VALUE

    al = context.user_data.get("al", {})
    chat_id = update.effective_chat.id

    async with ClientSession() as session:
        if al.get("kind") == "fx":
            fx = await get_dolares(session); row = fx.get(al["type"], {})
            cur = row.get(al["side"])
            if cur is None:
                await update.message.reply_text("No pude leer el valor actual. Probá de nuevo más tarde.")
                return ConversationHandler.END
            if al.get("mode") == "percent":
                thr = cur*(1 + (val/100.0)) if al["op"] == ">" else cur*(1 - (val/100.0))
            else:
                thr = val
            rule = {"kind":"fx","type":al["type"],"side":al["side"],"op":al["op"],"value":float(thr)}
            ALERTS.setdefault(chat_id, []).append(rule)
            fb = (f"Ahora: {al['type'].upper()} ({al['side']}) = {fmt_money_ars(cur)}\n"
                  f"Se avisará si {al['type'].upper()} ({al['side']}) "
                  f"{'supera' if al['op']=='>' else 'cae por debajo de'} "
                  f"{fmt_money_ars(thr)}"
                  + (f" (base {fmt_money_ars(cur)} {'+' if al['op']=='>' else '−'} {str(val).replace('.',',')}%)" if al.get("mode")=="percent" else "")
                  )

        elif al.get("kind") == "metric":
            rp = await get_riesgo_pais(session)
            infl = await get_inflacion_mensual(session)
            rv  = await get_reservas_lamacro(session)
            curmap = {
                "riesgo": float(rp[0]) if rp else None,
                "inflacion": float(infl[0]) if infl else None,
                "reservas": rv[0] if rv else None
            }
            cur = curmap.get(al["type"])
            if cur is None:
                await update.message.reply_text("No pude leer el valor actual. Probá de nuevo más tarde.")
                return ConversationHandler.END
            if al.get("mode") == "percent":
                thr = cur*(1 + (val/100.0)) if al["op"] == ">" else cur*(1 - (val/100.0))
            else:
                thr = val
            rule = {"kind":"metric","type":al["type"],"op":al["op"],"value":float(thr)}
            ALERTS.setdefault(chat_id, []).append(rule)
            if al["type"] == "riesgo":
                cur_s = f"{cur:.0f} pb"; thr_s = f"{thr:.0f} pb"
            elif al["type"] == "reservas":
                cur_s = f"{fmt_number(cur,0)} MUS$"; thr_s = f"{fmt_number(thr,0)} MUS$"
            else:
                cur_s = f"{str(round(cur,1)).replace('.',',')}%"; thr_s = f"{str(round(thr,1)).replace('.',',')}%"
            fb = (f"Ahora: {al['type'].upper()} = {cur_s}\n"
                  f"Se avisará si {al['type'].upper()} "
                  f"{'supera' if al['op']=='>' else 'cae por debajo de'} {thr_s}"
                  + (f" (base {cur_s} {'+' if al['op']=='>' else '−'} {str(val).replace('.',',')}%)" if al.get("mode")=="percent" else "")
                  )

        else:  # ticker
            sym, per = al.get("symbol"), al.get("period")
            metmap, _ = await metrics_for_symbols(session, [sym])
            cur = metmap.get(sym, {}).get(per)
            if cur is None:
                await update.message.reply_text("No pude leer el rendimiento actual. Probá más tarde.")
                return ConversationHandler.END
            thr = val  # puntos porcentuales
            rule = {"kind":"ticker","symbol":sym,"period":per,"op":al["op"],"value":float(thr)}
            ALERTS.setdefault(chat_id, []).append(rule)
            cur_s = pct(cur,1); thr_s = pct(thr,1)
            fb = (f"Ahora: {sym} ({per.upper()}) = {cur_s}\n"
                  f"Se avisará si {sym} ({per.upper()}) "
                  f"{'supera' if al['op']=='>' else 'cae por debajo de'} {thr_s}")

    await update.message.reply_text(f"Listo. Alerta agregada ✅\n{fb}", parse_mode=ParseMode.HTML, link_preview_options=LinkPreviewOptions(is_disabled=True))
    return ConversationHandler.END

# ------------- Loop de chequeo de alertas -------------
async def alerts_loop(app: Application):
    await asyncio.sleep(5)
    timeout = ClientTimeout(total=12)
    while True:
        try:
            has_any = any((len(v)>0) for v in ALERTS.values())
            if has_any:
                async with ClientSession(timeout=timeout) as session:
                    fx = await get_dolares(session)
                    rp = await get_riesgo_pais(session)
                    infl = await get_inflacion_mensual(session)
                    rv = await get_reservas_lamacro(session)
                    vals = {
                        "riesgo": float(rp[0]) if rp else None,
                        "inflacion": float(infl[0]) if infl else None,
                        "reservas": rv[0] if rv else None
                    }
                    sym_list = set()
                    for rules in ALERTS.values():
                        for r in rules:
                            if r.get("kind")=="ticker" and r.get("symbol"):
                                sym_list.add(r["symbol"])
                    metmap, _ = (await metrics_for_symbols(session, sorted(sym_list))) if sym_list else ({}, None)
                for chat_id, rules in list(ALERTS.items()):
                    if not rules: continue
                    trig = []
                    for r in rules:
                        if r.get("kind") == "fx":
                            row = fx.get(r["type"]); 
                            if not row: continue
                            cur = row.get(r["side"])
                            if cur is None: continue
                            ok = (cur > r["value"]) if r["op"] == ">" else (cur < r["value"])
                            if ok: trig.append(("fx", r["type"], r["side"], r["op"], r["value"], cur))
                        elif r.get("kind") == "metric":
                            cur = vals.get(r["type"])
                            if cur is None: continue
                            ok = (cur > r["value"]) if r["op"] == ">" else (cur < r["value"])
                            if ok: trig.append(("metric", r["type"], r["op"], r["value"], cur))
                        elif r.get("kind") == "ticker":
                            sym = r["symbol"]; per = r["period"]; m = metmap.get(sym, {})
                            cur = m.get(per)
                            if cur is None: continue
                            ok = (cur > r["value"]) if r["op"] == ">" else (cur < r["value"])
                            if ok: trig.append(("ticker", sym, per, r["op"], r["value"], cur))
                    if trig:
                        lines = [f"<b>🔔 Alertas</b>"]
                        for t, *rest in trig:
                            if t == "fx":
                                tipo, side, op, v, cur = rest
                                lines.append(f"{tipo.upper()} ({side}): {fmt_money_ars(cur)} ({'↑' if op=='>' else '↓'} {fmt_money_ars(v)})")
                            elif t == "metric":
                                tipo, op, v, cur = rest
                                if tipo=="riesgo":
                                    lines.append(f"Riesgo País: {cur:.0f} pb ({'↑' if op=='>' else '↓'} {v:.0f} pb)")
                                elif tipo=="inflacion":
                                    lines.append(f"Inflación Mensual: {str(round(cur,1)).replace('.',',')}% ({'↑' if op=='>' else '↓'} {str(round(v,1)).replace('.',',')}%)")
                                elif tipo=="reservas":
                                    lines.append(f"Reservas: {fmt_number(cur,0)} MUS$ ({'↑' if op=='>' else '↓'} {fmt_number(v,0)} MUS$)")
                            else:
                                sym, per, op, v, cur = rest
                                lines.append(f"{sym} ({per.upper()}): {pct(cur,1)} ({'↑' if op=='>' else '↓'} {pct(v,1)})")
                        try:
                            await app.bot.send_message(chat_id, "\n".join(lines), parse_mode=ParseMode.HTML, link_preview_options=LinkPreviewOptions(is_disabled=True))
                        except Exception as e:
                            log.warning("send alert failed %s: %s", chat_id, e)
            await asyncio.sleep(600)
        except Exception as e:
            log.warning("alerts_loop error: %s", e)
            await asyncio.sleep(30)

# ------------- Webhook / App -------------
async def keepalive_loop(app: Application):
    await asyncio.sleep(5)
    url = f"{BASE_URL}/"; timeout = ClientTimeout(total=6)
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
    await application.bot.set_webhook(url=WEBHOOK_URL, allowed_updates=["message","callback_query"], drop_pending_updates=True)
    cmds = [
        BotCommand("dolar", "Tipos de Cambio"),
        BotCommand("acciones", "Top 3 Acciones"),
        BotCommand("cedears", "Top 3 Cedears"),
        BotCommand("rankings_acciones", "Top 5 Acciones"),
        BotCommand("rankings_cedears", "Top 5 Cedears"),
        BotCommand("reservas", "Reservas BCRA"),
        BotCommand("inflacion", "Inflación Mensual"),
        BotCommand("riesgo", "Riesgo País"),
        BotCommand("resumen_diario", "Resumen Diario"),
        BotCommand("alertas", "Listar Alertas"),
        BotCommand("alertas_add", "Agregar Alerta"),
        BotCommand("alertas_clear", "Borrar Alertas"),
    ]
    try: await application.bot.set_my_commands(cmds)
    except Exception as e: log.warning("set_my_commands error: %s", e)
    log.info("Webhook set: %s", WEBHOOK_URL)
    asyncio.create_task(keepalive_loop(application))
    asyncio.create_task(alerts_loop(application))

async def on_cleanup(app: web.Application):
    await application.stop(); await application.shutdown()

async def handle_root(request: web.Request):
    return web.Response(text="ok", status=200)

async def handle_webhook(request: web.Request):
    if request.method != "POST": return web.Response(text="Method Not Allowed", status=405)
    try: data = await request.json()
    except Exception: return web.Response(text="Bad Request", status=400)
    update = Update.de_json(data, application.bot)
    await application.process_update(update)
    return web.Response(text="OK", status=200)

def build_web_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/", handle_root)
    app.router.add_post(WEBHOOK_PATH, handle_webhook)
    app.on_startup.append(on_startup); app.on_cleanup.append(on_cleanup)
    return app

defaults = Defaults(parse_mode=ParseMode.HTML, link_preview_options=LinkPreviewOptions(is_disabled=True), tzinfo=TZ)
application = Application.builder().token(TELEGRAM_TOKEN).defaults(defaults).updater(None).build()

# Handlers comandos
application.add_handler(CommandHandler("dolar", cmd_dolar))
application.add_handler(CommandHandler("acciones", cmd_acciones))
application.add_handler(CommandHandler("cedears", cmd_cedears))
application.add_handler(CommandHandler("rankings_acciones", cmd_rankings_acciones))
application.add_handler(CommandHandler("rankings_cedears", cmd_rankings_cedears))
application.add_handler(CommandHandler("reservas", cmd_reservas))
application.add_handler(CommandHandler("inflacion", cmd_inflacion))
application.add_handler(CommandHandler("riesgo", cmd_riesgo))
application.add_handler(CommandHandler("resumen_diario", cmd_resumen_diario))
application.add_handler(CommandHandler("alertas", cmd_alertas_list))
application.add_handler(CommandHandler("alertas_clear", cmd_alertas_clear))

# Conversación /alertas_add
conv = ConversationHandler(
    entry_points=[CommandHandler("alertas_add", alertas_add_start)],
    states={
        AL_KIND: [CallbackQueryHandler(alertas_add_kind, pattern=r"^(KIND:.*|CANCEL)$"),
                  CallbackQueryHandler(alertas_back, pattern=r"^BACK:.*$")],
        AL_FX_TYPE: [CallbackQueryHandler(alertas_add_fx_type, pattern=r"^(FXTYPE:.*|BACK:.*|CANCEL)$")],
        AL_FX_SIDE: [CallbackQueryHandler(alertas_add_fx_side, pattern=r"^(SIDE:.*|BACK:.*|CANCEL)$")],
        AL_METRIC_TYPE: [CallbackQueryHandler(alertas_add_metric_type, pattern=r"^(METRIC:.*|BACK:.*|CANCEL)$")],
        AL_OP: [CallbackQueryHandler(alertas_add_op, pattern=r"^(OP:.*|BACK:.*|CANCEL)$")],
        AL_MODE: [CallbackQueryHandler(alertas_add_mode, pattern=r"^(MODE:.*|BACK:.*|CANCEL)$")],
        AL_TICKER: [CallbackQueryHandler(alertas_add_ticker_cb, pattern=r"^(TICK:.*|BACK:.*|CANCEL)$"),
                    MessageHandler(filters.TEXT & ~filters.COMMAND, alertas_add_ticker_text)],
        AL_PERIOD: [CallbackQueryHandler(alertas_add_period, pattern=r"^(PERIOD:.*|BACK:.*|CANCEL)$")],
        AL_VALUE: [MessageHandler(filters.TEXT & ~filters.COMMAND, alertas_add_value)],
    },
    fallbacks=[CallbackQueryHandler(alertas_back, pattern=r"^BACK:.*$"),
               CallbackQueryHandler(alertas_add_start, pattern=r"^CANCEL$")],
    allow_reentry=True,
)
application.add_handler(conv)

if __name__ == "__main__":
    log.info("Iniciando Bot Económico AR (Render webhook)")
    app = build_web_app()
    web.run_app(app, host="0.0.0.0", port=PORT)
