# -*- coding: utf-8 -*-
import os, asyncio, logging, re, html as _html, json, math
from datetime import datetime, timedelta, time as dtime
from math import sqrt
from zoneinfo import ZoneInfo
from urllib.parse import urlparse
from typing import Dict, List, Tuple, Any, Optional, Set

from aiohttp import web, ClientSession, ClientTimeout
from telegram import (
    Update, BotCommand, InlineKeyboardMarkup, InlineKeyboardButton, LinkPreviewOptions
)
from telegram.constants import ParseMode
from telegram.ext import (
    Application, CommandHandler, ContextTypes, Defaults,
    CallbackQueryHandler, ConversationHandler, MessageHandler, filters
)

# ============ CONFIG ============
TZ = ZoneInfo("America/Argentina/Buenos_Aires")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN no configurado.")
BASE_URL = os.getenv("BASE_URL", os.getenv("RENDER_EXTERNAL_URL", "https://bot-economico-ar.onrender.com")).rstrip("/")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "tgwebhook").strip().strip("/")
WEBHOOK_PATH = f"/{WEBHOOK_SECRET}"
WEBHOOK_URL = f"{BASE_URL}{WEBHOOK_PATH}"
PORT = int(os.getenv("PORT", "10000"))
STATE_PATH = os.getenv("STATE_PATH", "state.json")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("bot-econ-ar")

def _ensure_state(path: str) -> str:
    try:
        d = os.path.dirname(path) or "."
        os.makedirs(d, exist_ok=True)
        with open(path, "a", encoding="utf-8"): pass
        return path
    except Exception:
        return "./state.json"

STATE_PATH = _ensure_state(STATE_PATH)

# Fuentes / endpoints
CRYPTOYA_DOLAR_URL = "https://criptoya.com/api/dolar"
DOLARAPI_BASE = "https://dolarapi.com/v1"
# ArgentinaDatos (fallback)
ARGDAT_BASES = ["https://api.argentinadatos.com/v1/finanzas/indices", "https://argentinadatos.com/v1/finanzas/indices"]
# Ámbito (riesgo país “en vivo”)
AMB_RP = "https://mercados.ambito.com//riesgo-pais/argentina"
AMB_RP_VAR = "https://mercados.ambito.com//riesgo-pais/variacion"

YF_URLS = ["https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
           "https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"]
YF_HEADERS = {"User-Agent": "Mozilla/5.0"}
REQ_HEADERS = {"User-Agent":"Mozilla/5.0", "Accept":"*/*"}

# Listados simples (recortados por brevedad)
ACCIONES_BA = ["GGAL.BA","YPFD.BA","PAMP.BA","CEPU.BA","ALUA.BA","TXAR.BA","TGSU2.BA","BYMA.BA","SUPV.BA","BMA.BA"]
CEDEARS_BA  = ["AAPL.BA","MSFT.BA","NVDA.BA","AMZN.BA","GOOGL.BA","TSLA.BA","META.BA","JNJ.BA","KO.BA","NFLX.BA"]
TICKER_NAME = {
    "GGAL.BA":"Grupo Financiero Galicia", "YPFD.BA":"YPF", "PAMP.BA":"Pampa Energía", "CEPU.BA":"Central Puerto",
    "ALUA.BA":"Aluar", "TXAR.BA":"Ternium Argentina", "TGSU2.BA":"TGS", "BYMA.BA":"BYMA",
    "SUPV.BA":"Supervielle", "BMA.BA":"Banco Macro",
    "AAPL.BA":"Apple", "MSFT.BA":"Microsoft", "NVDA.BA":"NVIDIA", "AMZN.BA":"Amazon", "GOOGL.BA":"Alphabet",
    "TSLA.BA":"Tesla", "META.BA":"Meta", "JNJ.BA":"Johnson & Johnson", "KO.BA":"Coca-Cola", "NFLX.BA":"Netflix",
}
NAME_ABBR = {k: v.split()[0] for k,v in TICKER_NAME.items()}

# Estado
ALERTS: Dict[int, List[Dict[str, Any]]] = {}
ALERTS_SILENT_UNTIL: Dict[int, float] = {}
ALERTS_PAUSED: Set[int] = set()
SUBS: Dict[int, Dict[str, Any]] = {}
PF: Dict[int, Dict[str, Any]] = {}

def load_state():
    global ALERTS, SUBS, PF
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        ALERTS = {int(k): v for k,v in data.get("alerts", {}).items()}
        SUBS = {int(k): v for k,v in data.get("subs", {}).items()}
        PF     = {int(k): v for k,v in data.get("pf", {}).items()}
        log.info("State loaded.")
    except Exception:
        log.info("No previous state found.")

def save_state():
    try:
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump({"alerts": ALERTS, "subs": SUBS, "pf": PF}, f, ensure_ascii=False)
    except Exception as e:
        log.warning("save_state error: %s", e)

# ============ Helpers ============
def fmt_number(n: Optional[float], nd=2) -> str:
    try:
        if n is None: return "—"
        s = f"{n:,.{nd}f}"
        return s.replace(",", "X").replace(".", ",").replace("X", ".")
    except: return str(n)
def fmt_money_ars(n: Optional[float], nd: int = 2) -> str:
    try:
        if n is None: return "$ —"
        return f"$ {fmt_number(float(n), nd)}"
    except: return f"$ {n}"
def pct(n: Optional[float], nd: int = 2) -> str:
    try: return f"{n:+.{nd}f}%".replace(".", ",")
    except: return "—"
def anchor(href: str, text: str) -> str: return f'<a href="{_html.escape(href, True)}">{_html.escape(text)}</a>'

def requires_integer_units(sym: str) -> bool: return sym.endswith(".BA")

# ============ HTTP ============
async def fetch_json(session: ClientSession, url: str, **kwargs) -> Optional[Any]:
    try:
        timeout = kwargs.pop("timeout", ClientTimeout(total=15))
        headers = kwargs.pop("headers", {})
        async with session.get(url, timeout=timeout, headers={**REQ_HEADERS, **headers}, **kwargs) as r:
            if r.status == 200:
                return await r.json(content_type=None)
            log.warning("GET %s -> %s", url, r.status)
    except Exception as e:
        log.warning("fetch_json error %s: %s", url, e)
    return None

async def fetch_text(session: ClientSession, url: str, **kwargs) -> Optional[str]:
    try:
        timeout = kwargs.pop("timeout", ClientTimeout(total=15))
        headers = kwargs.pop("headers", {})
        async with session.get(url, timeout=timeout, headers={**REQ_HEADERS, **headers}, **kwargs) as r:
            if r.status == 200:
                return await r.text()
            log.warning("GET %s -> %s", url, r.status)
    except Exception as e:
        log.warning("fetch_text error %s: %s", url, e)
    return None

# ============ Datos ============
async def get_dolares(session: ClientSession) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    cj = await fetch_json(session, CRYPTOYA_DOLAR_URL)
    def _safe(block: Dict[str, Any]):
        if not isinstance(block, dict): return (None, None)
        c, v = block.get("compra") or block.get("buy"), block.get("venta") or block.get("sell")
        try: return (float(c) if c is not None else None, float(v) if v is not None else None)
        except: return (None, None)
    if cj:
        for k in ["oficial","mayorista","blue","mep","ccl","cripto","tarjeta"]:
            c,v = _safe(cj.get(k,{}))
            if c is not None or v is not None:
                out[k] = {"compra": c, "venta": v, "fuente": "CriptoYa"}
    # Fallback DolarAPI
    mapping = {"oficial": "/dolares/oficial","mayorista": "/ambito/dolares/mayorista","blue": "/dolares/blue",
               "mep": "/dolares/bolsa","ccl": "/dolares/contadoconliqui","tarjeta": "/dolares/tarjeta","cripto": "/ambito/dolares/cripto"}
    for k, path in mapping.items():
        if k not in out or (out[k].get("compra") is None and out[k].get("venta") is None):
            j = await fetch_json(session, f"{DOLARAPI_BASE}{path}")
            if j:
                c, v = j.get("compra"), j.get("venta")
                try:
                    c = float(c) if c is not None else None
                    v = float(v) if v is not None else None
                except: c=v=None
                out[k] = {"compra": c, "venta": v, "fuente": "DolarAPI", "fecha": j.get("fechaActualizacion") or j.get("fecha")}
    return out

async def get_tc_value(session: ClientSession, tc_name: Optional[str]) -> Optional[float]:
    if not tc_name: return None
    fx = await get_dolares(session)
    row = fx.get(tc_name.lower(), {})
    v = row.get("venta")
    try: return float(v) if v is not None else None
    except: return None

# ===== Riesgo País (fuente alternativa + fallback) =====
def _ambito_rp_parse(j: Any) -> Optional[Tuple[int, Optional[str]]]:
    """
    Ámbito devuelve típicamente: ["fecha","valor","variación"] o similar.
    Aceptamos varios formatos y nos quedamos con valor entero.
    """
    try:
        if isinstance(j, list) and len(j) >= 2:
            # algunos endpoints dan ["22-09-2025","1234"]
            # otros: {"fecha":"...","valor":"..."}
            val_raw = j[1]
            # fecha puede ir en j[0]
            fecha = j[0] if isinstance(j[0], str) else None
            val = int(float(str(val_raw).replace(",", ".").strip()))
            return (val, fecha)
        if isinstance(j, dict) and "valor" in j:
            val = int(float(str(j["valor"]).replace(",", ".")))
            fecha = j.get("fecha")
            return (val, fecha)
    except Exception:
        return None
    return None

async def get_riesgo_pais(session: ClientSession) -> Optional[Tuple[int, Optional[str], str]]:
    """
    Devuelve (valor, fecha_opcional, fuente)
    1) Ámbito (casi real-time)
    2) Fallback: ArgentinaDatos (último dato oficial)
    """
    # 1) Ambito principal
    j = await fetch_json(session, AMB_RP)
    rp = _ambito_rp_parse(j) if j else None
    if rp:
        return (rp[0], rp[1], "Ámbito")
    # 2) Intento variación (a veces trae último con otro formato)
    j2 = await fetch_json(session, AMB_RP_VAR)
    rp2 = _ambito_rp_parse(j2) if j2 else None
    if rp2:
        return (rp2[0], rp2[1], "Ámbito")
    # 3) Fallback ArgentinaDatos
    for suf in ("/riesgo-pais/ultimo", "/riesgo-pais"):
        base_ok = None
        for base in ARGDAT_BASES:
            j3 = await fetch_json(session, base+suf)
            if j3:
                base_ok = j3
                break
        if base_ok:
            if isinstance(base_ok, dict):
                val = base_ok.get("valor")
                f = base_ok.get("fecha") or base_ok.get("periodo")
                try: return (int(float(val)), f, "ArgentinaDatos") if val is not None else None
                except: return None
            if isinstance(base_ok, list) and base_ok:
                last = base_ok[-1]
                val = last.get("valor")
                f = last.get("fecha") or last.get("periodo")
                try: return (int(float(val)), f, "ArgentinaDatos") if val is not None else None
                except: return None
    return None

# ============ Yahoo Finance (métricas mínimas para rankings) ============
async def _yf_chart_1y(session: ClientSession, symbol: str, interval: str) -> Optional[Dict[str, Any]]:
    for base in YF_URLS:
        j = await fetch_json(session, base.format(symbol=symbol), headers=YF_HEADERS,
                             params={"range": "1y", "interval": interval, "events": "div,split"})
        try:
            return j.get("chart", {}).get("result", [])[0]
        except Exception:
            continue
    return None

def _metrics_from_chart(res: Dict[str, Any]) -> Optional[Dict[str, Optional[float]]]:
    try:
        ts = res["timestamp"]; closes = res["indicators"]["adjclose"][0]["adjclose"]
        pairs = [(t,c) for t,c in zip(ts, closes) if (t and c)]
        if len(pairs) < 30: return None
        ts = [p[0] for p in pairs]; closes = [p[1] for p in pairs]
        last = closes[-1]; t_last = ts[-1]
        def first_on_or_after(tcut):
            for i,t in enumerate(ts):
                if t >= tcut: return closes[i]
            return closes[0]
        t6 = t_last - 180*24*3600; t3 = t_last - 90*24*3600; t1 = t_last - 30*24*3600
        base6, base3, base1 = first_on_or_after(t6), first_on_or_after(t3), first_on_or_after(t1)
        ret6 = (last/base6 - 1.0)*100.0 if base6 else None
        ret3 = (last/base3 - 1.0)*100.0 if base3 else None
        ret1 = (last/base1 - 1.0)*100.0 if base1 else None
        rets_d = []
        for i in range(1, len(closes)):
            if closes[i-1] and closes[i]: rets_d.append(closes[i]/closes[i-1]-1.0)
        sd = None
        if len(rets_d) >= 10:
            mu = sum(rets_d[-60:]) / min(60, len(rets_d))
            var = sum((r-mu)**2 for r in rets_d[-60:])/(min(60, len(rets_d))-1) if min(60, len(rets_d))>1 else 0.0
            sd = (var**0.5)* (252**0.5) *100.0
        hi52 = (last/max(closes) - 1.0)*100.0
        return {"6m": ret6, "3m": ret3, "1m": ret1, "last_ts": int(t_last), "vol_ann": sd, "hi52": hi52, "last_px": float(last)}
    except Exception:
        return None

async def metrics_for_symbols(session: ClientSession, symbols: List[str]) -> Tuple[Dict[str, Dict[str, Optional[float]]], Optional[int]]:
    out = {s: {"6m": None, "3m": None, "1m": None, "last_ts": None, "vol_ann": None, "hi52": None, "last_px": None} for s in symbols}
    sem = asyncio.Semaphore(4)
    async def w(sym: str):
        async with sem:
            for it in ("1d","1wk"):
                res = await _yf_chart_1y(session, sym, it)
                if not res: continue
                m = _metrics_from_chart(res)
                if m: out[sym] = m; break
    await asyncio.gather(*(w(s) for s in symbols))
    last_ts = None
    for d in out.values():
        if d.get("last_ts"): last_ts = max(last_ts or 0, d["last_ts"])
    return out, last_ts

# ============ Formatos ============
def _label_long(sym: str) -> str:
    if sym.endswith(".BA"): return f"{TICKER_NAME.get(sym, sym)} ({sym}) (ARS)"
    return sym
def _label_short(sym: str) -> str:
    if sym.endswith(".BA"): return f"{NAME_ABBR.get(sym, sym)} ({sym})"
    return sym

def format_dolar_message(d: Dict[str, Dict[str, Any]]) -> str:
    header = "<b>💵 Tipos de Cambio</b>"
    rows = ["<pre>Tipo          Compra         Venta</pre>"]
    order = [("oficial","Oficial"),("mayorista","Mayorista"),("blue","Blue"),("mep","MEP"),("ccl","CCL"),("cripto","Cripto"),("tarjeta","Tarjeta")]
    for k, lbl in order:
        row = d.get(k); 
        if not row: continue
        compra = fmt_money_ars(row.get("compra"))
        venta  = fmt_money_ars(row.get("venta"))
        rows.append(f"<pre>{lbl:<12}{compra:>12}    {venta:>12}</pre>")
    rows.append("<i>Fuentes: CriptoYa + DolarAPI</i>")
    return "\n".join([header] + rows)

def format_top3_table(title: str, fecha: Optional[str], syms: List[str], retmap: Dict[str, Dict[str, Optional[float]]]) -> str:
    head = f"<b>{title}</b>" + (f"  <i>Últ. Dato: {fecha}</i>" if fecha else "")
    lines = [head, "<pre>Rank  Empresa (Ticker)                 1M        3M        6M</pre>"]
    out = []
    for i,sym in enumerate(syms[:3],1):
        m = retmap.get(sym,{})
        p1 = pct(m.get("1m"), 2) if m.get("1m") is not None else "—"
        p3 = pct(m.get("3m"), 2) if m.get("3m") is not None else "—"
        p6 = pct(m.get("6m"), 2) if m.get("6m") is not None else "—"
        label = f"{_label_short(sym):<30}"
        out.append(f"<pre>{i:<4} {label}  {p1:^10}{p3:^10}{p6:^10}</pre>")
    if not out: out.append("<pre>—</pre>")
    return "\n".join([lines[0], lines[1]] + out)

# ============ Economía ============
async def cmd_menu_economia(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("Tipos de Cambio", callback_data="ECO:DOLAR")],
        [InlineKeyboardButton("Reservas", callback_data="ECO:RESERVAS")],
        [InlineKeyboardButton("Inflación", callback_data="ECO:INFLACION")],
        [InlineKeyboardButton("Riesgo País", callback_data="ECO:RIESGO")],
        [InlineKeyboardButton("Noticias de hoy", callback_data="ECO:NOTICIAS")],
    ])
    await update.effective_message.reply_text("🏛️ Menú Economía", reply_markup=kb)

async def econ_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    data = q.data
    # responder SIEMPRE en mensaje nuevo + volver a mostrar menú abajo
    mock_update = Update.de_json({"message": {"chat": {"id": q.message.chat_id}}}, context.bot)
    async with ClientSession() as session:
        if data == "ECO:DOLAR":
            fx = await get_dolares(session)
            await q.message.reply_text(format_dolar_message(fx) if fx else "No pude obtener cotizaciones ahora.")
            await cmd_menu_economia(mock_update, context); return
        if data == "ECO:RIESGO":
            rp = await get_riesgo_pais(session)
            if not rp:
                await q.message.reply_text("No pude obtener Riesgo País en este momento.")
            else:
                val, fecha, fuente = rp
                ftxt = f"  <i>Fuente: {fuente}{(' · '+fecha) if fecha else ''}</i>"
                await q.message.reply_text(f"<b>📈 Riesgo País</b>\n<b>{val} pb</b>\n{ftxt}")
            await cmd_menu_economia(mock_update, context); return
        if data == "ECO:INFLACION":
            # Fallback sencillo (ArgentinaDatos)
            inf = None
            for base in ARGDAT_BASES:
                j = await fetch_json(session, base+"/inflacion/mensual")
                if j and isinstance(j, list) and j:
                    last = j[-1]
                    try:
                        inf = (float(last.get("valor")), last.get("fecha") or last.get("periodo"))
                    except: pass
                    break
            if not inf: 
                await q.message.reply_text("No pude obtener Inflación ahora.")
            else:
                await q.message.reply_text(f"<b>📉 Inflación Mensual</b>  <i>{inf[1]}</i>\n<b>{str(round(inf[0],1)).replace('.',',')}%</b>\n<i>Fuente: ArgentinaDatos</i>")
            await cmd_menu_economia(mock_update, context); return
        if data == "ECO:RESERVAS":
            # lectura rápida desde LaMacro (html scrape simple)
            html = await fetch_text(session, "https://www.lamacro.ar/variables/1")
            val, fd = None, None
            if html:
                m_val = re.search(r"(?:Último dato|Valor actual)\s*:\s*([0-9\.\,]+)", html)
                m_date = re.search(r"([0-3]\d/[0-1]\d/\d{4})", html)
                if m_val:
                    s = m_val.group(1).replace('.', '').replace(',', '.')
                    try: val = float(s)
                    except: pass
                if m_date: fd = m_date.group(1)
            if val is None:
                await q.message.reply_text("No pude obtener Reservas ahora.")
            else:
                await q.message.reply_text(f"<b>🏦 Reservas BCRA</b>  <i>{fd or ''}</i>\n<b>{fmt_number(val,0)} MUS$</b>\n<i>Fuente: LaMacro</i>")
            await cmd_menu_economia(mock_update, context); return
        if data == "ECO:NOTICIAS":
            # RSS simple (2-3 fuentes)
            feeds = [
                "https://www.ambito.com/contenidos/economia.xml",
                "https://www.cronista.com/files/rss/economia.xml",
                "https://www.infobae.com/economia/rss",
            ]
            items: List[Tuple[str,str]] = []
            from xml.etree import ElementTree as ET
            for u in feeds:
                xml = await fetch_text(session, u, headers={"Accept":"application/rss+xml, application/atom+xml, */*"})
                if not xml: continue
                try:
                    root = ET.fromstring(xml)
                    for it in root.findall(".//item"):
                        t = (it.findtext("title") or "").strip()
                        l = (it.findtext("link") or "").strip()
                        if t and l: items.append((t,l))
                except: pass
            uniq = []
            seen = set()
            for t,l in items:
                if l not in seen:
                    seen.add(l)
                    uniq.append((t,l))
            uniq = uniq[:5]
            if not uniq:
                await q.message.reply_text("No pude obtener noticias ahora.")
            else:
                body = "\n\n".join([f"{i}. {anchor(l,t)}" for i,(t,l) in enumerate(uniq,1)])
                await q.message.reply_text("<b>📰 Noticias</b>\n"+body, link_preview_options=LinkPreviewOptions(is_disabled=False))
            await cmd_menu_economia(mock_update, context); return

# ============ Acciones / Cedears ============
async def cmd_acciones_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("Top 3 Acciones (Rendimiento)", callback_data="ACC:TOP3")],
        [InlineKeyboardButton("Top 5 Acciones (Proyección)", callback_data="ACC:TOP5")],
    ])
    await update.effective_message.reply_text("📊 Menú Acciones", reply_markup=kb)

async def cmd_cedears_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("Top 3 Cedears (Rendimiento)", callback_data="CED:TOP3")],
        [InlineKeyboardButton("Top 5 Cedears (Proyección)", callback_data="CED:TOP5")],
    ])
    await update.effective_message.reply_text("🌎 Menú Cedears", reply_markup=kb)

def _proj3(m): 
    # proyección simple usando 1m/3m/6m + hi52 + vol
    if not m or m.get("6m") is None: return -999
    r6 = m.get("6m") or -100; r3 = m.get("3m") or -50; r1 = m.get("1m") or -20
    hi = m.get("hi52") or -10; vol = m.get("vol_ann") or 40
    return 0.55*r6 + 0.3*r3 + 0.15*r1 + 0.15*hi - 0.06*vol
def _proj6(m):
    if not m or m.get("6m") is None: return -999
    r6 = m.get("6m") or -100; r3 = m.get("3m") or -50; r1 = m.get("1m") or -20
    hi = m.get("hi52") or -10; vol = m.get("vol_ann") or 40
    return 0.7*r6 + 0.2*r3 + 0.1*r1 + 0.2*hi - 0.06*vol

async def acc_ced_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    data = q.data
    mock_update = Update.de_json({"message": {"chat": {"id": q.message.chat_id}}}, context.bot)
    async with ClientSession() as session:
        if data in ("ACC:TOP3","CED:TOP3"):
            syms = ACCIONES_BA if data.startswith("ACC") else CEDEARS_BA
            mets, last_ts = await metrics_for_symbols(session, syms)
            fecha = datetime.fromtimestamp(last_ts, TZ).strftime("%d/%m/%Y") if last_ts else None
            pairs = sorted([(s, m.get("6m")) for s,m in mets.items() if m.get("6m") is not None], key=lambda x: x[1], reverse=True)
            top_syms = [s for s,_ in pairs[:3]]
            msg = format_top3_table("📈 Top 3 Acciones" if data.startswith("ACC") else "🌎 Top 3 Cedears", fecha, top_syms, mets)
            await q.message.reply_text(msg)
            if data.startswith("ACC"): await cmd_acciones_menu(mock_update, context)
            else: await cmd_cedears_menu(mock_update, context)
            return
        if data in ("ACC:TOP5","CED:TOP5"):
            syms = ACCIONES_BA if data.startswith("ACC") else CEDEARS_BA
            mets, last_ts = await metrics_for_symbols(session, syms)
            fecha = datetime.fromtimestamp(last_ts, TZ).strftime("%d/%m/%Y") if last_ts else None
            rows = []
            for s,m in mets.items():
                if m.get("6m") is None: continue
                rows.append((s, _proj3(m), _proj6(m)))
            rows.sort(key=lambda x: x[2], reverse=True)
            head = f"<b>🏁 Top 5 {'Acciones' if data.startswith('ACC') else 'Cedears'} (Proyección)</b>" + (f"  <i>Últ. Dato: {fecha}</i>" if fecha else "")
            detail = "\n".join([f"• {_label_short(s)} → 3M {pct(p3,1)} | 6M {pct(p6,1)}" for s,p3,p6 in rows[:5]]) or "—"
            await q.message.reply_text(f"{head}\n{detail}")
            if data.startswith("ACC"): await cmd_acciones_menu(mock_update, context)
            else: await cmd_cedears_menu(mock_update, context)
            return

# ============ Alertas (menú resumido con Agregar/Pausa funcionando) ============
AL_KIND, AL_FX_TYPE, AL_FX_SIDE, AL_OP, AL_MODE, AL_VALUE, AL_METRIC_TYPE, AL_TICKER = range(8)

def kb(rows): return InlineKeyboardMarkup([[InlineKeyboardButton(t, callback_data=d) for t,d in r] for r in rows])

async def cmd_alertas_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb_menu = kb([
        [("Listar","AL:LIST"),("Agregar","AL:ADD")],
        [("Borrar","AL:CLEAR")],
        [("Pausar","AL:PAUSE"),("Reanudar","AL:RESUME")],
    ])
    await update.effective_message.reply_text("🔔 Menú Alertas", reply_markup=kb_menu)

async def alertas_menu_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    data = q.data
    mock = Update.de_json({"message":{"chat":{"id":q.message.chat_id}}}, context.bot)
    if data == "AL:LIST":
        rules = ALERTS.get(q.message.chat_id, [])
        if not rules: await q.message.reply_text("No tenés alertas configuradas.")
        else:
            lines = ["<b>🔔 Alertas</b>"]
            for i,r in enumerate(rules,1):
                if r.get("kind")=="fx":
                    lines.append(f"{i}. {r['type'].upper()} ({r['side']}) {('↑' if r['op']=='>' else '↓')} {fmt_money_ars(r['value'])}")
                else:
                    lines.append(f"{i}. {r.get('symbol','?')} precio {('↑' if r['op']=='>' else '↓')} {fmt_money_ars(r['value'])}")
            await q.message.reply_text("\n".join(lines))
        await cmd_alertas_menu(mock, context)
    elif data == "AL:ADD":
        await alertas_add_start(mock, context)
    elif data == "AL:CLEAR":
        ALERTS[q.message.chat_id] = []; save_state()
        await q.message.reply_text("Alertas eliminadas.")
        await cmd_alertas_menu(mock, context)
    elif data == "AL:PAUSE":
        ALERTS_PAUSED.add(q.message.chat_id); ALERTS_SILENT_UNTIL.pop(q.message.chat_id, None)
        await q.message.reply_text("🔕 Alertas en pausa (indefinida).")
        await cmd_alertas_menu(mock, context)
    elif data == "AL:RESUME":
        ALERTS_PAUSED.discard(q.message.chat_id); ALERTS_SILENT_UNTIL.pop(q.message.chat_id, None)
        await q.message.reply_text("🔔 Alertas reanudadas.")
        await cmd_alertas_menu(mock, context)

def kb_submenu_fx() -> InlineKeyboardMarkup:
    return kb([
        [("Oficial","FXTYPE:oficial"),("Mayorista","FXTYPE:mayorista")],
        [("Blue","FXTYPE:blue"),("MEP","FXTYPE:mep")],
        [("CCL","FXTYPE:ccl"),("Tarjeta","FXTYPE:tarjeta")],
        [("Cripto","FXTYPE:cripto")],
        [("Cancelar","CANCEL")]
    ])

async def alertas_add_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["al"] = {}
    k = kb([[("Dólares","KIND:fx")],[("Acciones","KIND:acciones")],[("Cancelar","CANCEL")]])
    await update.effective_message.reply_text("¿Qué querés alertar?", reply_markup=k)
    return AL_KIND

async def alertas_add_kind(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    if q.data == "CANCEL": await q.message.reply_text("Cancelado."); return ConversationHandler.END
    kind = q.data.split(":",1)[1]
    context.user_data["al"] = {"kind": kind}
    if kind == "fx":
        await q.message.reply_text("Elegí el tipo de dólar:", reply_markup=kb_submenu_fx()); return AL_FX_TYPE
    if kind == "acciones":
        # simplificado: alerta por precio en acciones
        syms = ACCIONES_BA
        rows, row = [], []
        for s in syms:
            label = _label_long(s)
            row.append((label, f"TICK:{s}"))
            if len(row)==2: rows.append(row); row=[]
        if row: rows.append(row)
        rows.append([("Cancelar","CANCEL")])
        await q.message.reply_text("Elegí el ticker:", reply_markup=kb(rows)); return AL_TICKER
    await q.message.reply_text("Cancelado."); return ConversationHandler.END

async def alertas_add_fx_type(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    if q.data == "CANCEL": await q.message.reply_text("Cancelado."); return ConversationHandler.END
    t = q.data.split(":",1)[1]
    context.user_data["al"]["type"] = t
    kb_side = kb([[("Compra","SIDE:compra"),("Venta","SIDE:venta")],[("Cancelar","CANCEL")]]) if t!="tarjeta" else kb([[("Venta","SIDE:venta")],[("Cancelar","CANCEL")]])
    await q.message.reply_text("Elegí lado:", reply_markup=kb_side); return AL_FX_SIDE

async def alertas_add_fx_side(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    if q.data == "CANCEL": await q.message.reply_text("Cancelado."); return ConversationHandler.END
    side = q.data.split(":",1)[1]
    context.user_data["al"]["side"] = side
    kb_op = kb([[("↑ Sube","OP:>"),("↓ Baja","OP:<")],[("Cancelar","CANCEL")]])
    await q.message.reply_text("Elegí condición:", reply_markup=kb_op); return AL_OP

async def alertas_add_ticker_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    if q.data == "CANCEL": await q.message.reply_text("Cancelado."); return ConversationHandler.END
    sym = q.data.split(":",1)[1].upper()
    context.user_data["al"]["symbol"] = sym
    kb_op = kb([[("↑ Sube","OP:>"),("↓ Baja","OP:<")],[("Cancelar","CANCEL")]])
    await q.message.reply_text(f"Ticker: {_label_long(sym)}\nElegí condición:", reply_markup=kb_op); return AL_OP

async def alertas_add_op(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    if q.data == "CANCEL": await q.message.reply_text("Cancelado."); return ConversationHandler.END
    op = q.data.split(":",1)[1]
    context.user_data["al"]["op"] = op
    al = context.user_data.get("al", {})
    if al.get("kind") == "fx":
        kb_mode = kb([[("Ingresar Importe","MODE:absolute"),("Ingresar % vs actual","MODE:percent")],[("Cancelar","CANCEL")]])
        await q.message.reply_text("¿Cómo querés definir el umbral?", reply_markup=kb_mode); return AL_MODE
    else:
        await q.message.reply_text("Ingresá el <b>precio objetivo</b> (solo número).", parse_mode=ParseMode.HTML); return AL_VALUE

def _parse_num(s: str) -> Optional[float]:
    s = (s or "").strip()
    if re.search(r"[^\d\.,\-+]", s): return None
    s = s.replace(".","").replace(",",".")
    try: return float(s)
    except: return None

async def alertas_add_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    if q.data == "CANCEL": await q.message.reply_text("Cancelado."); return ConversationHandler.END
    mode = q.data.split(":",1)[1]
    context.user_data["al"]["mode"] = mode
    await q.message.reply_text("Ingresá el valor (solo número). Si elegiste %, ingresá el porcentaje."); return AL_VALUE

async def alertas_add_value(update: Update, context: ContextTypes.DEFAULT_TYPE):
    al = context.user_data.get("al", {})
    val = _parse_num(update.message.text)
    if val is None:
        await update.message.reply_text("Ingresá solo número (sin símbolos)."); return AL_VALUE
    chat_id = update.effective_chat.id
    if al.get("kind") == "fx":
        async with ClientSession() as session:
            fx = await get_dolares(session); row = fx.get(al["type"], {}) or {}
            cur = row.get("venta" if al.get("side")=="venta" else "compra")
            if cur is None:
                await update.message.reply_text("No pude leer el valor actual."); return ConversationHandler.END
            thr = cur*(1 + (val/100.0)) if (al.get("mode")=="percent" and al["op"]==">") else \
                  cur*(1 - (val/100.0)) if (al.get("mode")=="percent") else val
        ALERTS.setdefault(chat_id, []).append({"kind":"fx","type":al["type"],"side":al["side"],"op":al["op"],"value":float(thr)})
        save_state()
        await update.message.reply_text("Listo. Alerta agregada ✅")
        await cmd_alertas_menu(update, context)
        return ConversationHandler.END
    # ticker
    sym = al.get("symbol"); op = al.get("op")
    async with ClientSession() as session:
        mets, _ = await metrics_for_symbols(session, [sym])
    last_px = mets.get(sym, {}).get("last_px")
    if last_px is None:
        await update.message.reply_text("No pude leer el precio actual."); return ConversationHandler.END
    thr = float(val)
    if (op == ">" and thr <= last_px) or (op == "<" and thr >= last_px):
        await update.message.reply_text(f"El objetivo debe ser {'mayor' if op=='>' else 'menor'} que el precio actual ({fmt_money_ars(last_px)})."); return AL_VALUE
    ALERTS.setdefault(chat_id, []).append({"kind":"ticker","symbol":sym,"op":op,"value":thr})
    save_state()
    await update.message.reply_text("Listo. Alerta agregada ✅")
    await cmd_alertas_menu(update, context)
    return ConversationHandler.END

# ============ Suscripción diaria (igual que antes, estable) ============
SUBS_SET_TIME = range(1)
def _job_name_daily(chat_id: int) -> str: return f"daily_{chat_id}"

async def _job_send_daily(context: ContextTypes.DEFAULT_TYPE):
    chat_id = context.job.chat_id
    try:
        async with ClientSession() as session:
            fx = await get_dolares(session)
            rp = await get_riesgo_pais(session)
        lines = []
        if fx: lines.append(format_dolar_message(fx))
        if rp: lines.append(f"<b>📈 Riesgo País</b> {rp[0]} pb  <i>{rp[2]}</i>")
        await context.bot.send_message(chat_id, "\n\n".join(lines) if lines else "Sin novedades.")
    except Exception as e:
        log.warning("daily send err: %s", e)

def _schedule_daily_for_chat(app: Application, chat_id: int, hhmm: str):
    for j in app.job_queue.get_jobs_by_name(_job_name_daily(chat_id)): j.schedule_removal()
    from datetime import time as dtime
    h,m = [int(x) for x in hhmm.split(":")]
    app.job_queue.run_daily(_job_send_daily, time=dtime(hour=h, minute=m, tzinfo=TZ),
                            chat_id=chat_id, name=_job_name_daily(chat_id))

async def cmd_subs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    cur = SUBS.get(chat_id, {}).get("daily")
    txt = f"<b>📬 Suscripción</b>\nResumen Diario: {'ON ('+cur+')' if cur else 'OFF'}\nElegí un horario (hora AR):"
    # teclas
    rows, row = [], []
    for h in range(0,24):
        label = f"{h:02d}:00"
        row.append((label, f"SUBS:T:{label}"))
        if len(row)==4: rows.append(row); row=[]
    if row: rows.append(row)
    rows.append([("Desuscribirme","SUBS:OFF"),("Cerrar","SUBS:CLOSE")])
    await update.effective_message.reply_text(txt, reply_markup=kb(rows))
    return SUBS_SET_TIME

async def subs_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    data = q.data; chat_id = q.message.chat_id
    if data == "SUBS:CLOSE": return ConversationHandler.END
    if data == "SUBS:OFF":
        if chat_id in SUBS: SUBS[chat_id]["daily"]=None; save_state()
        for j in context.application.job_queue.get_jobs_by_name(_job_name_daily(chat_id)): j.schedule_removal()
        await q.message.reply_text("Suscripción cancelada."); return ConversationHandler.END
    if data.startswith("SUBS:T:"):
        hhmm = data.split(":",2)[2]
        SUBS.setdefault(chat_id, {})["daily"] = hhmm; save_state()
        _schedule_daily_for_chat(context.application, chat_id, hhmm)
        await q.message.reply_text(f"Te suscribí al Resumen Diario a las {hhmm} (hora AR)."); return ConversationHandler.END
    await q.message.reply_text("Acción inválida."); return ConversationHandler.END

# ============ Portafolio (menú + limpiar funcionando) ============
def pf_get(chat_id: int) -> Dict[str, Any]:
    return PF.setdefault(chat_id, {"base": {"moneda":"ARS","tc":"mep"}, "monto": 0.0, "items": []})

def kb_pf_main() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Ver composición", callback_data="PF:LIST")],
        [InlineKeyboardButton("Eliminar portafolio", callback_data="PF:CLEAR")],
    ])

async def cmd_portafolio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text("📦 Menú Portafolio", reply_markup=kb_pf_main())

async def pf_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    data = q.data; chat_id = q.message.chat_id
    if data == "PF:LIST":
        pf = pf_get(chat_id)
        if not pf["items"]:
            await q.message.reply_text("Tu portafolio está vacío.", reply_markup=kb_pf_main()); return
        lines = [f"<b>Portafolio</b> — Base: {pf['base']['moneda']}/{(pf['base'].get('tc') or '').upper()}",
                 f"Monto objetivo: {fmt_money_ars(pf['monto'])}"]
        for i,it in enumerate(pf["items"],1):
            if it.get("simbolo"):
                lines.append(f"{i}. {_label_long(it['simbolo'])}  |  Importe: {fmt_money_ars(it.get('importe'))}")
        await q.message.reply_text("\n".join(lines), reply_markup=kb_pf_main()); return
    if data == "PF:CLEAR":
        PF[chat_id] = {"base":{"moneda":"ARS","tc":"mep"},"monto":0.0,"items":[]}; save_state()
        await q.message.reply_text("Portafolio eliminado.", reply_markup=kb_pf_main()); return

# ============ Alert Tick ============
async def alerts_tick(context: ContextTypes.DEFAULT_TYPE):
    now_ts = datetime.now(TZ).timestamp()
    chats = [cid for cid, rules in ALERTS.items() if rules and cid not in ALERTS_PAUSED and not (cid in ALERTS_SILENT_UNTIL and ALERTS_SILENT_UNTIL[cid]>now_ts)]
    if not chats: return
    async with ClientSession(timeout=ClientTimeout(total=12)) as session:
        fx = await get_dolares(session)
        sym_all = sorted({r["symbol"] for cid in chats for r in ALERTS[cid] if r.get("kind")=="ticker"})
        mets, _ = (await metrics_for_symbols(session, sym_all)) if sym_all else ({}, None)
    for cid in chats:
        rules = ALERTS.get(cid, [])
        if not rules: continue
        trig = []
        for r in rules:
            if r.get("kind")=="fx":
                row = fx.get(r["type"], {}) or {}
                cur = row.get("venta" if r["side"]=="venta" else "compra")
                if cur is None: continue
                ok = (cur > r["value"]) if r["op"] == ">" else (cur < r["value"])
                if ok: trig.append(f"{r['type'].upper()} ({r['side']}): {fmt_money_ars(cur)} ({'↑' if r['op']=='>' else '↓'} {fmt_money_ars(r['value'])})")
            else:
                m = mets.get(r["symbol"], {})
                cur = m.get("last_px")
                if cur is None: continue
                ok = (cur > r["value"]) if r["op"] == ">" else (cur < r["value"])
                if ok: trig.append(f"{_label_long(r['symbol'])}: {fmt_money_ars(cur)} ({'↑' if r['op']=='>' else '↓'} {fmt_money_ars(r['value'])})")
        if trig:
            await context.bot.send_message(cid, "<b>🔔 Alertas</b>\n" + "\n".join("• "+t for t in trig))

# ============ Keepalive ============
async def keepalive_tick(context: ContextTypes.DEFAULT_TYPE):
    try:
        async with ClientSession(timeout=ClientTimeout(total=6)) as session:
            async with session.get(BASE_URL) as r:
                log.info("Keepalive %s -> %s", BASE_URL, r.status)
    except Exception as e:
        log.warning("keepalive err: %s", e)

# ============ Comandos ============
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text("Hola 👋 Elegí un menú:")
    await cmd_menu_economia(update, context)
    await cmd_acciones_menu(update, context)
    await cmd_cedears_menu(update, context)
    await cmd_portafolio(update, context)
    await cmd_alertas_menu(update, context)

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text("Usá los menús o comandos: /economia /acciones /cedears /portafolio /alertas /suscripcion")

async def cmd_economia(update: Update, context: ContextTypes.DEFAULT_TYPE): await cmd_menu_economia(update, context)
async def cmd_acciones(update: Update, context: ContextTypes.DEFAULT_TYPE): await cmd_acciones_menu(update, context)
async def cmd_cedears(update: Update, context: ContextTypes.DEFAULT_TYPE): await cmd_cedears_menu(update, context)
async def cmd_porta(update: Update, context: ContextTypes.DEFAULT_TYPE): await cmd_portafolio(update, context)
async def cmd_alertas(update: Update, context: ContextTypes.DEFAULT_TYPE): await cmd_alertas_menu(update, context)

# Suscripción
def _make_subs_conv():
    return ConversationHandler(
        entry_points=[CommandHandler("suscripcion", cmd_subs)],
        states={SUBS_SET_TIME: [CallbackQueryHandler(subs_cb, pattern=r"^SUBS:")]},
        fallbacks=[],
        per_message=False,
    )

# Alertas (Agregar)
def _make_alerts_conv():
    return ConversationHandler(
        entry_points=[CallbackQueryHandler(alertas_add_start, pattern=r"^AL:ADD$")],
        states={
            AL_KIND: [CallbackQueryHandler(alertas_add_kind, pattern=r"^(KIND:|CANCEL$)")],
            AL_FX_TYPE: [CallbackQueryHandler(alertas_add_fx_type, pattern=r"^(FXTYPE:|CANCEL$)")],
            AL_FX_SIDE: [CallbackQueryHandler(alertas_add_fx_side, pattern=r"^(SIDE:|CANCEL$)")],
            AL_TICKER: [CallbackQueryHandler(alertas_add_ticker_cb, pattern=r"^(TICK:|CANCEL$)")],
            AL_OP: [CallbackQueryHandler(alertas_add_op, pattern=r"^(OP:|CANCEL$)")],
            AL_MODE: [CallbackQueryHandler(alertas_add_mode, pattern=r"^(MODE:|CANCEL$)")],
            AL_VALUE: [MessageHandler(filters.TEXT & ~filters.COMMAND, alertas_add_value)],
        },
        fallbacks=[],
        per_message=False,
        allow_reentry=True,
    )

# ============ MAIN ============
async def on_startup(app: Application):
    load_state()
    # setMyCommands SIN /start y /help
    cmds = [
        BotCommand("economia","Menú economía"),
        BotCommand("acciones","Menú acciones"),
        BotCommand("cedears","Menú cedears"),
        BotCommand("portafolio","Menú portafolio"),
        BotCommand("alertas","Menú alertas"),
        BotCommand("suscripcion","Suscripción diaria"),
    ]
    await app.bot.set_my_commands(cmds)
    # webhook
    await app.bot.delete_webhook(drop_pending_updates=True)
    ok = await app.bot.set_webhook(url=WEBHOOK_URL, allowed_updates=["message","callback_query"])
    if ok: log.info("Webhook set: %s", WEBHOOK_URL)
    else: log.warning("No se pudo setear webhook.")
    # jobs
    app.job_queue.run_repeating(alerts_tick, interval=60, first=10)
    app.job_queue.run_repeating(keepalive_tick, interval=300, first=20)
    # reprogramar subs
    for chat_id, conf in SUBS.items():
        hhmm = conf.get("daily")
        if hhmm: _schedule_daily_for_chat(app, chat_id, hhmm)

def make_app() -> Application:
    defaults = Defaults(tzinfo=TZ, parse_mode=ParseMode.HTML, link_preview_options=LinkPreviewOptions(is_disabled=True))
    app = Application.builder().token(TELEGRAM_TOKEN).defaults(defaults).build()

    # Commands
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("economia", cmd_economia))
    app.add_handler(CommandHandler("acciones", cmd_acciones))
    app.add_handler(CommandHandler("cedears", cmd_cedears))
    app.add_handler(CommandHandler("portafolio", cmd_porta))
    app.add_handler(CommandHandler("alertas", cmd_alertas))

    # Menús/callbacks
    app.add_handler(CallbackQueryHandler(econ_cb, pattern=r"^ECO:"))
    app.add_handler(CallbackQueryHandler(acc_ced_cb, pattern=r"^(ACC:|CED:)"))
    app.add_handler(CallbackQueryHandler(alertas_menu_cb, pattern=r"^AL:"))
    app.add_handler(_make_alerts_conv())
    app.add_handler(CallbackQueryHandler(pf_cb, pattern=r"^PF:"))

    # Suscripción
    app.add_handler(_make_subs_conv())

    return app

async def main():
    app = make_app()
    await on_startup(app)
    # servidor webhook aiohttp integrado
    async def handle(request):
        update = Update.de_json(await request.json(), app.bot)
        await app.process_update(update)
        return web.Response(text="OK")
    srv = web.Application()
    srv.router.add_post(WEBHOOK_PATH, handle)
    srv.router.add_get("/", lambda r: web.Response(text="Bot OK"))
    runner = web.AppRunner(srv)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", PORT)
    await site.start()
    log.info("Starting webhook on 0.0.0.0:%s %s", PORT, WEBHOOK_PATH)
    # Mantener vivo
    while True:
        await asyncio.sleep(3600)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
