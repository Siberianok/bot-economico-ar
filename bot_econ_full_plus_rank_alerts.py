# bot_econ_full_plus_rank_alerts.py
# -*- coding: utf-8 -*-

import os, asyncio, logging, re, html as _html, json, math, io
from time import time
from math import sqrt, floor
from datetime import datetime, timedelta, time as dtime
from zoneinfo import ZoneInfo
from typing import Dict, List, Tuple, Any, Optional, Set
from urllib.parse import urlparse

_STORAGE_IMPORT_ERROR: Optional[Exception] = None
try:
    import storage  # type: ignore
except Exception as _err:
    storage = None  # type: ignore
    _STORAGE_IMPORT_ERROR = _err

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aiohttp import web, ClientSession, ClientTimeout
from telegram import (
    Update, LinkPreviewOptions, BotCommand, InlineKeyboardMarkup, InlineKeyboardButton,
)
from telegram.constants import ParseMode
from telegram.ext import (
    Application, CommandHandler, ContextTypes, Defaults, CallbackQueryHandler,
    MessageHandler, ConversationHandler, filters
)

# ============================ CONFIG ============================

TZ = ZoneInfo("America/Argentina/Buenos_Aires")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "tgwebhook").strip().strip("/")
PORT = int(os.getenv("PORT", "10000"))
BASE_URL = os.getenv("BASE_URL", os.getenv("RENDER_EXTERNAL_URL", "https://bot-economico-ar.onrender.com")).rstrip("/")
ENV_STATE_PATH = os.getenv("STATE_PATH", "state.json")

if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN no configurado.")

WEBHOOK_PATH = f"/{WEBHOOK_SECRET}"
WEBHOOK_URL = f"{BASE_URL}{WEBHOOK_PATH}"

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
REQ_HEADERS = {"User-Agent":"Mozilla/5.0", "Accept":"*/*"}

# ============================ LISTADOS ============================

ACCIONES_BA = [
    "GGAL.BA","YPFD.BA","PAMP.BA","CEPU.BA","ALUA.BA","TXAR.BA","TGSU2.BA","BYMA.BA","SUPV.BA","BMA.BA",
    "EDN.BA","CRES.BA","COME.BA","VALO.BA","TGNO4.BA","TRAN.BA","LOMA.BA","HARG.BA","CVH.BA","TECO2.BA"
]
CEDEARS_BA = [
    "AAPL.BA","MSFT.BA","NVDA.BA","AMZN.BA","GOOGL.BA","TSLA.BA","META.BA","JNJ.BA","KO.BA","NFLX.BA",
    "BRKB.BA","PG.BA","DISN.BA","AMD.BA","INTC.BA","NKE.BA","V.BA","MA.BA","PFE.BA","XOM.BA"
]
BONOS_AR = [
    "AL30","AL30D","AL35","AL29","GD30","GD30D","GD35","GD38","GD41","AE38",
    "AL41","AL38","GD46","AL32","GD29","AL36","AL39","GD35D","GD41D","AL29D"
]
FCI_LIST = [
    "FCI-MoneyMarket","FCI-BonosUSD","FCI-AccionesArg","FCI-Corporativos","FCI-Liquidez","FCI-Balanceado",
    "FCI-RentaMixta","FCI-RealEstate","FCI-Commodity","FCI-Tech","FCI-BonosCER","FCI-DurationCorta",
    "FCI-DurationMedia","FCI-DurationLarga","FCI-HighYield","FCI-BlueChips","FCI-Growth","FCI-Value",
    "FCI-Latam","FCI-Global"
]
LETES_LIST = [
    "LETRA-30D","LETRA-60D","LETRA-90D","LETRA-120D","LETRA-180D","LETRA-270D","LETRA-360D",
    "LETRA-12M","LETRA-18M","LETRA-24M","LETRA-USD-90D","LETRA-USD-180D","LETRA-USD-12M",
    "LETRA-CER-90D","LETRA-CER-180D","LETRA-CER-12M","LETRA-TNA-ALTA","LETRA-TNA-MEDIA","LETRA-TNA-BAJA","LETRA-ESPECIAL"
]
CRIPTO_TOP_NAMES = [
    "BTC","ETH","SOL","BNB","XRP","ADA","DOGE","TON","TRX","DOT",
    "AVAX","MATIC","LINK","LTC","UNI","BCH","ATOM","XLM","NEAR","APT"
]
def _crypto_to_symbol(cname: str) -> str: return f"{cname}-USD"

TICKER_NAME = {
    "GGAL.BA":"Grupo Financiero Galicia","YPFD.BA":"YPF","PAMP.BA":"Pampa Energía","CEPU.BA":"Central Puerto",
    "ALUA.BA":"Aluar","TXAR.BA":"Ternium Argentina","TGSU2.BA":"Transportadora de Gas del Sur",
    "BYMA.BA":"BYMA","SUPV.BA":"Supervielle","BMA.BA":"Banco Macro","EDN.BA":"Edenor","CRES.BA":"Cresud",
    "COME.BA":"Soc. Comercial del Plata","VALO.BA":"Gpo. Financiero Valores","TGNO4.BA":"Transportadora Gas Norte",
    "TRAN.BA":"Transener","LOMA.BA":"Loma Negra","HARG.BA":"Holcim Argentina","CVH.BA":"Cablevisión Holding",
    "TECO2.BA":"Telecom Argentina",
    "AAPL.BA":"Apple","MSFT.BA":"Microsoft","NVDA.BA":"NVIDIA","AMZN.BA":"Amazon","GOOGL.BA":"Alphabet",
    "TSLA.BA":"Tesla","META.BA":"Meta","JNJ.BA":"Johnson & Johnson","KO.BA":"Coca-Cola","NFLX.BA":"Netflix",
    "BRKB.BA":"Berkshire Hathaway B","PG.BA":"Procter & Gamble","DISN.BA":"Disney","AMD.BA":"AMD","INTC.BA":"Intel",
    "NKE.BA":"Nike","V.BA":"Visa","MA.BA":"Mastercard","PFE.BA":"Pfizer","XOM.BA":"ExxonMobil",
}
NAME_ABBR = {k: (v.split()[0] if ".BA" in k else v.split()[0]) for k,v in TICKER_NAME.items()}
def bono_moneda(sym: str) -> str: return "USD" if sym.endswith("D") else "ARS"

def label_with_currency(sym: str) -> str:
    if sym.endswith(".BA"):
        base = f"{TICKER_NAME.get(sym, sym)} ({sym})"
        return f"{base} (ARS)"
    if sym in BONOS_AR: return f"{sym} ({bono_moneda(sym)})"
    if sym.startswith("FCI-"):
        cur = "USD" if "USD" in sym.upper() else "ARS"
        return f"{sym.replace('-',' ')} ({cur})"
    if sym.startswith("LETRA"):
        cur = "USD" if "USD" in sym.upper() else "ARS"
        return f"{sym.replace('-',' ')} ({cur})"
    if sym in CRIPTO_TOP_NAMES: return f"{sym} (USD)"
    return sym

def requires_integer_units(sym: str) -> bool: return sym.endswith(".BA")

# ============================ LOGGING ============================

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("bot-econ-ar")

if storage is None and _STORAGE_IMPORT_ERROR:
    log.warning("No se pudo importar storage: %s", _STORAGE_IMPORT_ERROR)

APPSTASH_ENABLED = storage is not None and bool(getattr(storage, "REDIS_URL", ""))

# ============================ PERSISTENCIA ============================

def _writable_path(candidate: str) -> str:
    try:
        d = os.path.dirname(candidate) or "."
        if d and not os.path.exists(d):
            try: os.makedirs(d, exist_ok=True)
            except Exception: pass
        with open(candidate, "a", encoding="utf-8"): pass
        return candidate
    except Exception:
        fallback = "./state.json"
        try:
            with open(fallback, "a", encoding="utf-8"): pass
            log.warning("STATE_PATH no escribible (%s). Usando fallback: %s", candidate, fallback)
            return fallback
        except Exception as e:
            log.warning("No puedo escribir estado: %s", e)
            return fallback

STATE_PATH = _writable_path(ENV_STATE_PATH)
ALERTS: Dict[int, List[Dict[str, Any]]] = {}
SUBS: Dict[int, Dict[str, Any]] = {}
PF: Dict[int, Dict[str, Any]] = {}
ALERTS_SILENT_UNTIL: Dict[int, float] = {}
ALERTS_PAUSED: Set[int] = set()

async def _load_state_from_appstash() -> Tuple[
    Dict[int, List[Dict[str, Any]]], Dict[int, Dict[str, Any]], Set[int], Dict[int, float], bool, bool
]:
    alerts: Dict[int, List[Dict[str, Any]]] = {}
    portfolios: Dict[int, Dict[str, Any]] = {}
    paused: Set[int] = set()
    silent_until: Dict[int, float] = {}
    alerts_loaded = False
    pf_loaded = False

    if not APPSTASH_ENABLED or storage is None:
        return alerts, portfolios, paused, silent_until, alerts_loaded, pf_loaded

    try:
        chat_ids = await storage.alert_chats_all()
        alerts_loaded = True
        for chat_id in chat_ids:
            rules = await storage.alerts_list(chat_id)
            cleaned: List[Dict[str, Any]] = []
            for rule in rules:
                payload = dict(rule)
                payload.pop("_id", None)
                cleaned.append(payload)
            alerts[chat_id] = cleaned
            try:
                status = await storage.alerts_pause_status(chat_id)
                if status.get("paused"):
                    if status.get("indef"):
                        paused.add(chat_id)
                    else:
                        until = status.get("until")
                        if until:
                            silent_until[chat_id] = float(until)
            except Exception as e:
                log.debug("No se pudo obtener pausa de alertas para %s: %s", chat_id, e)
    except Exception as e:
        log.warning("No se pudieron cargar alertas desde AppStash: %s", e)
        alerts.clear()
        paused.clear()
        silent_until.clear()
        alerts_loaded = False

    try:
        chat_ids_pf = await storage.pf_chats_all()
        pf_loaded = True
        for chat_id in chat_ids_pf:
            data = await storage.pf_get(chat_id)
            if data is not None:
                portfolios[chat_id] = data
    except Exception as e:
        log.warning("No se pudieron cargar portafolios desde AppStash: %s", e)
        portfolios.clear()
        pf_loaded = False

    return alerts, portfolios, paused, silent_until, alerts_loaded, pf_loaded

async def _sync_state_to_appstash():
    if not APPSTASH_ENABLED or storage is None:
        return

    try:
        remote_alert_chats = set(await storage.alert_chats_all())
    except Exception as e:
        log.warning("No pude obtener alertas remotas desde AppStash: %s", e)
        remote_alert_chats = set()

    local_alert_chats = set(ALERTS.keys())
    for chat_id in local_alert_chats:
        rules = ALERTS.get(chat_id, [])
        try:
            await storage.alerts_del_all(chat_id)
            if rules:
                for rule in rules:
                    payload = dict(rule)
                    payload.pop("_id", None)
                    await storage.alerts_add(chat_id, payload)
        except Exception as e:
            log.warning("No pude sincronizar alertas para %s: %s", chat_id, e)

        try:
            if chat_id in ALERTS_PAUSED:
                await storage.alerts_pause_indef(chat_id)
            elif chat_id in ALERTS_SILENT_UNTIL:
                await storage.alerts_pause_until(chat_id, int(ALERTS_SILENT_UNTIL[chat_id]))
            else:
                await storage.alerts_resume(chat_id)
        except Exception as e:
            log.debug("No pude sincronizar pausa de alertas para %s: %s", chat_id, e)

    for chat_id in remote_alert_chats - local_alert_chats:
        try:
            await storage.alerts_del_all(chat_id)
            await storage.alerts_resume(chat_id)
        except Exception:
            pass

    try:
        remote_pf_chats = set(await storage.pf_chats_all())
    except Exception as e:
        log.warning("No pude obtener portafolios remotos desde AppStash: %s", e)
        remote_pf_chats = set()

    local_pf_chats = set(PF.keys())
    for chat_id in local_pf_chats:
        payload = PF.get(chat_id, {})
        try:
            has_items = bool(payload.get("items"))
            monto = float(payload.get("monto", 0) or 0.0)
            if has_items or monto > 0:
                await storage.pf_set(chat_id, payload)
            else:
                await storage.pf_del(chat_id)
        except Exception as e:
            log.warning("No pude sincronizar portafolio para %s: %s", chat_id, e)

    for chat_id in remote_pf_chats - local_pf_chats:
        try:
            await storage.pf_del(chat_id)
        except Exception:
            pass

async def load_state():
    global ALERTS, SUBS, PF, ALERTS_PAUSED, ALERTS_SILENT_UNTIL
    ALERTS_PAUSED = set()
    ALERTS_SILENT_UNTIL = {}
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        ALERTS = {int(k): v for k,v in data.get("alerts", {}).items()}
        SUBS = {int(k): v for k,v in data.get("subs", {}).items()}
        PF = {int(k): v for k,v in data.get("pf", {}).items()}
        log.info("State loaded. alerts=%d subs=%d pf=%d", sum(len(v) for v in ALERTS.values()), len(SUBS), len(PF))
    except Exception:
        log.info("No previous local state found.")

    if APPSTASH_ENABLED and storage is not None:
        try:
            (
                alerts_remote,
                pf_remote,
                paused_remote,
                silent_remote,
                alerts_ok,
                pf_ok,
            ) = await _load_state_from_appstash()
            if alerts_ok:
                ALERTS = alerts_remote
                ALERTS_PAUSED = paused_remote
                ALERTS_SILENT_UNTIL = silent_remote
            if pf_ok:
                PF = pf_remote
            if alerts_ok or pf_ok:
                log.info(
                    "Estado sincronizado desde AppStash. alerts=%d pf=%d",
                    sum(len(v) for v in ALERTS.values()),
                    len(PF),
                )
        except Exception as e:
            log.warning("Fallo al sincronizar estado desde AppStash: %s", e)

async def save_state():
    try:
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump({"alerts": ALERTS, "subs": SUBS, "pf": PF}, f, ensure_ascii=False)
    except Exception as e:
        log.warning("save_state error: %s", e)

    try:
        await _sync_state_to_appstash()
    except Exception as e:
        log.warning("No se pudo sincronizar AppStash: %s", e)

# ============================ UTILS ============================

def fmt_number(n: Optional[float], nd=2) -> str:
    try:
        if n is None: return "—"
        s = f"{n:,.{nd}f}"
        return s.replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return str(n)

def fmt_money_ars(n: Optional[float], nd: int = 2) -> str:
    try:
        if n is None: return "$ —"
        return f"$ {fmt_number(float(n), nd)}"
    except Exception:
        return f"$ {n}"

def fmt_money_usd(n: Optional[float], nd: int = 2) -> str:
    try:
        if n is None: return "US$ —"
        return f"US$ {fmt_number(float(n), nd)}"
    except Exception:
        return f"US$ {n}"

def pct(n: Optional[float], nd: int = 2) -> str:
    try: return f"{n:+.{nd}f}%".replace(".", ",")
    except Exception: return "—"

def anchor(href: str, text: str) -> str: return f'<a href="{_html.escape(href, True)}">{_html.escape(text)}</a>'
def html_op(op: str) -> str: return "↑" if op == ">" else "↓"
def pad(s: str, width: int) -> str: s=s[:width]; return s+(" "*(width-len(s)))
def center_text(s: str, width: int) -> str:
    s=str(s)[:width]; total=width-len(s); left=total//2; right=total-left; return " "*left+s+" "*right
def parse_iso_ddmmyyyy(s: Optional[str]) -> Optional[str]:
    if not s: return None
    try:
        if re.match(r"^\d{4}-\d{2}-\d{2}", s):
            return datetime.strptime(s[:10], "%Y-%m-%d").strftime("%d/%m/%Y")
    except Exception: pass
    return s

# ============================ HTTP HELPERS ============================

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

# ============================ DATA SOURCES ============================

async def get_dolares(session: ClientSession) -> Dict[str, Dict[str, Any]]:
    data: Dict[str, Dict[str, Any]] = {}
    cj = await fetch_json(session, CRYPTOYA_DOLAR_URL)
    if cj:
        def _safe(block: Dict[str, Any]):
            if not isinstance(block, dict): return (None, None)
            c, v = block.get("compra") or block.get("buy"), block.get("venta") or block.get("sell")
            try: return (float(c) if c is not None else None, float(v) if v is not None else None)
            except Exception: return (None, None)
        for k in ["oficial","mayorista","blue","mep","ccl","cripto","tarjeta"]:
            c,v = _safe(cj.get(k,{}))
            if c is not None or v is not None:
                data[k] = {"compra": c, "venta": v, "fuente": "CriptoYa"}

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

async def get_tc_value(session: ClientSession, tc_name: Optional[str]) -> Optional[float]:
    if not tc_name: return None
    fx = await get_dolares(session)
    row = fx.get(tc_name.lower(), {})
    v = row.get("venta")
    try: return float(v) if v is not None else None
    except: return None

async def get_riesgo_pais(session: ClientSession) -> Optional[Tuple[int, Optional[str]]]:
    for suf in ("/riesgo-pais/ultimo", "/riesgo-pais"):
        base_ok = None
        for base in ARG_DATOS_BASES:
            j = await fetch_json(session, base+suf)
            if j:
                base_ok = j; break
        if base_ok:
            j = base_ok; break
        else: j = None
    if isinstance(j, dict):
        val = j.get("valor"); f = j.get("fecha") or j.get("periodo")
        try: return (int(float(val)), f) if val is not None else None
        except Exception: return None
    if isinstance(j, list) and j:
        last = j[-1]; val = last.get("valor"); f = last.get("fecha") or last.get("periodo")
        try: return (int(float(val)), f) if val is not None else None
        except Exception: return None
    return None

async def get_inflacion_mensual(session: ClientSession) -> Optional[Tuple[float, Optional[str]]]:
    for suf in ("/inflacion", "/inflacion/mensual/ultimo", "/inflacion/mensual"):
        j = None
        for base in ARG_DATOS_BASES:
            j = await fetch_json(session, base+suf)
            if j:
                if isinstance(j, dict) and "serie" in j and isinstance(j["serie"], list) and j["serie"]:
                    j = j["serie"]
                break
        if j: break
    if isinstance(j, list) and j:
        last = j[-1]; val = last.get("valor"); per = last.get("fecha") or last.get("periodo")
    elif isinstance(j, dict):
        val = j.get("valor"); per = j.get("fecha") or j.get("periodo")
    else: return None
    if val is None: return None
    try: return (float(val), per)
    except Exception: return None

async def get_reservas_lamacro(session: ClientSession) -> Optional[Tuple[float, Optional[str]]]:
    html = await fetch_text(session, LAMACRO_RESERVAS_URL)
    if not html: return None
    m_val = re.search(r"(?:Último dato|Valor actual)\s*:\s*([0-9\.\,]+)", html)
    m_date = re.search(r"([0-3]\d/[0-1]\d/\d{4})", html)
    if not m_val: return None
    s = m_val.group(1).replace('.', '').replace(',', '.')
    try: val = float(s)
    except Exception: return None
    fecha = m_date.group(1) if m_date else None
    return (val, fecha)

# ============================ YF MÉTRICAS ============================

async def _yf_chart_1y(session: ClientSession, symbol: str, interval: str) -> Optional[Dict[str, Any]]:
    for base in YF_URLS:
        params = {"range": "1y", "interval": interval, "events": "div,split"}
        j = await fetch_json(session, base.format(symbol=symbol), headers=YF_HEADERS, params=params)
        try:
            res = j.get("chart", {}).get("result", [])[0]
            return res
        except Exception:
            continue
    return None

def _metrics_from_chart(res: Dict[str, Any]) -> Optional[Dict[str, Optional[float]]]:
    try:
        ts = res["timestamp"]; closes_raw = res["indicators"]["adjclose"][0]["adjclose"]
        pairs = [(t,c) for t,c in zip(ts, closes_raw) if (t is not None and c is not None)]
        if len(pairs) < 30: return None
        ts = [p[0] for p in pairs]; closes = [p[1] for p in pairs]
        idx_last = len(closes)-1; last = closes[idx_last]; t_last = ts[idx_last]

        def first_on_or_after(tcut):
            for i, t in enumerate(ts):
                if t >= tcut: return closes[i]
            return closes[0]

        t6 = t_last - 180*24*3600; t3 = t_last - 90*24*3600; t1 = t_last - 30*24*3600
        base6 = first_on_or_after(t6); base3 = first_on_or_after(t3); base1 = first_on_or_after(t1)
        ret6 = (last/base6 - 1.0)*100.0 if base6 else None
        ret3 = (last/base3 - 1.0)*100.0 if base3 else None
        ret1 = (last/base1 - 1.0)*100.0 if base1 else None

        rets_d = []
        for i in range(1, len(closes)):
            if closes[i-1] and closes[i]:
                rets_d.append(closes[i]/closes[i-1]-1.0)
        look = 60 if len(rets_d) >= 60 else max(10, len(rets_d)-1)
        vol_ann = None
        if len(rets_d) >= 10:
            mu = sum(rets_d[-look:]) / len(rets_d[-look:])
            var = sum((r-mu)**2 for r in rets_d[-look:])/(len(rets_d[-look:])-1) if len(rets_d[-look:])>1 else 0.0
            sd = sqrt(var); vol_ann = sd*sqrt(252)*100.0

        idx_cut = next((i for i,t in enumerate(ts) if t >= t6), 0)
        peak = closes[idx_cut]; dd_min = 0.0
        for v in closes[idx_cut:]:
            if v > peak: peak = v
            dd = v/peak - 1.0
            if dd < 0 and abs(dd) > abs(dd_min): dd_min = dd
        dd6 = abs(dd_min)*100.0 if dd_min < 0 else 0.0
        hi52 = (last/max(closes) - 1.0)*100.0

        def _sma(vals, w):
            out, s, q = [None]*len(vals), 0.0, []
            for i, v in enumerate(vals):
                q.append(v); s += v
                if len(q) > w: s -= q.pop(0)
                if len(q) == w: out[i] = s/w
            return out
        sma50 = _sma(closes, 50); sma200 = _sma(closes, 200)
        s50_last = sma50[idx_last] if idx_last < len(sma50) else None
        s50_prev = sma50[idx_last-20] if idx_last-20 >= 0 else None
        slope50 = ((s50_last/s50_prev - 1.0)*100.0) if (s50_last and s50_prev) else 0.0
        s200_last = sma200[idx_last] if idx_last < len(sma200) else None
        trend_flag = 1 if (s200_last and last > s200_last) else (-1 if s200_last else 0)
        return {"1m": ret1, "3m": ret3, "6m": ret6, "last_ts": int(t_last), "vol_ann": vol_ann,
                "dd6m": dd6, "hi52": hi52, "slope50": slope50, "trend_flag": float(trend_flag), "last_px": float(last)}
    except Exception:
        return None

async def _yf_metrics_1y(session: ClientSession, symbol: str) -> Dict[str, Optional[float]]:
    out = {"6m": None, "3m": None, "1m": None, "last_ts": None, "vol_ann": None, "dd6m": None, "hi52": None, "slope50": None, "trend_flag": None, "last_px": None}
    for interval in ("1d", "1wk"):
        res = await _yf_chart_1y(session, symbol, interval)
        if res:
            m = _metrics_from_chart(res)
            if m: out.update(m); break
    return out

async def metrics_for_symbols(session: ClientSession, symbols: List[str]) -> Tuple[Dict[str, Dict[str, Optional[float]]], Optional[int]]:
    out = {s: {"6m": None, "3m": None, "1m": None, "last_ts": None, "vol_ann": None, "dd6m": None, "hi52": None, "slope50": None, "trend_flag": None, "last_px": None} for s in symbols}
    sem = asyncio.Semaphore(4)
    async def work(sym: str):
        async with sem:
            out[sym] = await _yf_metrics_1y(session, sym)
    await asyncio.gather(*(work(s) for s in symbols))
    last_ts = None
    for d in out.values():
        ts = d.get("last_ts")
        if ts: last_ts = ts if last_ts is None else max(last_ts, ts)
    return out, last_ts

# ============================ NOTICIAS ============================

from xml.etree import ElementTree as ET
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
    for kw in ("sube","baja","récord","acelera","cae","acuerdo","medida","ley","resolución","reperfil","brecha","dólar","inflación"):
        if kw in t: score += 1
    return score

def _parse_feed_entries(xml: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    try:
        root = ET.fromstring(xml)
    except Exception:
        return out
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

def _impact_lines(title: str) -> str:
    t = title.lower()
    if any(k in t for k in ["dólar","mep","ccl","blue","brecha"]):
        parts = ["Impacto probable: presión en brecha y expectativas devaluatorias.",
                 "Qué mirar: CCL/MEP, intervención BCRA y flujos en bonos/cedears."]
    elif any(k in t for k in ["inflación","ipc","precios"]):
        parts = ["Impacto probable: ajuste de expectativas de tasas y salarios.",
                 "Qué mirar: núcleo, regulados y pass-through."]
    elif any(k in t for k in ["bcra","reservas","pases","tasas"]):
        parts = ["Impacto probable: anclaje de expectativas y tipo de cambio.",
                 "Qué mirar: intervención spot, pases y deuda."]
    elif "riesgo" in t or "bonos" in t:
        parts = ["Impacto probable: costo de financiamiento y apetito riesgo.",
                 "Qué mirar: spreads, vencimientos y FMI."]
    else:
        parts = ["Impacto probable: variable macro/mercado relevante.",
                 "Qué mirar: precios relativos y expectativas."]
    return "\n".join([f"<i>{p}</i>" for p in parts])

async def fetch_rss_entries(session: ClientSession, limit: int = 5) -> List[Tuple[str, str]]:
    entries: List[Tuple[str, str]] = []
    for url in RSS_FEEDS:
        xml = await fetch_text(session, url, headers={"Accept":"application/rss+xml, application/atom+xml, */*"})
        if not xml: continue
        try: entries.extend(_parse_feed_entries(xml))
        except Exception as e: log.warning("RSS parse %s: %s", url, e)

    uniq: Dict[str, str] = {l:t for t,l in entries if l.startswith("http")}
    if not uniq:
        return [("Mercados: sin novedades relevantes", "https://www.ambito.com/"),
                ("Actividad: esperando datos de inflación", "https://www.cronista.com/"),
                ("Dólar: foco en brecha y CCL/MEP", "https://www.infobae.com/economia/")][:limit]

    scored = sorted([(t,l,_score_title(t), domain_of(l)) for l,t in uniq.items()], key=lambda x: x[2], reverse=True)
    picked: List[Tuple[str,str]] = []; used_domains = set()
    for t,l,_,dom in scored:
        if dom in used_domains and len(used_domains) < 4: continue
        used_domains.add(dom); picked.append((t,l))
        if len(picked) >= limit: break
    for t,l,_,dom in scored:
        if len(picked) >= limit: break
        if (t,l) not in picked: picked.append((t,l))
    return picked[:limit]

def format_news_block(news: List[Tuple[str, str]]) -> str:
    if not news: return "<b>📰 Noticias</b>\n—"
    body = "\n\n".join([f"{i}. {anchor(l, t)}\n{_impact_lines(t)}" for i,(t,l) in enumerate(news, 1)])
    return "<b>📰 Noticias</b>\n" + body

# ============================ FORMATS & RANKINGS ============================

def _label_long(sym: str) -> str: return label_with_currency(sym)
def _label_short(sym: str) -> str:
    if sym.endswith(".BA"): return f"{NAME_ABBR.get(sym, sym)} ({sym})"
    return label_with_currency(sym)

def format_dolar_message(d: Dict[str, Dict[str, Any]]) -> str:
    fecha = None
    for row in d.values():
        f = row.get("fecha")
        if f: fecha = parse_iso_ddmmyyyy(f)
    header = "<b>💵 Dólares</b>" + (f" <i>Actualizado: {fecha}</i>" if fecha else "")
    lines = [header, "<pre>Tipo         Compra        Venta</pre>"]
    rows = []
    order = [("oficial","Oficial"),("mayorista","Mayorista"),("blue","Blue"),("mep","MEP"),("ccl","CCL"),("cripto","Cripto"),("tarjeta","Tarjeta")]
    for k, label in order:
        row = d.get(k)
        if not row: continue
        compra_val = row.get("compra"); venta_val = row.get("venta")
        compra = fmt_money_ars(compra_val) if compra_val is not None else "—"
        venta = fmt_money_ars(venta_val) if venta_val is not None else "—"
        l = f"{label:<12}{compra:>12} {venta:>12}"
        rows.append(f"<pre>{l}</pre>")
    rows.append("<i>Fuentes: CriptoYa + DolarAPI</i>")
    return "\n".join([lines[0], lines[1]] + rows)

def format_top3_table(title: str, fecha: Optional[str], rows_syms: List[str], retmap: Dict[str, Dict[str, Optional[float]]]) -> str:
    head = f"<b>{title}</b>" + (f" <i>Últ. Dato: {fecha}</i>" if fecha else "")
    lines = [head, "<pre>Rank Empresa (Ticker)             1M        3M        6M</pre>"]
    out = []
    for idx, sym in enumerate(rows_syms[:3], start=1):
        d = retmap.get(sym, {})
        p1 = pct(d.get("1m"), 2) if d.get("1m") is not None else "—"
        p3 = pct(d.get("3m"), 2) if d.get("3m") is not None else "—"
        p6 = pct(d.get("6m"), 2) if d.get("6m") is not None else "—"
        label = pad(_label_short(sym), 28)
        c1 = center_text(p1, 10); c3 = center_text(p3, 10); c6 = center_text(p6, 10)
        l = f"{idx:<4} {label}{c1}{c3}{c6}"
        out.append(f"<pre>{l}</pre>")
    if not out: out.append("<pre>—</pre>")
    return "\n".join([lines[0], lines[1]] + out)

def format_proj_dual(title: str, fecha: Optional[str], rows: List[Tuple[str, float, float]]) -> str:
    head = f"<b>{title}</b>" + (f" <i>Últ. Dato: {fecha}</i>" if fecha else "")
    sub = "<i>Proy. 3M (corto) y Proy. 6M (medio)</i>"
    lines = [head, sub, "<pre>Rank Empresa (Ticker)             Proy. 3M     Proy. 6M</pre>"]
    out = []
    if not rows: out.append("<pre>—</pre>")
    else:
        for idx, (sym, p3v, p6v) in enumerate(rows[:5], start=1):
            p3 = pct(p3v, 1) if p3v is not None else "—"
            p6 = pct(p6v, 1) if p6v is not None else "—"
            label = pad(_label_short(sym), 28)
            c3 = center_text(p3, 12); c6 = center_text(p6, 12)
            l = f"{idx:<4} {label}{c3}{c6}"
            out.append(f"<pre>{l}</pre>")
    return "\n".join(lines + out)

def _nz(x: Optional[float], fb: float) -> float: return float(x) if x is not None else fb

def projection_3m(m: Dict[str, Optional[float]]) -> float:
    r6, r3, r1 = _nz(m.get("6m"), -80.0), _nz(m.get("3m"), -40.0), _nz(m.get("1m"), -15.0)
    momentum = 0.1*r6 + 0.65*r3 + 0.25*r1
    trend = 2.2*_nz(m.get("trend_flag"), 0.0) + 0.28*_nz(m.get("slope50"), 0.0)
    hi52 = 0.18*_nz(m.get("hi52"), 0.0)
    risk = -0.055*_nz(m.get("vol_ann"), 40.0) - 0.045*_nz(m.get("dd6m"), 30.0)
    meanrev = -0.06*max(0.0, abs(r1)-12.0)
    return momentum + trend + hi52 + risk + meanrev

def projection_6m(m: Dict[str, Optional[float]]) -> float:
    r6, r3, r1 = _nz(m.get("6m"), -100.0), _nz(m.get("3m"), -50.0), _nz(m.get("1m"), -20.0)
    momentum = 0.6*r6 + 0.3*r3 + 0.1*r1
    trend = 3.1*_nz(m.get("trend_flag"), 0.0) + 0.22*_nz(m.get("slope50"), 0.0)
    hi52 = 0.22*_nz(m.get("hi52"), 0.0)
    risk = -0.06*_nz(m.get("vol_ann"), 40.0) - 0.05*_nz(m.get("dd6m"), 30.0)
    meanrev = -0.05*max(0.0, abs(r1)-15.0)
    return momentum + trend + hi52 + risk + meanrev

async def _rank_top3(update: Update, symbols: List[str], title: str):
    async with ClientSession() as session:
        mets, last_ts = await metrics_for_symbols(session, symbols)
        fecha = datetime.fromtimestamp(last_ts, TZ).strftime("%d/%m/%Y") if last_ts else None
        pairs = sorted([(sym, m["6m"]) for sym,m in mets.items() if m.get("6m") is not None], key=lambda x: x[1], reverse=True)
        top_syms = [sym for sym,_ in pairs[:3]]
        msg = format_top3_table(title, fecha, top_syms, mets)
        await update.effective_message.reply_text(msg)

async def _rank_proj5(update: Update, symbols: List[str], title: str):
    async with ClientSession() as session:
        mets, last_ts = await metrics_for_symbols(session, symbols)
        fecha = datetime.fromtimestamp(last_ts, TZ).strftime("%d/%m/%Y") if last_ts else None
        rows = []
        for sym, m in mets.items():
            if m.get("6m") is None: continue
            rows.append((sym, projection_3m(m), projection_6m(m)))
        rows.sort(key=lambda x: x[2], reverse=True)
        msg = format_proj_dual(title, fecha, rows[:5])
        await update.effective_message.reply_text(msg)

# ============================ COMANDOS / MENÚS ============================

def set_menu_counter(context: ContextTypes.DEFAULT_TYPE, name: str, n: int):
    context.user_data.setdefault("menu_counts", {})[name] = n
async def dec_and_maybe_show(update: Update, context: ContextTypes.DEFAULT_TYPE, name: str, show_func):
    cnt = context.user_data.get("menu_counts", {}).get(name, 0)
    cnt = max(0, cnt-1)
    context.user_data["menu_counts"][name] = cnt
    if cnt > 0:
        await show_func(update, context)

async def cmd_dolar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    async with ClientSession() as session:
        data = await get_dolares(session)
    msg = format_dolar_message(data) if data else "No pude obtener cotizaciones ahora."
    await update.effective_message.reply_text(msg)

async def cmd_acciones_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    set_menu_counter(context, "acciones", 2)
    kb_menu = InlineKeyboardMarkup([
        [InlineKeyboardButton("Top 3 Acciones (Rendimiento)", callback_data="ACC:TOP3")],
        [InlineKeyboardButton("Top 5 Acciones (Proyección)", callback_data="ACC:TOP5")],
    ])
    await update.effective_message.reply_text("📊 Menú Acciones", reply_markup=kb_menu)

async def cmd_cedears_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    set_menu_counter(context, "cedears", 2)
    kb_menu = InlineKeyboardMarkup([
        [InlineKeyboardButton("Top 3 Cedears (Rendimiento)", callback_data="CED:TOP3")],
        [InlineKeyboardButton("Top 5 Cedears (Proyección)", callback_data="CED:TOP5")],
    ])
    await update.effective_message.reply_text("🌎 Menú Cedears", reply_markup=kb_menu)

async def acc_ced_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    data = q.data
    if data == "ACC:TOP3":
        await _rank_top3(update, ACCIONES_BA, "📈 Top 3 Acciones (Rendimiento)")
        await dec_and_maybe_show(update, context, "acciones", cmd_acciones_menu)
    elif data == "ACC:TOP5":
        await _rank_proj5(update, ACCIONES_BA, "🏁 Top 5 Acciones (Proyección)")
        await dec_and_maybe_show(update, context, "acciones", cmd_acciones_menu)
    elif data == "CED:TOP3":
        await _rank_top3(update, CEDEARS_BA, "🌎 Top 3 Cedears (Rendimiento)")
        await dec_and_maybe_show(update, context, "cedears", cmd_cedears_menu)
    elif data == "CED:TOP5":
        await _rank_proj5(update, CEDEARS_BA, "🏁 Top 5 Cedears (Proyección)")
        await dec_and_maybe_show(update, context, "cedears", cmd_cedears_menu)

# ---------- Macro ----------

async def cmd_reservas(update: Update, context: ContextTypes.DEFAULT_TYPE):
    async with ClientSession() as session:
        res = await get_reservas_lamacro(session)
    if not res:
        txt = "No pude obtener reservas ahora."
    else:
        val, fecha = res
        txt = (f"<b>🏦 Reservas BCRA</b>{f' <i>Últ. Act.: {fecha}</i>' if fecha else ''}\n"
               f"<b>{fmt_number(val,0)} MUS$</b>\n<i>Fuente: LaMacro</i>")
    await update.effective_message.reply_text(txt)

async def cmd_inflacion(update: Update, context: ContextTypes.DEFAULT_TYPE):
    async with ClientSession() as session:
        tup = await get_inflacion_mensual(session)
    if tup is None:
        txt = "No pude obtener inflación ahora."
    else:
        val, fecha = tup; val_str = str(round(val,1)).replace(".", ",")
        txt = f"<b>📉 Inflación Mensual</b>{f' <i>{fecha}</i>' if fecha else ''}\n<b>{val_str}%</b>\n<i>Fuente: ArgentinaDatos</i>"
    await update.effective_message.reply_text(txt)

async def cmd_riesgo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    async with ClientSession() as session:
        tup = await get_riesgo_pais(session)
    if tup is None:
        txt = "No pude obtener riesgo país ahora."
    else:
        rp, f = tup; f_str = parse_iso_ddmmyyyy(f)
        txt = f"<b>📈 Riesgo País</b>{f' <i>{f_str}</i>' if f_str else ''}\n<b>{rp} pb</b>\n<i>Fuente: ArgentinaDatos</i>"
    await update.effective_message.reply_text(txt)

async def cmd_noticias(update: Update, context: ContextTypes.DEFAULT_TYPE):
    async with ClientSession() as session:
        news = await fetch_rss_entries(session, limit=5)
    txt = format_news_block(news or [])
    await update.effective_message.reply_text(txt)

async def cmd_menu_economia(update: Update, context: ContextTypes.DEFAULT_TYPE):
    set_menu_counter(context, "economia", 5)
    kb_menu = InlineKeyboardMarkup([
        [InlineKeyboardButton("Tipos de Cambio", callback_data="ECO:DOLAR")],
        [InlineKeyboardButton("Reservas", callback_data="ECO:RESERVAS")],
        [InlineKeyboardButton("Inflación", callback_data="ECO:INFLACION")],
        [InlineKeyboardButton("Riesgo País", callback_data="ECO:RIESGO")],
        [InlineKeyboardButton("Noticias de hoy", callback_data="ECO:NOTICIAS")],
    ])
    await update.effective_message.reply_text("🏛️ Menú Economía", reply_markup=kb_menu)

async def econ_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    data = q.data
    if data == "ECO:DOLAR":     await cmd_dolar(update, context)
    if data == "ECO:RESERVAS":  await cmd_reservas(update, context)
    if data == "ECO:INFLACION": await cmd_inflacion(update, context)
    if data == "ECO:RIESGO":    await cmd_riesgo(update, context)
    if data == "ECO:NOTICIAS":  await cmd_noticias(update, context)
    await dec_and_maybe_show(update, context, "economia", cmd_menu_economia)

# ============================ ALERTAS ============================

AL_KIND, AL_FX_TYPE, AL_FX_SIDE, AL_OP, AL_MODE, AL_VALUE, AL_METRIC_TYPE, AL_TICKER = range(8)

def kb(rows: List[List[Tuple[str,str]]]) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton(text, callback_data=data) for text, data in r] for r in rows])

def kb_tickers(symbols: List[str], back_target: str, prefix: str) -> InlineKeyboardMarkup:
    rows: List[List[Tuple[str,str]]] = []; row: List[Tuple[str,str]] = []
    for s in symbols:
        label = _label_long(s)
        row.append((label, f"{prefix}:{s}"))
        if len(row) == 2: rows.append(row); row = []
    if row: rows.append(row)
    rows.append([("Volver","BACK:"+back_target), ("Cancelar","CANCEL")])
    return kb(rows)

def _parse_float_user_strict(s: str) -> Optional[float]:
    s = (s or "").strip()
    if re.search(r"[^\d\.,\-+]", s): return None
    s = s.replace(".", "").replace(",", ".")
    try: return float(s)
    except Exception: return None

def _fx_display_value(row: Dict[str, Any], side: str) -> Optional[float]:
    if side == "compra": return row.get("compra")
    if side == "venta":  return row.get("venta")
    return None

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
    if data == "AL:LIST":   await cmd_alertas_list(update, context); await cmd_alertas_menu(update, context)
    if data == "AL:CLEAR":  await cmd_alertas_clear(update, context)
    if data == "AL:PAUSE":  await cmd_alertas_pause(update, context)
    if data == "AL:RESUME": await cmd_alertas_resume(update, context); await cmd_alertas_menu(update, context)

async def cmd_alertas_list(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    rules = ALERTS.get(chat_id, [])
    if not rules:
        txt = "No tenés alertas configuradas.\nUsá /alertas_menu → Agregar."
    else:
        lines = ["<b>🔔 Alertas Configuradas</b>"]
        for i, r in enumerate(rules, 1):
            if r.get("kind") == "fx":
                t, side, op, v = r["type"], r["side"], r["op"], r["value"]
                lines.append(f"{i}. {t.upper()} ({side}) {html_op(op)} {fmt_money_ars(v)}")
            elif r.get("kind") == "metric":
                t, op, v = r["type"], r["op"], r["value"]
                if t=="riesgo":     val = f"{v:.0f} pb"
                elif t=="reservas": val = f"{fmt_number(v,0)} MUS$"
                else:               val = f"{str(round(v,1)).replace('.',',')}%"
                lines.append(f"{i}. {t.upper()} {html_op(op)} {val}")
            else:
                sym, op, v = r["symbol"], r["op"], r["value"]
                lines.append(f"{i}. {_label_long(sym)} (Precio) {html_op(op)} {fmt_money_ars(v)}")
        if chat_id in ALERTS_PAUSED:
            lines.append("\n<i>Alertas en pausa (indefinida)</i>")
        txt = "\n".join(lines)
    await update.effective_message.reply_text(txt)

async def cmd_alertas_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    rules = ALERTS.get(chat_id, [])
    if not rules:
        await update.effective_message.reply_text("No tenés alertas guardadas."); return
    buttons: List[List[Tuple[str,str]]] = []
    for i, r in enumerate(rules, 1):
        if r.get("kind") == "fx":
            label = f"{i}. {r['type'].upper()}({r['side']}) {html_op(r['op'])} {fmt_money_ars(r['value'])}"
        elif r.get("kind") == "metric":
            if r["type"]=="riesgo":     val = f"{r['value']:.0f} pb"
            elif r["type"]=="reservas": val = f"{fmt_number(r['value'],0)} MUS$"
            else:                       val = f"{str(round(r['value'],1)).replace('.',',')}%"
            label = f"{i}. {r['type'].upper()} {html_op(r['op'])} {val}"
        else:
            label = f"{i}. {_label_long(r['symbol'])} {html_op(r['op'])} {fmt_money_ars(r['value'])}"
        buttons.append([(label, f"CLR:{i-1}")])
    buttons.append([("Borrar Todas","CLR:ALL"), ("Cancelar","CLR:CANCEL")])
    await update.effective_message.reply_text("Elegí qué alerta borrar:", reply_markup=InlineKeyboardMarkup(
        [[InlineKeyboardButton(t, callback_data=d) for t,d in row] for row in buttons]
    ))

async def alertas_clear_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    chat_id = q.message.chat_id
    rules = ALERTS.get(chat_id, [])
    data = q.data.split(":",1)[1]
    if data == "CANCEL":
        await q.edit_message_text("Operación cancelada."); return
    if data == "ALL":
        cnt = len(rules); ALERTS[chat_id] = []; await save_state()
        await q.edit_message_text(f"Se eliminaron {cnt} alertas."); return
    try: idx = int(data)
    except Exception:
        await q.edit_message_text("Acción inválida."); return
    if 0 <= idx < len(rules):
        rules.pop(idx); await save_state(); await q.edit_message_text("Alerta eliminada.")
    else:
        await q.edit_message_text("Número fuera de rango.")

async def cmd_alertas_pause(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb_pause = InlineKeyboardMarkup([
        [InlineKeyboardButton("Pausar (Indefinida)","AP:PAUSE:INF")],
        [InlineKeyboardButton("Pausar 1h","AP:PAUSE:1"),InlineKeyboardButton("Pausar 3h","AP:PAUSE:3")],
        [InlineKeyboardButton("Pausar 6h","AP:PAUSE:6"),InlineKeyboardButton("Pausar 12h","AP:PAUSE:12")],
        [InlineKeyboardButton("Pausar 24h","AP:PAUSE:24"),InlineKeyboardButton("Reanudar","AP:RESUME")],
        [InlineKeyboardButton("Cerrar","AP:CLOSE")]
    ])
    await update.effective_message.reply_text("Pausa de alertas:", reply_markup=kb_pause)

async def alerts_pause_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    chat_id = q.message.chat_id
    data = q.data
    if data == "AP:CLOSE":
        await q.edit_message_text("Listo."); return
    if data == "AP:RESUME":
        ALERTS_PAUSED.discard(chat_id); ALERTS_SILENT_UNTIL.pop(chat_id, None)
        if APPSTASH_ENABLED and storage is not None:
            try:
                await storage.alerts_resume(chat_id)
            except Exception as e:
                log.debug("No pude reanudar alertas en AppStash para %s: %s", chat_id, e)
        await q.edit_message_text("🔔 Alertas reanudadas."); return
    if data.startswith("AP:PAUSE:"):
        arg = data.split(":")[-1]
        if arg == "INF":
            ALERTS_PAUSED.add(chat_id); ALERTS_SILENT_UNTIL.pop(chat_id, None)
            if APPSTASH_ENABLED and storage is not None:
                try:
                    await storage.alerts_pause_indef(chat_id)
                except Exception as e:
                    log.debug("No pude pausar indefinidamente en AppStash para %s: %s", chat_id, e)
            await q.edit_message_text("🔕 Alertas en pausa (indefinida)."); return
        try:
            hrs = int(arg); until = datetime.now(TZ) + timedelta(hours=hrs)
            ALERTS_SILENT_UNTIL[chat_id] = until.timestamp(); ALERTS_PAUSED.discard(chat_id)
            if APPSTASH_ENABLED and storage is not None:
                try:
                    await storage.alerts_pause_until(chat_id, int(until.timestamp()))
                except Exception as e:
                    log.debug("No pude pausar temporalmente en AppStash para %s: %s", chat_id, e)
            await q.edit_message_text(f"🔕 Alertas en pausa por {hrs}h (hasta {until.strftime('%d/%m %H:%M')})."); return
        except Exception:
            await q.edit_message_text("Acción inválida."); return

async def cmd_alertas_resume(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ALERTS_PAUSED.discard(chat_id); ALERTS_SILENT_UNTIL.pop(chat_id, None)
    if APPSTASH_ENABLED and storage is not None:
        try:
            await storage.alerts_resume(chat_id)
        except Exception as e:
            log.debug("No pude reanudar alertas en AppStash para %s: %s", chat_id, e)
    await update.effective_message.reply_text("🔔 Alertas reanudadas.")

# ---- Conversación Agregar Alerta ----

def kb_submenu_fx() -> InlineKeyboardMarkup:
    return kb([
        [("Oficial","FXTYPE:oficial"),("Mayorista","FXTYPE:mayorista")],
        [("Blue","FXTYPE:blue"),("MEP","FXTYPE:mep"),("CCL","FXTYPE:ccl")],
        [("Tarjeta","FXTYPE:tarjeta"),("Cripto","FXTYPE:cripto")],
        [("Volver","BACK:KIND"),("Cancelar","CANCEL")]
    ])

def kb_submenu_metric() -> InlineKeyboardMarkup:
    return kb([
        [("Riesgo País","METRIC:riesgo")],
        [("Inflación Mensual","METRIC:inflacion")],
        [("Reservas BCRA","METRIC:reservas")],
        [("Volver","BACK:KIND"),("Cancelar","CANCEL")]
    ])

def kb_fx_side_for(t: str) -> InlineKeyboardMarkup:
    if t == "tarjeta":
        return kb([[("Venta","SIDE:venta")],[("Volver","BACK:FXTYPE"),("Cancelar","CANCEL")]])
    return kb([[("Compra","SIDE:compra"),("Venta","SIDE:venta")],[("Volver","BACK:FXTYPE"),("Cancelar","CANCEL")]])

async def alertas_add_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["al"] = {}
    k = kb([
        [("Dólares", "KIND:fx"), ("Economía", "KIND:metric")],
        [("Acciones", "KIND:acciones"), ("Cedears", "KIND:cedears")],
        [("Cancelar", "CANCEL")]
    ])
    if update.callback_query:
        q = update.callback_query; await q.answer()
        await q.edit_message_text("¿Qué querés alertar?", reply_markup=k)
    else:
        await update.effective_message.reply_text("¿Qué querés alertar?", reply_markup=k)
    return AL_KIND

async def alertas_back(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    target = q.data.split(":",1)[1]
    al = context.user_data.get("al", {})
    if target == "KIND":
        k = kb([
            [("Dólares", "KIND:fx"), ("Economía", "KIND:metric")],
            [("Acciones", "KIND:acciones"), ("Cedears", "KIND:cedears")],
            [("Cancelar", "CANCEL")]
        ])
        await q.edit_message_text("¿Qué querés alertar?", reply_markup=k); return AL_KIND
    if target == "FXTYPE":
        await q.edit_message_text("Elegí el tipo de dólar:", reply_markup=kb_submenu_fx()); return AL_FX_TYPE
    if target == "FXSIDE":
        t = al.get("type","?"); await q.edit_message_text(f"Tipo: {t.upper()}\nElegí lado:", reply_markup=kb_fx_side_for(t)); return AL_FX_SIDE
    if target == "METRIC":
        await q.edit_message_text("Elegí la métrica:", reply_markup=kb_submenu_metric()); return AL_METRIC_TYPE
    if target == "TICKERS_ACC":
        await q.edit_message_text("Elegí el ticker (Acciones .BA):", reply_markup=kb_tickers(ACCIONES_BA, "KIND", "TICK")); return AL_TICKER
    if target == "TICKERS_CEDEARS":
        await q.edit_message_text("Elegí el ticker (Cedears .BA):", reply_markup=kb_tickers(CEDEARS_BA, "KIND", "TICK")); return AL_TICKER
    if target == "OP":
        kind = al.get("kind")
        if kind == "ticker":
            kb_op = kb([[("↑ Sube", "OP:>"), ("↓ Baja", "OP:<")],[("Volver","BACK:" + ("TICKERS_ACC" if al.get("segment")=="acciones" else "TICKERS_CEDEARS")),("Cancelar","CANCEL")]])
            await q.edit_message_text("Elegí condición:", reply_markup=kb_op)
        elif kind == "fx":
            kb_op = kb([[("↑ Sube", "OP:>"), ("↓ Baja", "OP:<")],[("Volver","BACK:FXSIDE"),("Cancelar","CANCEL")]])
            await q.edit_message_text("Elegí condición:", reply_markup=kb_op)
        else:
            kb_op = kb([[("↑ Sube", "OP:>"), ("↓ Baja", "OP:<")],[("Volver","BACK:METRIC"),("Cancelar","CANCEL")]])
            await q.edit_message_text("Elegí condición:", reply_markup=kb_op)
        return AL_OP
    if target == "MODE":
        kb_mode = kb([[("Ingresar Importe", "MODE:absolute"),("Ingresar % vs valor actual", "MODE:percent")], [("Volver","BACK:OP"),("Cancelar","CANCEL")]])
        await q.edit_message_text("¿Cómo querés definir el umbral?", reply_markup=kb_mode); return AL_MODE
    await q.edit_message_text("Operación cancelada."); return ConversationHandler.END

async def alertas_add_kind(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    data = q.data
    if data == "CANCEL": await q.edit_message_text("Operación cancelada."); return ConversationHandler.END
    kind = data.split(":",1)[1]
    context.user_data["al"] = {}
    al = context.user_data["al"]
    if kind == "fx":
        al["kind"] = "fx"; await q.edit_message_text("Elegí el tipo de dólar:", reply_markup=kb_submenu_fx()); return AL_FX_TYPE
    if kind == "metric":
        al["kind"] = "metric"; await q.edit_message_text("Elegí la métrica:", reply_markup=kb_submenu_metric()); return AL_METRIC_TYPE
    if kind == "acciones":
        al["kind"] = "ticker"; al["segment"] = "acciones"
        await q.edit_message_text("Elegí el ticker (Acciones .BA):", reply_markup=kb_tickers(ACCIONES_BA, "KIND", "TICK")); return AL_TICKER
    if kind == "cedears":
        al["kind"] = "ticker"; al["segment"] = "cedears"
        await q.edit_message_text("Elegí el ticker (Cedears .BA):", reply_markup=kb_tickers(CEDEARS_BA, "KIND", "TICK")); return AL_TICKER
    await q.edit_message_text("Operación cancelada."); return ConversationHandler.END

async def alertas_add_fx_type(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    if q.data == "CANCEL": await q.edit_message_text("Operación cancelada."); return ConversationHandler.END
    if q.data.startswith("BACK:"): return await alertas_back(update, context)
    t = q.data.split(":",1)[1]
    context.user_data["al"]["type"] = t
    await q.edit_message_text(f"Tipo: {t.upper()}\nElegí lado:", reply_markup=kb_fx_side_for(t))
    return AL_FX_SIDE

async def alertas_add_fx_side(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    if q.data == "CANCEL": await q.edit_message_text("Operación cancelada."); return ConversationHandler.END
    if q.data.startswith("BACK:"): return await alertas_back(update, context)
    side = q.data.split(":",1)[1]
    context.user_data["al"]["side"] = side
    kb_op = kb([[("↑ Sube", "OP:>"), ("↓ Baja", "OP:<")],[("Volver","BACK:FXSIDE"),("Cancelar","CANCEL")]])
    await q.edit_message_text(f"Lado: {side}\nElegí condición:", reply_markup=kb_op)
    return AL_OP

async def alertas_add_metric_type(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    if q.data == "CANCEL": await q.edit_message_text("Operación cancelada."); return ConversationHandler.END
    if q.data.startswith("BACK:"): return await alertas_back(update, context)
    m = q.data.split(":",1)[1]
    context.user_data["al"]["type"] = m
    kb_op = kb([[("↑ Sube", "OP:>"), ("↓ Baja", "OP:<")],[("Volver","BACK:METRIC"),("Cancelar","CANCEL")]])
    await q.edit_message_text(f"Métrica: {m.upper()}\nElegí condición:", reply_markup=kb_op)
    return AL_OP

async def alertas_add_ticker_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    if q.data == "CANCEL": await q.edit_message_text("Operación cancelada."); return ConversationHandler.END
    if q.data.startswith("BACK:"): return await alertas_back(update, context)
    sym = q.data.split(":",1)[1].upper()
    context.user_data["al"]["symbol"] = sym
    k = kb([[("↑ Sube", "OP:>"), ("↓ Baja", "OP:<")],
            [("Volver","BACK:" + ("TICKERS_ACC" if context.user_data["al"].get("segment")=="acciones" else "TICKERS_CEDEARS")),("Cancelar","CANCEL")]])
    await q.edit_message_text(f"Ticker: {_label_long(sym)}\nElegí condición:", reply_markup=k)
    return AL_OP

async def alertas_add_op(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    if q.data == "CANCEL": await q.edit_message_text("Operación cancelada."); return ConversationHandler.END
    if q.data.startswith("BACK:"): return await alertas_back(update, context)
    op = q.data.split(":",1)[1]
    context.user_data["al"]["op"] = op
    al = context.user_data.get("al", {})
    if al.get("kind") == "ticker":
        async with ClientSession() as session:
            sym = al.get("symbol")
            metmap, _ = await metrics_for_symbols(session, [sym]) if sym else ({}, None)
            last_px = metmap.get(sym, {}).get("last_px") if metmap else None
        price_s = fmt_money_ars(last_px) if last_px is not None else "—"
        msg = (
            f"Ticker: {_label_long(sym)} | Condición: {'↑ Sube' if op=='>' else '↓ Baja'}\n"
            f"Actual: Precio {price_s}\n\n"
            "Ingresá el <b>precio objetivo</b> (solo número, sin símbolos ni separadores). Ej: 3500\n"
            "<i>Válidos: 100 | 1000.5 · Inválidos: $100, 1.000,50, 100%</i>"
        )
        await q.edit_message_text(msg, parse_mode=ParseMode.HTML); return AL_VALUE
    kb_mode = kb([[("Ingresar Importe", "MODE:absolute"),("Ingresar % vs valor actual", "MODE:percent")], [("Volver","BACK:OP"),("Cancelar","CANCEL")]])
    await q.edit_message_text("¿Cómo querés definir el umbral?", reply_markup=kb_mode)
    return AL_MODE

async def alertas_add_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    if q.data == "CANCEL": await q.edit_message_text("Operación cancelada."); return ConversationHandler.END
    if q.data.startswith("BACK:"): return await alertas_back(update, context)
    mode = q.data.split(":",1)[1]
    context.user_data["al"]["mode"] = mode
    al = context.user_data.get("al", {})
    op_text = "↑ Sube" if al.get("op")==">" else "↓ Baja"
    async with ClientSession() as session:
        if al.get("kind") == "fx":
            fx = await get_dolares(session); row = fx.get(al.get("type",""), {}) or {}
            cur = _fx_display_value(row, al.get("side","venta"))
            cur_s = fmt_money_ars(cur) if cur is not None else "—"
            if mode == "percent":
                msg = (f"Tipo: {al.get('type','?').upper()} | Lado: {al.get('side','?')} | Condición: {op_text}\n"
                       f"Ahora: {cur_s}\n\nIngresá el <b>%</b> (solo número). Ej: 10 | 7.5")
            else:
                msg = (f"Tipo: {al.get('type','?').upper()} | Lado: {al.get('side','?')} | Condición: {op_text}\n"
                       f"Ahora: {cur_s}\n\nIngresá el <b>importe</b> AR$ (solo número). Ej: 1580 | 25500")
            await q.edit_message_text(msg, parse_mode=ParseMode.HTML); return AL_VALUE
        if al.get("kind") == "metric":
            rp = await get_riesgo_pais(session); infl = await get_inflacion_mensual(session); rv = await get_reservas_lamacro(session)
            curmap = {"riesgo": (f"{rp[0]:.0f} pb" if rp else "—", rp[0] if rp else None, "pb"),
                      "inflacion": ((str(round(infl[0],1)).replace('.',','))+"%" if infl else "—", infl[0] if infl else None, "%"),
                      "reservas": (f"{fmt_number(rv[0],0)} MUS$" if rv else "—", rv[0] if rv else None, "MUS$")}
            label, curval, unidad = curmap.get(al.get("type",""), ("—", None, ""))
            if mode == "percent":
                msg = (f"Métrica: {al.get('type','?').upper()} | Condición: {op_text}\nAhora: {label}\n\nIngresá el <b>%</b> (solo número).")
            else:
                msg = (f"Métrica: {al.get('type','?').upper()} | Condición: {op_text}\nAhora: {label}\n\nIngresá el <b>importe</b> (solo número, en {unidad}).")
            await q.edit_message_text(msg, parse_mode=ParseMode.HTML); return AL_VALUE
    await q.edit_message_text("Operación cancelada."); return ConversationHandler.END

async def alertas_add_value(update: Update, context: ContextTypes.DEFAULT_TYPE):
    al = context.user_data.get("al", {})
    val = _parse_float_user_strict(update.message.text)
    if val is None:
        await update.message.reply_text("Ingresá solo número (sin $ ni % ni separadores)."); return AL_VALUE
    chat_id = update.effective_chat.id
    async with ClientSession() as session:
        if al.get("kind") == "fx":
            fx = await get_dolares(session); row = fx.get(al["type"], {}) or {}
            cur = _fx_display_value(row, al.get("side","venta"))
            if cur is None:
                await update.message.reply_text("No pude leer el valor actual."); return ConversationHandler.END
            thr = cur*(1 + (val/100.0)) if al.get("mode")=="percent" and al["op"] == ">" else \
                  cur*(1 - (val/100.0)) if al.get("mode")=="percent" else val
            if al.get("mode") == "absolute":
                if (al["op"] == ">" and thr <= cur) or (al["op"] == "<" and thr >= cur):
                    await update.message.reply_text(f"El objetivo debe ser {'mayor' if al['op']=='>' else 'menor'} que {fmt_money_ars(cur)}."); return AL_VALUE
            rule = {"kind":"fx","type":al["type"],"side":al["side"],"op":al["op"],"value":float(thr),"created_at":int(time())}
            ALERTS.setdefault(chat_id, []).append(rule)
            await save_state()
            await update.message.reply_text("Listo. Alerta agregada ✅"); return ConversationHandler.END

        if al.get("kind") == "metric":
            rp = await get_riesgo_pais(session); infl = await get_inflacion_mensual(session); rv = await get_reservas_lamacro(session)
            curmap = {"riesgo": float(rp[0]) if rp else None, "inflacion": float(infl[0]) if infl else None, "reservas": rv[0] if rv else None}
            cur = curmap.get(al["type"])
            if cur is None:
                await update.message.reply_text("No pude leer el valor actual."); return ConversationHandler.END
            thr = cur*(1 + (val/100.0)) if al.get("mode")=="percent" and al["op"] == ">" else \
                  cur*(1 - (val/100.0)) if al.get("mode")=="percent" else val
            if al.get("mode") == "absolute":
                if (al["op"] == ">" and thr <= cur) or (al["op"] == "<" and thr >= cur):
                    await update.message.reply_text("El objetivo debe ser válido respecto al valor actual."); return AL_VALUE
            rule = {"kind":"metric","type":al["type"],"op":al["op"],"value":float(thr),"created_at":int(time())}
            ALERTS.setdefault(chat_id, []).append(rule)
            await save_state()
            await update.message.reply_text("Listo. Alerta agregada ✅"); return ConversationHandler.END

        # ticker
        sym, op = al.get("symbol"), al.get("op")
        metmap, _ = await metrics_for_symbols(session, [sym])
        last_px = metmap.get(sym, {}).get("last_px")
        if last_px is None:
            await update.message.reply_text("No pude leer el precio actual."); return ConversationHandler.END
        thr = val
        if (op == ">" and thr <= last_px) or (op == "<" and thr >= last_px):
            await update.message.reply_text(f"El precio objetivo debe ser {'mayor' if op=='>' else 'menor'} que {fmt_money_ars(last_px)}."); return AL_VALUE
        rule = {"kind":"ticker","symbol":sym,"op":op,"value":float(thr),"mode":"absolute","created_at":int(time())}
        ALERTS.setdefault(chat_id, []).append(rule)
        await save_state()
        await update.message.reply_text("Listo. Alerta agregada ✅"); return ConversationHandler.END

# ============================ LOOP ALERTAS ============================

async def alerts_loop(app: Application):
    await asyncio.sleep(5)
    timeout = ClientTimeout(total=12)
    while True:
        try:
            now_ts = datetime.now(TZ).timestamp()
            active_chats = []
            for cid, rules in ALERTS.items():
                if not rules: continue
                if cid in ALERTS_PAUSED: continue
                if cid in ALERTS_SILENT_UNTIL and ALERTS_SILENT_UNTIL[cid] > now_ts: continue
                active_chats.append(cid)
            if active_chats:
                async with ClientSession(timeout=timeout) as session:
                    fx = await get_dolares(session)
                    rp = await get_riesgo_pais(session)
                    infl = await get_inflacion_mensual(session)
                    rv = await get_reservas_lamacro(session)
                    vals = {"riesgo": float(rp[0]) if rp else None,
                            "inflacion": float(infl[0]) if infl else None,
                            "reservas": rv[0] if rv else None}
                    sym_list = {r["symbol"] for cid in active_chats for r in ALERTS.get(cid, []) if r.get("kind")=="ticker" and r.get("symbol")}
                    metmap, _ = (await metrics_for_symbols(session, sorted(sym_list))) if sym_list else ({}, None)
                    for chat_id in active_chats:
                        rules = ALERTS.get(chat_id, [])
                        if not rules: continue
                        trig = []
                        for r in rules:
                            if r.get("kind") == "fx":
                                row = fx.get(r["type"], {}) or {}
                                cur = _fx_display_value(row, r["side"])
                                if cur is None: continue
                                ok = (cur > r["value"]) if r["op"] == ">" else (cur < r["value"])
                                if ok: trig.append(("fx", r["type"], r["side"], r["op"], r["value"], cur))
                            elif r.get("kind") == "metric":
                                cur = vals.get(r["type"])
                                if cur is None: continue
                                ok = (cur > r["value"]) if r["op"] == ">" else (cur < r["value"])
                                if ok: trig.append(("metric", r["type"], r["op"], r["value"], cur))
                            elif r.get("kind") == "ticker":
                                sym = r["symbol"]; m = metmap.get(sym, {}); cur = m.get("last_px"); 
                                if cur is None: continue
                                ok = (cur > r["value"]) if r["op"] == ">" else (cur < r["value"])
                                if ok: trig.append(("ticker_px", sym, r["op"], r["value"], cur))
                        if trig:
                            lines = [f"<b>🔔 Alertas</b>"]
                            for t, *rest in trig:
                                if t == "fx":
                                    tipo, side, op, v, cur = rest
                                    lines.append(f"{tipo.upper()} ({side}): {fmt_money_ars(cur)} ({html_op(op)} {fmt_money_ars(v)})")
                                elif t == "metric":
                                    tipo, op, v, cur = rest
                                    if tipo=="riesgo":
                                        lines.append(f"Riesgo País: {cur:.0f} pb ({html_op(op)} {v:.0f} pb)")
                                    elif tipo=="inflacion":
                                        lines.append(f"Inflación Mensual: {str(round(cur,1)).replace('.',',')}% ({html_op(op)} {str(round(v,1)).replace('.',',')}%)")
                                    elif tipo=="reservas":
                                        lines.append(f"Reservas: {fmt_number(cur,0)} MUS$ ({html_op(op)} {fmt_number(v,0)} MUS$)")
                                else:
                                    sym, op, v, cur = rest
                                    lines.append(f"{_label_long(sym)} (Precio): {fmt_money_ars(cur)} ({html_op(op)} {fmt_money_ars(v)})")
                            try:
                                await app.bot.send_message(chat_id, "\n".join(lines))
                            except Exception as e:
                                log.warning("send alert failed %s: %s", chat_id, e)
            await asyncio.sleep(600)
        except Exception as e:
            log.warning("alerts_loop error: %s", e)
            await asyncio.sleep(30)

# ============================ SUSCRIPCIONES ============================

SUBS_SET_TIME = range(1)

async def _job_send_daily(context: ContextTypes.DEFAULT_TYPE):
    chat_id = context.job.chat_id
    try:
        await cmd_resumen_diario(Update.de_json({"message":{"chat":{"id":chat_id}}}, context.bot), context)
    except Exception as e:
        log.warning("send daily failed %s: %s", chat_id, e)

def _job_name_daily(chat_id: int) -> str: return f"daily_{chat_id}"

def _schedule_daily_for_chat(app: Application, chat_id: int, hhmm: str):
    for j in app.job_queue.get_jobs_by_name(_job_name_daily(chat_id)): j.schedule_removal()
    h, m = [int(x) for x in hhmm.split(":")]
    app.job_queue.run_daily(_job_send_daily, time=dtime(hour=h, minute=m, tzinfo=TZ), chat_id=chat_id, name=_job_name_daily(chat_id))

def _schedule_all_subs(app: Application):
    for chat_id, conf in SUBS.items():
        hhmm = conf.get("daily")
        if hhmm: _schedule_daily_for_chat(app, chat_id, hhmm)

def kb_times_full() -> InlineKeyboardMarkup:
    rows, row = [], []
    for h in range(24):
        label = f"{h:02d}:00"; row.append((label, f"SUBS:T:{label}"))
        if len(row) == 4: rows.append(row); row = []
    if row: rows.append(row)
    rows.append([("Desuscribirme","SUBS:OFF"),("Cerrar","SUBS:CLOSE")])
    return kb(rows)

async def cmd_subs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    cur = SUBS.get(chat_id, {}).get("daily")
    txt = f"<b>📬 Suscripción</b>\nResumen Diario: {'ON ('+cur+')' if cur else 'OFF'}\nElegí un horario (hora AR):"
    await update.effective_message.reply_text(txt, reply_markup=kb_times_full())
    return SUBS_SET_TIME

async def subs_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    chat_id = q.message.chat_id; data = q.data
    if data == "SUBS:CLOSE": await q.edit_message_text("Listo."); return ConversationHandler.END
    if data == "SUBS:OFF":
        if chat_id in SUBS and SUBS[chat_id].get("daily"):
            SUBS[chat_id]["daily"] = None; await save_state()
            for j in context.application.job_queue.get_jobs_by_name(_job_name_daily(chat_id)): j.schedule_removal()
        await q.edit_message_text("Suscripción cancelada."); return ConversationHandler.END
    if data.startswith("SUBS:T:"):
        hhmm = data.split(":",2)[2]
        SUBS.setdefault(chat_id, {})["daily"] = hhmm; await save_state()
        _schedule_daily_for_chat(context.application, chat_id, hhmm)
        await q.edit_message_text(f"Te suscribí al Resumen Diario a las {hhmm} (hora AR)."); return ConversationHandler.END
    await q.edit_message_text("Acción inválida."); return ConversationHandler.END

# ============================ PORTAFOLIO (salida debajo del menú + torta) ============================

def pf_get(chat_id: int) -> Dict[str, Any]:
    return PF.setdefault(chat_id, {"base": {"moneda":"ARS", "tc":"mep"}, "monto": 0.0, "items": []})

def kb_pf_main() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Ayuda", callback_data="PF:HELP")],
        [InlineKeyboardButton("Fijar base", callback_data="PF:SETBASE"), InlineKeyboardButton("Fijar monto", callback_data="PF:SETMONTO")],
        [InlineKeyboardButton("Agregar instrumento", callback_data="PF:ADD")],
        [InlineKeyboardButton("Ver composición", callback_data="PF:LIST"), InlineKeyboardButton("Editar instrumento", callback_data="PF:EDIT")],
        [InlineKeyboardButton("Rendimiento", callback_data="PF:RET"), InlineKeyboardButton("Proyección", callback_data="PF:PROJ")],
        [InlineKeyboardButton("Eliminar portafolio", callback_data="PF:CLEAR")],
    ])

def kb_pick_generic(symbols: List[str], back: str, prefix: str) -> InlineKeyboardMarkup:
    rows = []; row = []
    for s in symbols:
        label = _label_long(s)
        row.append((label, f"{prefix}:{s}"))
        if len(row) == 2: rows.append(row); row = []
    if row: rows.append(row)
    rows.append([("Volver","PF:ADD")])
    return InlineKeyboardMarkup([[InlineKeyboardButton(t, callback_data=d) for t,d in r] for r in rows])

async def cmd_portafolio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text("📦 Menú Portafolio", reply_markup=kb_pf_main())

# --- helper para mandar "debajo del menú" ---
async def _send_below_menu(context: ContextTypes.DEFAULT_TYPE, chat_id: int, text: Optional[str]=None, photo_bytes: Optional[bytes]=None, reply_markup=None):
    if text:
        await context.bot.send_message(chat_id, text, parse_mode=ParseMode.HTML, reply_markup=reply_markup, disable_web_page_preview=True)
    elif photo_bytes:
        await context.bot.send_photo(chat_id, photo=photo_bytes, caption=None, reply_markup=reply_markup)

async def _pf_total_usado(chat_id: int) -> float:
    pf = pf_get(chat_id)
    total = 0.0
    for it in pf["items"]:
        if it.get("importe") is not None:
            total += float(it["importe"])
    return total

async def pf_menu_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    chat_id = q.message.chat_id; data = q.data

    if data == "PF:HELP":
        txt = ("<b>Cómo armar tu portafolio</b>\n\n"
               "1) Fijá base y tipo de cambio.\n2) Definí el monto total (solo número).\n"
               "3) Agregá instrumentos (por cantidad, importe o % del monto).\n"
               "4) Ver composición y editar.\n5) Rendimiento (actual) y Proyección (3/6M).\n\n"
               "<i>Formato de números: solo dígitos y decimal. Sin $ ni % ni comas.</i>")
        await _send_below_menu(context, chat_id, text=txt); return

    if data == "PF:SETBASE":
        kb_base = InlineKeyboardMarkup([
            [InlineKeyboardButton("ARS / MEP", callback_data="PF:BASE:ARS:mep"),
             InlineKeyboardButton("ARS / CCL", callback_data="PF:BASE:ARS:ccl")],
            [InlineKeyboardButton("ARS / Oficial", callback_data="PF:BASE:ARS:oficial"),
             InlineKeyboardButton("USD / Oficial", callback_data="PF:BASE:USD:oficial")],
            [InlineKeyboardButton("USD / MEP", callback_data="PF:BASE:USD:mep"),
             InlineKeyboardButton("USD / CCL", callback_data="PF:BASE:USD:ccl")],
            [InlineKeyboardButton("Volver", callback_data="PF:BACK")]
        ])
        await q.edit_message_text("Elegí base del portafolio:", reply_markup=kb_base); return

    if data.startswith("PF:BASE:"):
        _,_,mon,tc = data.split(":")
        pf = pf_get(chat_id)
        pf["base"] = {"moneda": mon, "tc": tc}
        await save_state()
        msg = f"Base fijada: {mon.upper()} / {tc.upper()}"
        # re-mostramos el menú principal (edit) y además una confirmación DEBAJO
        await q.edit_message_text("📦 Menú Portafolio", reply_markup=kb_pf_main())
        await _send_below_menu(context, chat_id, text=msg); 
        return

    if data == "PF:SETMONTO":
        context.user_data["pf_mode"] = "set_monto"
        await _send_below_menu(context, chat_id, text="Ingresá el <b>monto total</b> (solo número)."); return

    if data == "PF:ADD":
        kb_add = InlineKeyboardMarkup([
            [InlineKeyboardButton("Acción (.BA, ARS)", callback_data="PF:ADD:accion"),
             InlineKeyboardButton("Cedear (.BA, ARS)", callback_data="PF:ADD:cedear")],
            [InlineKeyboardButton("Bono (ARS/USD)", callback_data="PF:ADD:bono"),
             InlineKeyboardButton("FCI (ARS/USD)", callback_data="PF:ADD:fci")],
            [InlineKeyboardButton("Letras (ARS/USD)", callback_data="PF:ADD:lete"),
             InlineKeyboardButton("Cripto (USD)", callback_data="PF:ADD:cripto")],
            [InlineKeyboardButton("Volver", callback_data="PF:BACK")]
        ])
        await q.edit_message_text("¿Qué querés agregar?", reply_markup=kb_add); return

    if data.startswith("PF:ADD:"):
        tipo = data.split(":")[2]
        context.user_data["pf_add_tipo"] = tipo
        if tipo == "accion":
            await q.edit_message_text("Elegí la acción:", reply_markup=kb_pick_generic(ACCIONES_BA, "PF:ADD", "PF:PICK"))
        elif tipo == "cedear":
            await q.edit_message_text("Elegí el cedear:", reply_markup=kb_pick_generic(CEDEARS_BA, "PF:ADD", "PF:PICK"))
        elif tipo == "bono":
            await q.edit_message_text("Elegí el bono:", reply_markup=kb_pick_generic(BONOS_AR, "PF:ADD", "PF:PICK"))
        elif tipo == "fci":
            await q.edit_message_text("Elegí el FCI:", reply_markup=kb_pick_generic(FCI_LIST, "PF:ADD", "PF:PICK"))
        elif tipo == "lete":
            await q.edit_message_text("Elegí la Letra:", reply_markup=kb_pick_generic(LETES_LIST, "PF:ADD", "PF:PICK"))
        else:
            await q.edit_message_text("Elegí la cripto:", reply_markup=kb_pick_generic(CRIPTO_TOP_NAMES, "PF:ADD", "PF:PICK"))
        return

    if data.startswith("PF:PICK:"):
        sym = data.split(":")[2]
        if sym in CRIPTO_TOP_NAMES:
            context.user_data["pf_add_simbolo"] = _crypto_to_symbol(sym)
            sel_label = _label_long(sym)
        else:
            context.user_data["pf_add_simbolo"] = sym
            sel_label = _label_long(sym)
        kb_ask = InlineKeyboardMarkup([
            [InlineKeyboardButton("Por cantidad", callback_data="PF:ADDQTY"), InlineKeyboardButton("Por importe", callback_data="PF:ADDAMT")],
            [InlineKeyboardButton("Por % del monto", callback_data="PF:ADDPCT")],
            [InlineKeyboardButton("Volver", callback_data="PF:ADD")]
        ])
        await _send_below_menu(context, chat_id, text=f"Seleccionado: {sel_label}\n¿Cómo cargar?")
        await q.edit_message_reply_markup(reply_markup=kb_ask)  # actualiza solo los botones
        return

    if data == "PF:ADDQTY":
        context.user_data["pf_mode"] = "pf_add_qty"
        await _send_below_menu(context, chat_id, text="Ingresá la <b>cantidad</b> (solo número)."); return
    if data == "PF:ADDAMT":
        context.user_data["pf_mode"] = "pf_add_amt"
        await _send_below_menu(context, chat_id, text="Ingresá el <b>importe</b> (solo número)."); return
    if data == "PF:ADDPCT":
        context.user_data["pf_mode"] = "pf_add_pct"
        await _send_below_menu(context, chat_id, text="Ingresá el <b>porcentaje</b> del monto (solo número). Ej: 10 = 10%"); return

    if data == "PF:LIST":
        await pf_send_composition(context, chat_id)
        return

    if data == "PF:EDIT":
        pf = pf_get(chat_id)
        if not pf["items"]:
            await _send_below_menu(context, chat_id, text="No hay instrumentos para editar."); return
        buttons = []
        for i,it in enumerate(pf["items"],1):
            label = f"{i}. " + (_label_long(it['simbolo']) if it.get("simbolo") else it.get("tipo","").upper())
            buttons.append([InlineKeyboardButton(label, callback_data=f"PF:EDIT:{i-1}")])
        buttons.append([InlineKeyboardButton("Volver", callback_data="PF:BACK")])
        # Menú principal queda como está; enviamos menú de edición como mensaje aparte
        await _send_below_menu(context, chat_id, text="Elegí instrumento a editar:")
        await context.bot.send_message(chat_id, " ", reply_markup=InlineKeyboardMarkup(buttons))
        return

    if data.startswith("PF:EDIT:"):
        idx = int(data.split(":")[2])
        context.user_data["pf_edit_idx"] = idx
        kb_ed = InlineKeyboardMarkup([
            [InlineKeyboardButton("+ Cantidad", callback_data="PF:ED:ADDQ"), InlineKeyboardButton("- Cantidad", callback_data="PF:ED:SUBQ")],
            [InlineKeyboardButton("Cambiar importe", callback_data="PF:ED:AMT")],
            [InlineKeyboardButton("Eliminar este", callback_data="PF:ED:DEL")],
            [InlineKeyboardButton("Volver", callback_data="PF:EDIT")]
        ])
        await _send_below_menu(context, chat_id, text="¿Qué querés hacer?")
        await context.bot.send_message(chat_id, " ", reply_markup=kb_ed)
        return

    if data == "PF:ED:ADDQ":
        context.user_data["pf_mode"] = "edit_addq"
        await _send_below_menu(context, chat_id, text="Ingresá la <b>cantidad a sumar</b>."); return
    if data == "PF:ED:SUBQ":
        context.user_data["pf_mode"] = "edit_subq"
        await _send_below_menu(context, chat_id, text="Ingresá la <b>cantidad a restar</b>."); return
    if data == "PF:ED:AMT":
        context.user_data["pf_mode"] = "edit_amt"
        await _send_below_menu(context, chat_id, text="Ingresá el <b>nuevo importe</b> (moneda BASE)."); return
    if data == "PF:ED:DEL":
        pf = pf_get(chat_id); idx = context.user_data.get("pf_edit_idx", -1)
        if 0 <= idx < len(pf["items"]):
            pf["items"].pop(idx); await save_state()
            await _send_below_menu(context, chat_id, text="Instrumento eliminado."); return
        await _send_below_menu(context, chat_id, text="Índice inválido."); return

    if data == "PF:RET":
        await pf_show_return_below(context, chat_id)
        return
    if data == "PF:PROJ":
        await pf_show_projection_below(context, chat_id)
        return

    if data == "PF:CLEAR":
        PF.pop(chat_id, None); await save_state()
        await _send_below_menu(context, chat_id, text="Portafolio eliminado.")
        await q.edit_message_text("📦 Menú Portafolio", reply_markup=kb_pf_main());
        return

    if data == "PF:BACK":
        await q.edit_message_text("📦 Menú Portafolio", reply_markup=kb_pf_main()); return

def _parse_num_text(s: str) -> Optional[float]:
    if re.search(r"[^\d\.,\-+]", s): return None
    s2 = s.replace(".","").replace(",",".")
    try: return float(s2)
    except: return None

async def pf_text_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    mode = context.user_data.get("pf_mode")
    if not mode: return
    text = (update.message.text or "").strip()
    pf = pf_get(chat_id)

    def _restante_str(usado: float) -> str:
        pf_base = pf["base"]["moneda"].upper()
        return (fmt_money_ars if pf_base=="ARS" else fmt_money_usd)(max(0.0, pf["monto"] - usado))

    # Monto total
    if mode == "set_monto":
        v = _parse_num_text(text)
        if v is None:
            await update.message.reply_text("Ingresá solo número (sin símbolos)."); return
        pf["monto"] = float(v); await save_state()
        usado = await _pf_total_usado(chat_id)
        pf_base = pf["base"]["moneda"].upper()
        f_money = fmt_money_ars if pf_base=="ARS" else fmt_money_usd
        await update.message.reply_text(f"Monto fijado: {f_money(v)} · Restante: {_restante_str(usado)}")
        context.user_data["pf_mode"]=None; return

    # Alta por cantidad/importe/% (símbolo ya elegido)
    if mode in ("pf_add_qty","pf_add_amt","pf_add_pct"):
        v = _parse_num_text(text)
        if v is None:
            await update.message.reply_text("Ingresá solo número (sin símbolos)."); return

        tipo = context.user_data.get("pf_add_tipo")
        sym = context.user_data.get("pf_add_simbolo","")
        yfsym = sym

        price_native = None  # precio en MONEDA NATIVA
        async with ClientSession() as session:
            if yfsym.endswith(".BA") or yfsym.endswith("-USD"):
                mets, _ = await metrics_for_symbols(session, [yfsym])
                price_native = mets.get(yfsym,{}).get("last_px")
            tc_key = (pf_get(chat_id)["base"].get("tc") or "oficial").lower()
            tc_val = await get_tc_value(session, tc_key)

        pf_base = pf_get(chat_id)["base"]["moneda"].upper()  # ARS o USD
        inst_moneda = "USD" if yfsym.endswith("-USD") else "ARS"

        # Precio expresado en MONEDA BASE
        price_base = None
        if price_native is not None:
            if pf_base == inst_moneda:
                price_base = float(price_native)
            else:
                if tc_val and tc_val > 0:
                    if pf_base == "ARS" and inst_moneda == "USD":
                        price_base = float(price_native) * float(tc_val)
                    elif pf_base == "USD" and inst_moneda == "ARS":
                        price_base = float(price_native) / float(tc_val)

        cantidad, importe_base = None, None

        if mode == "pf_add_qty":
            cantidad = float(v)
            if requires_integer_units(yfsym): cantidad = math.floor(cantidad)
            if price_base is not None: importe_base = float(cantidad) * float(price_base)

        elif mode == "pf_add_amt":
            importe_base = float(v)  # EN MONEDA BASE
            if price_base and price_base > 0:
                raw_qty = importe_base / float(price_base)
                if requires_integer_units(yfsym):
                    cantidad = float(math.floor(raw_qty))
                    importe_base = float(cantidad) * float(price_base)
                else:
                    cantidad = round(raw_qty, 6)

        else:  # pf_add_pct
            if pf["monto"] <= 0:
                await update.message.reply_text("Primero fijá el monto total del portafolio."); return
            pct_val = max(0.0, float(v))
            importe_base = round(pf["monto"] * pct_val / 100.0, 2)
            if price_base and price_base > 0:
                raw_qty = importe_base / float(price_base)
                if requires_integer_units(yfsym):
                    cantidad = float(math.floor(raw_qty))
                    importe_base = float(cantidad) * float(price_base)
                else:
                    cantidad = round(raw_qty, 6)

        usado_pre = await _pf_total_usado(chat_id)
        add_val = float(importe_base or 0.0)
        if pf["monto"] > 0 and (usado_pre + add_val) > pf["monto"] + 1e-6:
            await update.message.reply_text(f"🚫 Te pasás del presupuesto. Restante: {_restante_str(usado_pre)}"); return

        item = {"tipo":tipo, "simbolo": yfsym if yfsym else sym}
        if cantidad is not None: item["cantidad"] = float(cantidad)
        if importe_base is not None: item["importe"] = float(importe_base)  # en MONEDA BASE
        pf["items"].append(item); await save_state()

        pf_base = pf["base"]["moneda"].upper()
        qty_str = ""
        if cantidad is not None:
            qty_str = f"(cant: {int(cantidad)}) " if requires_integer_units(yfsym) else f"(cant: {cantidad}) "
        unit_px_str = ""
        if price_base is not None:
            unit_px_str = f"a {(fmt_money_ars(price_base) if pf_base=='ARS' else fmt_money_usd(price_base))} c/u "
        total_str = ""
        if importe_base is not None:
            total_str = f"(= {(fmt_money_ars(importe_base) if pf_base=='ARS' else fmt_money_usd(importe_base))})"
        usado_post = await _pf_total_usado(chat_id)
        restante_str = _restante_str(usado_post)
        base_label = _label_long(sym if not sym.endswith("-USD") else sym.replace("-USD"," (USD)"))
        det = f"Agregado {base_label} {qty_str}{unit_px_str}{total_str}.\nRestante: {restante_str}"
        await update.message.reply_text(det)
        context.user_data["pf_mode"]=None; return

    # Ediciones
    if mode in ("edit_addq","edit_subq","edit_amt"):
        v = _parse_num_text(text)
        if v is None:
            await update.message.reply_text("Ingresá solo número (sin símbolos)."); return
        idx = context.user_data.get("pf_edit_idx", -1)
        if not (0 <= idx < len(pf["items"])):
            await update.message.reply_text("Índice inválido."); context.user_data["pf_mode"]=None; return
        it = pf["items"][idx]

        yfsym = it.get("simbolo")
        async with ClientSession() as session:
            if yfsym and (yfsym.endswith(".BA") or yfsym.endswith("-USD")):
                mets, _ = await metrics_for_symbols(session, [yfsym])
                px = mets.get(yfsym,{}).get("last_px")
            else:
                px = None
            tc_key = (pf_get(chat_id)["base"].get("tc") or "oficial").lower()
            tc_val = await get_tc_value(session, tc_key)

        pf_base = pf_get(chat_id)["base"]["moneda"].upper()
        inst_moneda = "USD" if yfsym and yfsym.endswith("-USD") else "ARS"
        price_base = None
        if px is not None:
            if pf_base == inst_moneda:
                price_base = float(px)
            else:
                if tc_val and tc_val > 0:
                    if pf_base == "ARS" and inst_moneda == "USD":
                        price_base = float(px) * float(tc_val)
                    elif pf_base == "USD" and inst_moneda == "ARS":
                        price_base = float(px) / float(tc_val)

        if mode == "edit_amt":
            nuevo_importe = float(v)  # en MONEDA BASE
            usado_pre = await _pf_total_usado(chat_id)
            usado_sin = usado_pre - float(it.get("importe") or 0.0)
            if pf["monto"] > 0 and (usado_sin + nuevo_importe) > pf["monto"] + 1e-6:
                restante = pf["monto"] - usado_sin
                f_money = fmt_money_ars if pf_base=="ARS" else fmt_money_usd
                await update.message.reply_text(f"🚫 Te pasás del presupuesto. Restante: {f_money(max(0.0, restante))}")
                return
            it["importe"] = nuevo_importe
            if price_base and price_base > 0:
                raw_qty = nuevo_importe/price_base
                it["cantidad"] = float(math.floor(raw_qty)) if requires_integer_units(yfsym) else round(raw_qty, 6)
        else:
            delta = float(v) if mode=="edit_addq" else -float(v)
            cur = float(it.get("cantidad") or 0.0)
            nueva_cant = cur + delta
            if requires_integer_units(yfsym): nueva_cant = float(max(0, math.floor(nueva_cant)))
            else: nueva_cant = max(0.0, nueva_cant)
            if price_base and price_base > 0:
                nuevo_importe = nueva_cant * float(price_base)
                usado_pre = await _pf_total_usado(chat_id)
                delta_importe = nuevo_importe - float((cur*price_base) if price_base else 0.0)
                if pf["monto"] > 0 and (usado_pre + delta_importe) > pf["monto"] + 1e-6:
                    restante = pf["monto"] - usado_pre
                    f_money = fmt_money_ars if pf_base=="ARS" else fmt_money_usd
                    await update.message.reply_text(f"🚫 Te pasás del presupuesto. Restante: {f_money(max(0.0, restante))}")
                    return
                it["importe"] = nuevo_importe
            it["cantidad"] = nueva_cant

        await save_state()
        usado = await _pf_total_usado(chat_id)
        f_money = fmt_money_ars if pf_base=="ARS" else fmt_money_usd
        await update.message.reply_text("Actualizado ✅ · Restante: " + f_money(max(0.0, pf["monto"]-usado)))
        context.user_data["pf_mode"]=None; return

# --- Composición: texto + torta (debajo del menú) ---

def _pie_image_from_items(pf: Dict[str, Any]) -> Optional[bytes]:
    vals = []
    labels = []
    total = 0.0
    for it in pf["items"]:
        v = float(it.get("importe") or 0.0)
        if v > 0:
            labels.append(_label_short(it.get("simbolo","")))
            vals.append(v)
            total += v
    if total <= 0: return None
    # combinar menores a 3% como "Otros"
    vals2, labels2 = [], []
    otros = 0.0
    for v,l in zip(vals, labels):
        if v/total < 0.03:
            otros += v
        else:
            vals2.append(v); labels2.append(l)
    if otros > 0:
        vals2.append(otros); labels2.append("Otros")
    fig = plt.figure(figsize=(5,5), dpi=160)
    plt.pie(vals2, labels=labels2, autopct=lambda p: f"{p:.1f}%" if p >= 3 else "")
    plt.title("Composición del Portafolio")
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.read()

async def pf_send_composition(context: ContextTypes.DEFAULT_TYPE, chat_id: int):
    pf = pf_get(chat_id)
    pf_base = pf["base"]["moneda"].upper()
    f_money = fmt_money_ars if pf_base=="ARS" else fmt_money_usd
    if not pf["items"]:
        await _send_below_menu(context, chat_id, text="Tu portafolio está vacío. Usá «Agregar instrumento»."); return
    lines = [f"<b>Portafolio</b> — Base: {pf['base']['moneda'].upper()}/{pf['base']['tc'].upper()}",
             f"Monto objetivo: {f_money(pf['monto'])}"]
    for i,it in enumerate(pf["items"],1):
        desc = f"{i}. "
        if it.get("simbolo"): desc += f"{_label_long(it['simbolo'])} [{it['tipo'].upper()}]"
        else: desc += it.get("tipo","").upper()
        if it.get("cantidad") is not None:
            desc += f" | Cant: {int(it['cantidad']) if it.get('simbolo','').endswith('.BA') else it['cantidad']}"
        if it.get("importe") is not None:
            desc += f" | Importe(Base): {f_money(it['importe'])}"
        lines.append(desc)
    usado = await _pf_total_usado(chat_id)
    lines.append(f"\nUsado (Base): {f_money(usado)} · Restante: {f_money(max(0.0, pf['monto']-usado))}")
    await _send_below_menu(context, chat_id, text="\n".join(lines))
    # torta
    img = _pie_image_from_items(pf)
    if img:
        await _send_below_menu(context, chat_id, photo_bytes=img)

# --- Rendimiento (debajo del menú) ---

async def pf_show_return_below(context: ContextTypes.DEFAULT_TYPE, chat_id: int):
    pf = pf_get(chat_id)
    if not pf["items"]:
        await _send_below_menu(context, chat_id, text="Tu portafolio está vacío. Agregá instrumentos primero."); return
    pf_base = pf["base"]["moneda"].upper()
    f_money = fmt_money_ars if pf_base=="ARS" else fmt_money_usd
    # Nota: como guardamos importes en base, usamos eso como proxy del valor actual (conservador)
    total_invertido = 0.0; total_actual = 0.0
    lines = ["<b>📈 Rendimiento del portafolio (aprox.)</b>"]
    for it in pf["items"]:
        simb = it.get("simbolo","")
        inv = float(it.get("importe") or 0.0)
        val_act = inv
        total_invertido += inv; total_actual += val_act
        delta = val_act - inv
        r = (delta / inv * 100.0) if inv > 0 else 0.0
        lines.append(f"• {_label_long(simb)}: {f_money(val_act)} ({pct(r,2)})")
    delta_t = total_actual - total_invertido
    r_t = (delta_t/total_invertido*100.0) if total_invertido>0 else 0.0
    lines.append(f"\nInvertido: {f_money(total_invertido)}")
    lines.append(f"Valor actual: {f_money(total_actual)}")
    lines.append(f"Variación: {f_money(delta_t)} ({pct(r_t,2)})")
    await _send_below_menu(context, chat_id, text="\n".join(lines))

# --- Proyección (debajo del menú) ---

async def pf_show_projection_below(context: ContextTypes.DEFAULT_TYPE, chat_id: int):
    pf = pf_get(chat_id)
    if not pf["items"]:
        await _send_below_menu(context, chat_id, text="Tu portafolio está vacío. Agregá instrumentos primero."); return
    async with ClientSession() as session:
        syms = [it["simbolo"] for it in pf["items"] if it.get("simbolo") and (it["simbolo"].endswith(".BA") or it["simbolo"].endswith("-USD"))]
        syms = sorted(set(syms))
        mets, last_ts = await metrics_for_symbols(session, syms) if syms else ({}, None)
        valores = []; total_val = 0.0
        for it in pf["items"]:
            s = it.get("simbolo","")
            v = float(it.get("importe") or 0.0)
            valores.append((s, v)); total_val += v
        if total_val <= 0:
            await _send_below_menu(context, chat_id, text="Sin valores suficientes para proyectar."); return
        w3 = 0.0; w6 = 0.0; detail = []
        for s, v in valores:
            w = v/total_val
            m = mets.get(s, {})
            p3 = projection_3m(m) if m else 0.0
            p6 = projection_6m(m) if m else 0.0
            w3 += w*p3; w6 += w*p6
            if s in mets:
                detail.append(f"• {_label_short(s)} → 3M {pct(p3,1)} | 6M {pct(p6,1)} (peso {pct(w*100,1)})")
        fecha = datetime.fromtimestamp(last_ts, TZ).strftime("%d/%m/%Y") if last_ts else None
        lines = [f"<b>🔮 Proyección del Portafolio</b>" + (f" <i>Últ. dato: {fecha}</i>" if fecha else ""),
                 f"Proyección 3M (aprox): {pct(w3,1)}",
                 f"Proyección 6M (aprox): {pct(w6,1)}", "", *detail]
        await _send_below_menu(context, chat_id, text="\n".join(lines))

# ============================ WEBHOOK / APP ============================

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

async def cmd_resumen_diario(update: Update, context: ContextTypes.DEFAULT_TYPE):
    async with ClientSession() as session:
        fx = await get_dolares(session)
        rp = await get_riesgo_pais(session)
        infl = await get_inflacion_mensual(session)
        rv = await get_reservas_lamacro(session)
        news = await fetch_rss_entries(session, 5)
    lines = []
    if fx:   lines.append(format_dolar_message(fx))
    if rp:   lines.append(f"<b>📈 Riesgo País:</b> {rp[0]} pb")
    if infl: lines.append(f"<b>📉 Inflación mensual:</b> {str(round(infl[0],1)).replace('.',',')}% ({infl[1] or ''})")
    if rv:   lines.append(f"<b>🏦 Reservas BCRA:</b> {fmt_number(rv[0],0)} MUS$ ({rv[1] or ''})")
    if news: lines.append(format_news_block(news))
    await update.effective_message.reply_text("\n\n".join(lines) if lines else "Sin datos para el resumen.")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(
        "Hola! Menús rápidos:\n"
        "/economia_menu · /acciones_menu · /cedears_menu · /alertas_menu · /portafolio\n"
        "Sugerencia: configurá tu portafolio con /portafolio"
    )

def build_app() -> Application:
    defaults = Defaults(parse_mode=ParseMode.HTML, tzinfo=TZ, link_preview_options=LinkPreviewOptions(is_disabled=True))
    app = Application.builder().token(TELEGRAM_TOKEN).defaults(defaults).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("economia_menu", cmd_menu_economia))
    app.add_handler(CommandHandler("acciones_menu", cmd_acciones_menu))
    app.add_handler(CommandHandler("cedears_menu", cmd_cedears_menu))
    app.add_handler(CallbackQueryHandler(econ_cb, pattern=r"^ECO:"))
    app.add_handler(CallbackQueryHandler(acc_ced_cb, pattern=r"^(ACC|CED):"))

    # Alertas
    app.add_handler(CommandHandler("alertas_menu", cmd_alertas_menu))
    app.add_handler(CallbackQueryHandler(alertas_menu_cb, pattern=r"^AL:"))
    app.add_handler(CallbackQueryHandler(alertas_clear_cb, pattern=r"^CLR:"))
    app.add_handler(CallbackQueryHandler(alerts_pause_cb, pattern=r"^AP:"))  # <— fix: patrón para Pausa/Reanudar
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, alertas_add_value))  # captura inputs de valor cuando corresponde
    # Inicio flujo "Agregar alerta"
    app.add_handler(CallbackQueryHandler(alertas_add_start, pattern=r"^AL:ADD$"))
    app.add_handler(CallbackQueryHandler(alertas_add_kind, pattern=r"^KIND:"))
    app.add_handler(CallbackQueryHandler(alertas_add_fx_type, pattern=r"^FXTYPE:"))
    app.add_handler(CallbackQueryHandler(alertas_add_fx_side, pattern=r"^SIDE:"))
    app.add_handler(CallbackQueryHandler(alertas_add_metric_type, pattern=r"^METRIC:"))
    app.add_handler(CallbackQueryHandler(alertas_add_ticker_cb, pattern=r"^TICK:"))
    app.add_handler(CallbackQueryHandler(alertas_back, pattern=r"^BACK:"))
    app.add_handler(CallbackQueryHandler(alertas_add_op, pattern=r"^OP:"))
    app.add_handler(CallbackQueryHandler(alertas_add_mode, pattern=r"^MODE:"))

    # Portafolio
    app.add_handler(CommandHandler("portafolio", cmd_portafolio))
    app.add_handler(CallbackQueryHandler(pf_menu_cb, pattern=r"^PF:"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, pf_text_input))  # comparte con alertas_add_value; cada modo usa su flag

    # Macros directos
    app.add_handler(CommandHandler("dolar", cmd_dolar))
    app.add_handler(CommandHandler("reservas", cmd_reservas))
    app.add_handler(CommandHandler("inflacion", cmd_inflacion))
    app.add_handler(CommandHandler("riesgo", cmd_riesgo))
    app.add_handler(CommandHandler("noticias", cmd_noticias))
    app.add_handler(CommandHandler("resumen", cmd_resumen_diario))

    # Background loops
    app.post_init.append(lambda app: _schedule_all_subs(app))
    app.post_init.append(lambda app: app.create_task(alerts_loop(app)))
    app.post_init.append(lambda app: app.create_task(keepalive_loop(app)))

    return app

# ============================ SERVER (Render Webhook) ============================

async def tg_webhook(request: web.Request):
    app: Application = request.app["bot_app"]
    if request.match_info.get("token") != WEBHOOK_SECRET:
        return web.Response(status=403, text="forbidden")
    data = await request.json()
    update = Update.de_json(data, app.bot)
    await app.process_update(update)
    return web.Response(text="ok")

async def health(request: web.Request):
    return web.Response(text="ok")

async def on_startup(app: web.Application):
    await load_state()
    bot_app: Application = app["bot_app"]
    await bot_app.bot.set_webhook(url=WEBHOOK_URL, allowed_updates=["message","callback_query"])

def main():
    app = web.Application()
    bot_app = build_app()
    app["bot_app"] = bot_app
    app.router.add_post(WEBHOOK_PATH, tg_webhook)
    app.router.add_get("/", health)
    app.on_startup.append(on_startup)
    web.run_app(app, port=PORT)

if __name__ == "__main__":
    main()
