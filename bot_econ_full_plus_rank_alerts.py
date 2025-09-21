# bot_econ_full_plus_rank_alerts.py
# -*- coding: utf-8 -*-

import os, asyncio, logging, re, html as _html, json
from time import time
from math import sqrt
from datetime import datetime, timedelta, time as dtime
from zoneinfo import ZoneInfo
from typing import Dict, List, Tuple, Any, Optional, Set
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

# ============================ LISTADOS (AMPLIADOS) ============================
ACCIONES_BA = [
    "GGAL.BA","YPFD.BA","PAMP.BA","CEPU.BA","ALUA.BA","TXAR.BA","TGSU2.BA","BYMA.BA","SUPV.BA","BMA.BA",
    "EDN.BA","CRES.BA","COME.BA","VALO.BA","TGNO4.BA","TRAN.BA","LOMA.BA","HARG.BA","CVH.BA","TECO2.BA"
]
CEDEARS_BA  = [
    "AAPL.BA","MSFT.BA","NVDA.BA","AMZN.BA","GOOGL.BA","TSLA.BA","META.BA","JNJ.BA","KO.BA","NFLX.BA",
    "BRKB.BA","PG.BA","DISN.BA","AMD.BA","INTC.BA","NKE.BA","V.BA","MA.BA","PFE.BA","XOM.BA"
]
BONOS_AR    = [
    "AL30","AL30D","AL35","AL29","GD30","GD30D","GD35","GD38","GD41","AE38",
    "AL41","AL38","GD46","AL32","GD29","AL36","AL39","GD35D","GD41D","AL29D"
]
FCI_LIST    = [
    "FCI-MoneyMarket","FCI-BonosUSD","FCI-AccionesArg","FCI-Corporativos","FCI-Liquidez","FCI-Balanceado",
    "FCI-RentaMixta","FCI-RealEstate","FCI-Commodity","FCI-Tech","FCI-BonosCER","FCI-DurationCorta",
    "FCI-DurationMedia","FCI-DurationLarga","FCI-HighYield","FCI-BlueChips","FCI-Growth","FCI-Value",
    "FCI-Latam","FCI-Global"
]
LETES_LIST  = [
    "LETRA-30D","LETRA-60D","LETRA-90D","LETRA-120D","LETRA-180D","LETRA-270D","LETRA-360D",
    "LETRA-12M","LETRA-18M","LETRA-24M","LETRA-US$-90D","LETRA-US$-180D","LETRA-USD-12M",
    "LETRA-CER-90D","LETRA-CER-180D","LETRA-CER-12M","LETRA-TNA-ALTA","LETRA-TNA-MEDIA","LETRA-TNA-BAJA","LETRA-ESPECIAL"
]
CRIPTO_TOP  = [
    "BTC-USD","ETH-USD","SOL-USD","BNB-USD","XRP-USD","ADA-USD","DOGE-USD","TON-USD","TRX-USD","DOT-USD",
    "AVAX-USD","MATIC-USD","LINK-USD","LTC-USD","UNI-USD","BCH-USD","ATOM-USD","XLM-USD","NEAR-USD","APT-USD"
]

TICKER_NAME = {
    # Acciones AR
    "GGAL.BA":"Grupo Financiero Galicia","YPFD.BA":"YPF","PAMP.BA":"Pampa Energía","CEPU.BA":"Central Puerto",
    "ALUA.BA":"Aluar","TXAR.BA":"Ternium Argentina","TGSU2.BA":"Transportadora de Gas del Sur",
    "BYMA.BA":"BYMA","SUPV.BA":"Supervielle","BMA.BA":"Banco Macro","EDN.BA":"Edenor","CRES.BA":"Cresud",
    "COME.BA":"Soc. Comercial del Plata","VALO.BA":"Gpo. Financiero Valores","TGNO4.BA":"Transportadora Gas Norte",
    "TRAN.BA":"Transener","LOMA.BA":"Loma Negra","HARG.BA":"Holcim Argentina","CVH.BA":"Cablevisión Holding",
    "TECO2.BA":"Telecom Argentina",
    # Cedears
    "AAPL.BA":"Apple","MSFT.BA":"Microsoft","NVDA.BA":"NVIDIA","AMZN.BA":"Amazon","GOOGL.BA":"Alphabet",
    "TSLA.BA":"Tesla","META.BA":"Meta","JNJ.BA":"Johnson & Johnson","KO.BA":"Coca-Cola","NFLX.BA":"Netflix",
    "BRKB.BA":"Berkshire Hathaway B","PG.BA":"Procter & Gamble","DISN.BA":"Disney","AMD.BA":"AMD","INTC.BA":"Intel",
    "NKE.BA":"Nike","V.BA":"Visa","MA.BA":"Mastercard","PFE.BA":"Pfizer","XOM.BA":"ExxonMobil",
    # Bonos (descriptivo)
    "AL30":"Bono AL30 (ARS)","AL30D":"Bono AL30D (USD)","AL35":"Bono AL35","AL29":"Bono AL29",
    "GD30":"Bono GD30 (ARS)","GD30D":"Bono GD30D (USD)","GD35":"Bono GD35 (ARS)","GD38":"Bono GD38",
    "GD41":"Bono GD41","AE38":"Bono AE38","AL41":"Bono AL41","AL38":"Bono AL38",
    "GD46":"Bono GD46","AL32":"Bono AL32","GD29":"Bono GD29","AL36":"Bono AL36","AL39":"Bono AL39",
    "GD35D":"Bono GD35D (USD)","GD41D":"Bono GD41D (USD)","AL29D":"Bono AL29D (USD)",
    # FCI y Letras: descriptivos genéricos
    "FCI-MoneyMarket":"FCI Money Market","FCI-BonosUSD":"FCI Bonos USD","FCI-AccionesArg":"FCI Acciones AR",
    "FCI-Corporativos":"FCI Bonos Corporativos","FCI-Liquidez":"FCI Liquidez","FCI-Balanceado":"FCI Balanceado",
    "FCI-RentaMixta":"FCI Renta Mixta","FCI-RealEstate":"FCI Real Estate","FCI-Commodity":"FCI Commodities",
    "FCI-Tech":"FCI Tech","FCI-BonosCER":"FCI Bonos CER","FCI-DurationCorta":"FCI Duration Corta",
    "FCI-DurationMedia":"FCI Duration Media","FCI-DurationLarga":"FCI Duration Larga","FCI-HighYield":"FCI High Yield",
    "FCI-BlueChips":"FCI Blue Chips","FCI-Growth":"FCI Growth","FCI-Value":"FCI Value",
    "FCI-Latam":"FCI Latam","FCI-Global":"FCI Global",
    # Letras
    **{k:k.replace("-", " ") for k in LETES_LIST},
    # Cripto
    "BTC-USD":"Bitcoin","ETH-USD":"Ethereum","SOL-USD":"Solana","BNB-USD":"BNB","XRP-USD":"XRP","ADA-USD":"Cardano",
    "DOGE-USD":"Dogecoin","TON-USD":"Toncoin","TRX-USD":"TRON","DOT-USD":"Polkadot","AVAX-USD":"Avalanche",
    "MATIC-USD":"Polygon","LINK-USD":"Chainlink","LTC-USD":"Litecoin","UNI-USD":"Uniswap",
    "BCH-USD":"Bitcoin Cash","ATOM-USD":"Cosmos","XLM-USD":"Stellar","NEAR-USD":"NEAR","APT-USD":"Aptos",
}
NAME_ABBR = {k: (v.split()[0] if ".BA" in k else v.split()[0]) for k,v in TICKER_NAME.items()}

# ============================ LOGGING ============================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("bot-econ-ar")

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

def load_state():
    global ALERTS, SUBS, PF
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        ALERTS = {int(k): v for k,v in data.get("alerts", {}).items()}
        SUBS = {int(k): v for k,v in data.get("subs", {}).items()}
        PF     = {int(k): v for k,v in data.get("pf", {}).items()}
        log.info("State loaded. alerts=%d subs=%d pf=%d",
                 sum(len(v) for v in ALERTS.values()), len(SUBS), len(PF))
    except Exception:
        log.info("No previous state found.")

def save_state():
    try:
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump({"alerts": ALERTS, "subs": SUBS, "pf": PF}, f, ensure_ascii=False)
    except Exception as e:
        log.warning("save_state error: %s", e)

# ============================ UTILS ============================
def fmt_number(n: Optional[float], nd=2) -> str:
    try:
        if n is None: return "—"
        s = f"{n:,.{nd}f}"
        return s.replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return str(n)
def fmt_money_ars(n: Optional[float]) -> str: return f"$ {fmt_number(n, 2)}"
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
    # CriptoYa
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
    # DolarAPI (complemento)
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

async def get_riesgo_pais(session: ClientSession) -> Optional[Tuple[int, Optional[str]]]:
    # ArgentinaDatos
    for suf in ("/riesgo-pais/ultimo", "/riesgo-pais"):
        base_ok = None
        for base in ARG_DATOS_BASES:
            j = await fetch_json(session, base+suf)
            if j:
                base_ok = j; break
        if base_ok:
            j = base_ok
            break
    else:
        j = None
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
    else:
        return None
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
            if closes[i-1] and closes[i]: rets_d.append(closes[i]/closes[i-1]-1.0)
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
    if not uniq: return []
    scored = sorted([(t,l,_score_title(t), domain_of(l)) for l,t in uniq.items()],
                    key=lambda x: x[2], reverse=True)
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
def _label_long(sym: str) -> str: return f"{TICKER_NAME.get(sym, sym)} ({sym})"
def _label_short(sym: str) -> str: return f"{NAME_ABBR.get(sym, sym)} ({sym})"

def format_dolar_message(d: Dict[str, Dict[str, Any]]) -> str:
    fecha = None
    for row in d.values():
        f = row.get("fecha")
        if f: fecha = parse_iso_ddmmyyyy(f)
    header = "<b>💵 Dólares</b>" + (f"  <i>Actualizado: {fecha}</i>" if fecha else "")
    lines = [header, "<pre>Tipo          Venta         Compra</pre>"]
    rows = []
    order = [("oficial","Oficial"),("mayorista","Mayorista"),("blue","Blue"),("mep","MEP"),("ccl","CCL"),("cripto","Cripto"),("tarjeta","Tarjeta")]
    for k, label in order:
        row = d.get(k)
        if not row: continue
        venta_val  = row.get("compra")
        compra_val = row.get("venta")
        venta  = fmt_money_ars(venta_val)  if venta_val  is not None else "—"
        compra = fmt_money_ars(compra_val) if compra_val is not None else "—"
        l = f"{label:<12}{venta:>12}    {compra:>12}"
        rows.append(f"<pre>{l}</pre>")
    rows.append("<i>Fuentes: CriptoYa + DolarAPI</i>")
    return "\n".join([lines[0], lines[1]] + rows)

def format_top3_table(title: str, fecha: Optional[str], rows_syms: List[str],
                      retmap: Dict[str, Dict[str, Optional[float]]]) -> str:
    head = f"<b>{title}</b>" + (f"  <i>Últ. Dato: {fecha}</i>" if fecha else "")
    lines = [head, "<pre>Rank  Empresa (Ticker)                 1M        3M        6M</pre>"]
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
    head = f"<b>{title}</b>" + (f"  <i>Últ. Dato: {fecha}</i>" if fecha else "")
    sub  = "<i>Proy. 3M (corto) y Proy. 6M (medio)</i>"
    lines = [head, sub, "<pre>Rank  Empresa (Ticker)                 Proy. 3M    Proy. 6M</pre>"]
    out = []
    if not rows:
        out.append("<pre>—</pre>")
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
    trend  = 2.2*_nz(m.get("trend_flag"), 0.0) + 0.28*_nz(m.get("slope50"), 0.0)
    hi52   = 0.18*_nz(m.get("hi52"), 0.0)
    risk   = -0.055*_nz(m.get("vol_ann"), 40.0) - 0.045*_nz(m.get("dd6m"), 30.0)
    meanrev = -0.06*max(0.0, abs(r1)-12.0)
    return momentum + trend + hi52 + risk + meanrev
def projection_6m(m: Dict[str, Optional[float]]) -> float:
    r6, r3, r1 = _nz(m.get("6m"), -100.0), _nz(m.get("3m"), -50.0), _nz(m.get("1m"), -20.0)
    momentum = 0.6*r6 + 0.3*r3 + 0.1*r1
    trend  = 3.1*_nz(m.get("trend_flag"), 0.0) + 0.22*_nz(m.get("slope50"), 0.0)
    hi52   = 0.22*_nz(m.get("hi52"), 0.0)
    risk   = -0.06*_nz(m.get("vol_ann"), 40.0) - 0.05*_nz(m.get("dd6m"), 30.0)
    meanrev = -0.05*max(0.0, abs(r1)-15.0)
    return momentum + trend + hi52 + risk + meanrev

# Helpers para enviar cada ranking por separado
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

# ============================ COMANDOS BÁSICOS ============================
async def cmd_dolar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    async with ClientSession() as session:
        data = await get_dolares(session)
    msg = format_dolar_message(data) if data else "No pude obtener cotizaciones ahora."
    await update.effective_message.reply_text(msg)

async def cmd_acciones_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb_menu = InlineKeyboardMarkup([
        [InlineKeyboardButton("Top 3 Acciones (Rendimiento)", callback_data="ACC:TOP3")],
        [InlineKeyboardButton("Top 5 Acciones (Proyección)", callback_data="ACC:TOP5")],
    ])
    await update.effective_message.reply_text("📊 Menú Acciones", reply_markup=kb_menu)

async def cmd_cedears_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
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
    elif data == "ACC:TOP5":
        await _rank_proj5(update, ACCIONES_BA, "🏁 Top 5 Acciones (Proyección)")
    elif data == "CED:TOP3":
        await _rank_top3(update, CEDEARS_BA, "🌎 Top 3 Cedears (Rendimiento)")
    elif data == "CED:TOP5":
        await _rank_proj5(update, CEDEARS_BA, "🏁 Top 5 Cedears (Proyección)")

# ---------- Macro ----------
async def cmd_reservas(update: Update, context: ContextTypes.DEFAULT_TYPE):
    async with ClientSession() as session:
        res = await get_reservas_lamacro(session)
    if not res: txt = "No pude obtener reservas ahora."
    else:
        val, fecha = res
        txt = (f"<b>🏦 Reservas BCRA</b>{f'  <i>Últ. Act.: {fecha}</i>' if fecha else ''}\n"
               f"<b>{fmt_number(val,0)} MUS$</b>\n<i>Fuente: LaMacro</i>")
    await update.effective_message.reply_text(txt)

async def cmd_inflacion(update: Update, context: ContextTypes.DEFAULT_TYPE):
    async with ClientSession() as session:
        tup = await get_inflacion_mensual(session)
    if tup is None: txt = "No pude obtener inflación ahora."
    else:
        val, fecha = tup; val_str = str(round(val,1)).replace(".", ",")
        txt = f"<b>📉 Inflación Mensual</b>{f'  <i>{fecha}</i>' if fecha else ''}\n<b>{val_str}%</b>\n<i>Fuente: ArgentinaDatos</i>"
    await update.effective_message.reply_text(txt)

async def cmd_riesgo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    async with ClientSession() as session:
        tup = await get_riesgo_pais(session)
    if tup is None: txt = "No pude obtener riesgo país ahora."
    else:
        rp, f = tup; f_str = parse_iso_ddmmyyyy(f)
        txt = f"<b>📈 Riesgo País</b>{f'  <i>{f_str}</i>' if f_str else ''}\n<b>{rp} pb</b>\n<i>Fuente: ArgentinaDatos</i>"
    await update.effective_message.reply_text(txt)

async def cmd_noticias(update: Update, context: ContextTypes.DEFAULT_TYPE):
    async with ClientSession() as session:
        news = await fetch_rss_entries(session, limit=5)
    txt = format_news_block(news or [])
    await update.effective_message.reply_text(txt)

async def cmd_resumen_diario(update: Update, context: ContextTypes.DEFAULT_TYPE):
    async with ClientSession() as session:
        dolares = await get_dolares(session)
        riesgo  = await get_riesgo_pais(session)
        reservas = await get_reservas_lamacro(session)
        infl    = await get_inflacion_mensual(session)
        news    = await fetch_rss_entries(session, limit=5)
    blocks = ["<b>🗞️ Resumen Diario</b>", format_dolar_message(dolares)]
    if riesgo:
        rp,f = riesgo; f_str = parse_iso_ddmmyyyy(f)
        blocks.append(f"<b>📈 Riesgo País</b>{f'  <i>{f_str}</i>' if f_str else ''}\n<b>{rp} pb</b>\n<i>Fuente: ArgentinaDatos</i>")
    if reservas:
        v,fe = reservas; blocks.append(f"<b>🏦 Reservas BCRA</b>{f'  <i>Últ. Act.: {fe}</i>' if fe else ''}\n<b>{fmt_number(v,0)} MUS$</b>\n<i>Fuente: LaMacro</i>")
    if infl:
        iv,ip = infl; blocks.append(f"<b>📉 Inflación Mensual</b>{f'  <i>{ip}</i>' if ip else ''}\n<b>{str(round(iv,1)).replace('.',',')}%</b>\n<i>Fuente: ArgentinaDatos</i>")
    blocks.append(format_news_block(news or []))
    await update.effective_message.reply_text("\n\n".join(blocks))

# ============================ ALERTAS ============================
AL_KIND, AL_FX_TYPE, AL_FX_SIDE, AL_OP, AL_MODE, AL_VALUE, AL_METRIC_TYPE, AL_TICKER = range(8)
ALERTS_SILENT_UNTIL: Dict[int, float] = {}
ALERTS_PAUSED: Set[int] = set()

def kb(rows: List[List[Tuple[str,str]]]) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton(text, callback_data=data) for text, data in r] for r in rows])

def kb_tickers(symbols: List[str], back_target: str, prefix: str) -> InlineKeyboardMarkup:
    rows: List[List[Tuple[str,str]]] = []
    row: List[Tuple[str,str]] = []
    for s in symbols:
        label = _label_long(s)
        row.append((label, f"{prefix}:{s}"))
        if len(row) == 2: rows.append(row); row = []
    if row: rows.append(row)
    rows.append([("Volver","BACK:"+back_target), ("Cancelar","CANCEL")])
    return kb(rows)

def _parse_float_user_strict(s: str) -> Optional[float]:
    s = (s or "").strip()
    if re.search(r"[^\d\.,\-+]", s):
        return None
    s = s.replace(".", "").replace(",", ".")
    try: return float(s)
    except Exception: return None

def _fx_display_value(row: Dict[str, Any], side: str) -> Optional[float]:
    # mostramos “venta” al usuario como precio de referencia
    return row.get("compra") if side == "venta" else row.get("venta")

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
    if data == "AL:LIST": return await cmd_alertas_list(update, context)
    if data == "AL:CLEAR":return await cmd_alertas_clear(update, context)
    if data == "AL:PAUSE":return await cmd_alertas_pause(update, context)
    if data == "AL:RESUME":return await cmd_alertas_resume(update, context)

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
                if t=="riesgo": val = f"{v:.0f} pb"
                elif t=="reservas": val = f"{fmt_number(v,0)} MUS$"
                else: val = f"{str(round(v,1)).replace('.',',')}%"
                lines.append(f"{i}. {t.upper()} {html_op(op)} {val}")
            else:
                sym, op, v = r["symbol"], r["op"], r["value"]
                lines.append(f"{i}. {_label_long(sym)} (Precio) {html_op(op)} {fmt_money_ars(v)}")
        if chat_id in ALERTS_PAUSED:
            lines.append("\n<i>Alertas en pausa (indefinida)</i>")
        txt = "\n".join(lines)
    await update.effective_message.reply_text(txt)

# ---- CLEAR/PAUSE/RESUME ----
async def cmd_alertas_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    rules = ALERTS.get(chat_id, [])
    if not rules: await update.effective_message.reply_text("No tenés alertas guardadas."); return
    buttons: List[List[Tuple[str,str]]] = []
    for i, r in enumerate(rules, 1):
        if r.get("kind") == "fx":
            label = f"{i}. {r['type'].upper()}({r['side']}) {html_op(r['op'])} {fmt_money_ars(r['value'])}"
        elif r.get("kind") == "metric":
            if r["type"]=="riesgo": val = f"{r['value']:.0f} pb"
            elif r["type"]=="reservas": val = f"{fmt_number(r['value'],0)} MUS$"
            else: val = f"{str(round(r['value'],1)).replace('.',',')}%"
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
    if data == "CANCEL": await q.edit_message_text("Operación cancelada."); return
    if data == "ALL":
        cnt = len(rules); ALERTS[chat_id] = []; save_state()
        await q.edit_message_text(f"Se eliminaron {cnt} alertas."); return
    try: idx = int(data)
    except Exception: await q.edit_message_text("Acción inválida."); return
    if 0 <= idx < len(rules):
        rules.pop(idx); save_state(); await q.edit_message_text("Alerta eliminada.")
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
    if data == "AP:CLOSE": await q.edit_message_text("Listo."); return
    if data == "AP:RESUME":
        ALERTS_PAUSED.discard(chat_id); ALERTS_SILENT_UNTIL.pop(chat_id, None)
        await q.edit_message_text("🔔 Alertas reanudadas."); return
    if data.startswith("AP:PAUSE:"):
        arg = data.split(":")[-1]
        if arg == "INF":
            ALERTS_PAUSED.add(chat_id); ALERTS_SILENT_UNTIL.pop(chat_id, None)
            await q.edit_message_text("🔕 Alertas en pausa (indefinida)."); return
        try:
            hrs = int(arg); until = datetime.now(TZ) + timedelta(hours=hrs)
            ALERTS_SILENT_UNTIL[chat_id] = until.timestamp(); ALERTS_PAUSED.discard(chat_id)
            await q.edit_message_text(f"🔕 Alertas en pausa por {hrs}h (hasta {until.strftime('%d/%m %H:%M')})."); return
        except Exception:
            await q.edit_message_text("Acción inválida."); return
async def cmd_alertas_resume(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ALERTS_PAUSED.discard(chat_id); ALERTS_SILENT_UNTIL.pop(chat_id, None)
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
    return kb([[("Venta","SIDE:venta"),("Compra","SIDE:compra")],[("Volver","BACK:FXTYPE"),("Cancelar","CANCEL")]])

async def alertas_add_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # entry point del flujo, soporta callback directo "AL:ADD"
    if update.callback_query:
        q = update.callback_query; await q.answer()
        context.user_data["al"] = {}
        k = kb([
            [("Dólares", "KIND:fx"), ("Economía", "KIND:metric")],
            [("Acciones", "KIND:acciones"), ("Cedears", "KIND:cedears")],
            [("Cancelar", "CANCEL")]
        ])
        await q.edit_message_text("¿Qué querés alertar?", reply_markup=k)
    else:
        context.user_data["al"] = {}
        k = kb([
            [("Dólares", "KIND:fx"), ("Economía", "KIND:metric")],
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
            [("Dólares", "KIND:fx"), ("Economía", "KIND:metric")],
            [("Acciones", "KIND:acciones"), ("Cedears", "KIND:cedears")],
            [("Cancelar", "CANCEL")]
        ])
        await q.edit_message_text("¿Qué querés alertar?", reply_markup=k); return AL_KIND
    if target == "FXTYPE":
        await q.edit_message_text("Elegí el tipo de dólar:", reply_markup=kb_submenu_fx()); return AL_FX_TYPE
    if target == "FXSIDE":
        t = al.get("type","?")
        await q.edit_message_text(f"Tipo: {t.upper()}\nElegí lado:", reply_markup=kb_fx_side_for(t)); return AL_FX_SIDE
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
        kb_mode = kb([[("Ingresar Importe", "MODE:absolute"),("Ingresar % vs valor actual", "MODE:percent")],
                      [("Volver","BACK:OP"),("Cancelar","CANCEL")]])
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
        al["kind"] = "fx"
        await q.edit_message_text("Elegí el tipo de dólar:", reply_markup=kb_submenu_fx()); return AL_FX_TYPE
    if kind == "metric":
        al["kind"] = "metric"
        await q.edit_message_text("Elegí la métrica:", reply_markup=kb_submenu_metric()); return AL_METRIC_TYPE
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
                "<i>Válidos: 100  |  1000.5   ·  Inválidos: $100, 1.000,50, 100%</i>"
            )
            await q.edit_message_text(msg, parse_mode=ParseMode.HTML)
        return AL_VALUE
    kb_mode = kb([[("Ingresar Importe", "MODE:absolute"),("Ingresar % vs valor actual", "MODE:percent")],
                  [("Volver","BACK:OP"),("Cancelar","CANCEL")]])
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
                       f"Ahora: {cur_s}\n\nIngresá el <b>%</b> (solo número). Ej: 10  |  7.5")
            else:
                msg = (f"Tipo: {al.get('type','?').upper()} | Lado: {al.get('side','?')} | Condición: {op_text}\n"
                       f"Ahora: {cur_s}\n\nIngresá el <b>importe</b> AR$ (solo número). Ej: 1580  |  25500")
            await q.edit_message_text(msg, parse_mode=ParseMode.HTML); return AL_VALUE
        if al.get("kind") == "metric":
            rp = await get_riesgo_pais(session); infl = await get_inflacion_mensual(session); rv  = await get_reservas_lamacro(session)
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
            if cur is None: await update.message.reply_text("No pude leer el valor actual."); return ConversationHandler.END
            thr = cur*(1 + (val/100.0)) if al.get("mode")=="percent" and al["op"] == ">" else \
                  cur*(1 - (val/100.0)) if al.get("mode")=="percent" else val
            if al.get("mode") == "absolute":
                if (al["op"] == ">" and thr <= cur) or (al["op"] == "<" and thr >= cur):
                    await update.message.reply_text(f"El objetivo debe ser {'mayor' if al['op']=='>' else 'menor'} que {fmt_money_ars(cur)}."); return AL_VALUE
            ALERTS.setdefault(chat_id, []).append({"kind":"fx","type":al["type"],"side":al["side"],"op":al["op"],"value":float(thr)})
            save_state()
            await update.message.reply_text("Listo. Alerta agregada ✅")
            return ConversationHandler.END
        if al.get("kind") == "metric":
            rp = await get_riesgo_pais(session); infl = await get_inflacion_mensual(session); rv  = await get_reservas_lamacro(session)
            curmap = {"riesgo": float(rp[0]) if rp else None, "inflacion": float(infl[0]) if infl else None, "reservas": rv[0] if rv else None}
            cur = curmap.get(al["type"])
            if cur is None: await update.message.reply_text("No pude leer el valor actual."); return ConversationHandler.END
            thr = cur*(1 + (val/100.0)) if al.get("mode")=="percent" and al["op"] == ">" else \
                  cur*(1 - (val/100.0)) if al.get("mode")=="percent" else val
            if al.get("mode") == "absolute":
                if (al["op"] == ">" and thr <= cur) or (al["op"] == "<" and thr >= cur):
                    await update.message.reply_text("El objetivo debe ser válido respecto al valor actual."); return AL_VALUE
            ALERTS.setdefault(chat_id, []).append({"kind":"metric","type":al["type"],"op":al["op"],"value":float(thr)})
            save_state()
            await update.message.reply_text("Listo. Alerta agregada ✅")
            return ConversationHandler.END
        # ticker
        sym, op = al.get("symbol"), al.get("op")
        metmap, _ = await metrics_for_symbols(session, [sym])
        last_px = metmap.get(sym, {}).get("last_px")
        if last_px is None: await update.message.reply_text("No pude leer el precio actual."); return ConversationHandler.END
        thr = val
        if (op == ">" and thr <= last_px) or (op == "<" and thr >= last_px):
            await update.message.reply_text(f"El precio objetivo debe ser {'mayor' if op=='>' else 'menor'} que {fmt_money_ars(last_px)}."); return AL_VALUE
        ALERTS.setdefault(chat_id, []).append({"kind":"ticker","symbol":sym,"op":op,"value":float(thr),"mode":"absolute"})
        save_state()
        await update.message.reply_text("Listo. Alerta agregada ✅")
        return ConversationHandler.END

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
                    vals = {"riesgo": float(rp[0]) if rp else None, "inflacion": float(infl[0]) if infl else None, "reservas": rv[0] if rv else None}
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
                            sym = r["symbol"]; m = metmap.get(sym, {})
                            cur = m.get("last_px"); 
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
    app.job_queue.run_daily(_job_send_daily, time=dtime(hour=h, minute=m, tzinfo=TZ),
                            chat_id=chat_id, name=_job_name_daily(chat_id))

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
    if data == "SUBS:CLOSE":
        await q.edit_message_text("Listo."); return ConversationHandler.END
    if data == "SUBS:OFF":
        if chat_id in SUBS and SUBS[chat_id].get("daily"):
            SUBS[chat_id]["daily"] = None; save_state()
        for j in context.application.job_queue.get_jobs_by_name(_job_name_daily(chat_id)): j.schedule_removal()
        await q.edit_message_text("Suscripción cancelada."); return ConversationHandler.END
    if data.startswith("SUBS:T:"):
        hhmm = data.split(":",2)[2]
        SUBS.setdefault(chat_id, {})["daily"] = hhmm; save_state()
        _schedule_daily_for_chat(context.application, chat_id, hhmm)
        await q.edit_message_text(f"Te suscribí al Resumen Diario a las {hhmm} (hora AR)."); return ConversationHandler.END
    await q.edit_message_text("Acción inválida."); return ConversationHandler.END

# ============================ PORTAFOLIO ============================
def pf_get(chat_id: int) -> Dict[str, Any]:
    return PF.setdefault(chat_id, {"base": {"moneda":"ARS", "tc":"mep"}, "monto": 0.0, "items": []})

def kb_pf_main() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Ayuda", callback_data="PF:HELP")],
        [InlineKeyboardButton("Fijar base", callback_data="PF:SETBASE"),
         InlineKeyboardButton("Fijar monto", callback_data="PF:SETMONTO")],
        [InlineKeyboardButton("Agregar instrumento", callback_data="PF:ADD")],
        [InlineKeyboardButton("Ver composición", callback_data="PF:LIST"),
         InlineKeyboardButton("Editar instrumento", callback_data="PF:EDIT")],
        [InlineKeyboardButton("Rendimiento", callback_data="PF:RET"),
         InlineKeyboardButton("Proyección", callback_data="PF:PROJ")],
        [InlineKeyboardButton("Eliminar portafolio", callback_data="PF:CLEAR")],
    ])

def kb_pick_generic(symbols: List[str], back: str, prefix: str) -> InlineKeyboardMarkup:
    rows = []
    row = []
    for s in symbols:
        label = _label_long(s)
        row.append((label, f"{prefix}:{s}"))
        if len(row) == 2:
            rows.append(row); row = []
    if row: rows.append(row)
    # ← volver un paso (al submenú de tipo de instrumento), NO al menú principal
    rows.append([("Volver","PF:ADD")])
    return InlineKeyboardMarkup([[InlineKeyboardButton(t, callback_data=d) for t,d in r] for r in rows])

async def cmd_portafolio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text("📦 Menú Portafolio", reply_markup=kb_pf_main())

async def pf_menu_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    chat_id = q.message.chat_id; data = q.data
    if data == "PF:HELP":
        txt = ("<b>Cómo armar tu portafolio</b>\n\n"
               "1) Fijá base y tipo de cambio.\n2) Definí el monto total.\n"
               "3) Agregá instrumentos (lista seleccionable).\n"
               "4) Ver composición y editar.\n5) Rendimiento y Proyección (en desarrollo).\n\n"
               "<i>Formato de números: solo dígitos y decimal. Sin $ ni % ni comas.</i>")
        await q.edit_message_text(txt, reply_markup=kb_pf_main()); return
    if data == "PF:SETBASE":
        kb_base = InlineKeyboardMarkup([
            [InlineKeyboardButton("ARS / MEP", callback_data="PF:BASE:ARS:mep"),
             InlineKeyboardButton("ARS / CCL", callback_data="PF:BASE:ARS:ccl")],
            [InlineKeyboardButton("ARS / Oficial", callback_data="PF:BASE:ARS:oficial"),
             InlineKeyboardButton("USD", callback_data="PF:BASE:USD:-")],
            [InlineKeyboardButton("Volver", callback_data="PF:BACK")]
        ])
        await q.edit_message_text("Elegí base del portafolio:", reply_markup=kb_base); return
    if data.startswith("PF:BASE:"):
        _,_,mon,tc = data.split(":")
        pf = pf_get(chat_id)
        pf["base"] = {"moneda": mon, "tc": tc if tc!="-" else None}; save_state()
        msg = f"Base fijada: {mon}" + (f" / {tc.upper()}" if (tc and tc != "-") else "")
        await q.edit_message_text(msg, reply_markup=kb_pf_main()); return
    if data == "PF:SETMONTO":
        context.user_data["pf_mode"] = "set_monto"
        await q.edit_message_text("Ingresá el <b>monto total</b> (solo número).", parse_mode=ParseMode.HTML, reply_markup=kb_pf_main()); return
    if data == "PF:ADD":
        kb_add = InlineKeyboardMarkup([
            [InlineKeyboardButton("Acción (.BA)", callback_data="PF:ADD:accion"),
             InlineKeyboardButton("Cedear (.BA)", callback_data="PF:ADD:cedear")],
            [InlineKeyboardButton("Bono", callback_data="PF:ADD:bono"),
             InlineKeyboardButton("FCI", callback_data="PF:ADD:fci")],
            [InlineKeyboardButton("Letras", callback_data="PF:ADD:lete"),
             InlineKeyboardButton("Cripto (top)", callback_data="PF:ADD:cripto")],
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
        else: # cripto
            await q.edit_message_text("Elegí la cripto:", reply_markup=kb_pick_generic(CRIPTO_TOP, "PF:ADD", "PF:PICK"))
        return
    if data.startswith("PF:PICK:"):
        sym = data.split(":")[2]
        context.user_data["pf_add_simbolo"] = sym
        kb_ask = InlineKeyboardMarkup([
            [InlineKeyboardButton("Por cantidad", callback_data="PF:ADDQTY"),
             InlineKeyboardButton("Por importe", callback_data="PF:ADDAMT")],
            [InlineKeyboardButton("Volver", callback_data="PF:ADD")]
        ])
        await q.edit_message_text(f"Seleccionado: {_label_long(sym)}\n¿Cargar por cantidad o por importe?", reply_markup=kb_ask); return
    if data == "PF:ADDQTY":
        context.user_data["pf_mode"] = "pf_add_qty"
        await q.edit_message_text("Ingresá la <b>cantidad</b> (solo número).", parse_mode=ParseMode.HTML); return
    if data == "PF:ADDAMT":
        context.user_data["pf_mode"] = "pf_add_amt"
        await q.edit_message_text("Ingresá el <b>importe</b> (solo número).", parse_mode=ParseMode.HTML); return
    if data == "PF:LIST":
        pf = pf_get(chat_id)
        if not pf["items"]:
            await q.edit_message_text("Tu portafolio está vacío. Usá «Agregar instrumento».", reply_markup=kb_pf_main()); return
        lines = [f"<b>Portafolio</b> — Base: {pf['base']['moneda']}" + (f"/{pf['base']['tc'].upper()}" if pf['base'].get('tc') else ""),
                 f"Monto objetivo: {fmt_money_ars(pf['monto'])}"]
        for i,it in enumerate(pf["items"],1):
            desc = f"{i}. {it.get('tipo','').upper()}"
            if it.get("simbolo"): desc = f"{i}. {_label_long(it['simbolo'])} [{it['tipo'].upper()}]"
            if it.get("cantidad") is not None: desc += f" | Cant: {it['cantidad']}"
            if it.get("importe")  is not None: desc += f" | Importe: {fmt_money_ars(it['importe'])}"
            if it.get("tna")      is not None: desc += f" | TNA: {str(it['tna']).replace('.',',')}%"
            lines.append(desc)
        await q.edit_message_text("\n".join(lines), reply_markup=kb_pf_main()); return
    if data == "PF:EDIT":
        pf = pf_get(chat_id)
        if not pf["items"]:
            await q.edit_message_text("No hay instrumentos para editar.", reply_markup=kb_pf_main()); return
        buttons = []
        for i,it in enumerate(pf["items"],1):
            label = f"{i}. " + (_label_long(it['simbolo']) if it.get("simbolo") else it.get("tipo","").upper())
            buttons.append([InlineKeyboardButton(label, callback_data=f"PF:EDIT:{i-1}")])
        buttons.append([InlineKeyboardButton("Volver", callback_data="PF:BACK")])
        await q.edit_message_text("Elegí instrumento a editar:", reply_markup=InlineKeyboardMarkup(buttons)); return
    if data.startswith("PF:EDIT:"):
        idx = int(data.split(":")[2])
        context.user_data["pf_edit_idx"] = idx
        kb_ed = InlineKeyboardMarkup([
            [InlineKeyboardButton("+ Cantidad", callback_data="PF:ED:ADDQ"),
             InlineKeyboardButton("- Cantidad", callback_data="PF:ED:SUBQ")],
            [InlineKeyboardButton("Cambiar importe", callback_data="PF:ED:AMT")],
            [InlineKeyboardButton("Eliminar este", callback_data="PF:ED:DEL")],
            [InlineKeyboardButton("Volver", callback_data="PF:EDIT")]
        ])
        await q.edit_message_text("¿Qué querés hacer?", reply_markup=kb_ed); return
    if data == "PF:ED:ADDQ":
        context.user_data["pf_mode"] = "edit_addq"
        await q.edit_message_text("Ingresá la <b>cantidad a sumar</b>.", parse_mode=ParseMode.HTML); return
    if data == "PF:ED:SUBQ":
        context.user_data["pf_mode"] = "edit_subq"
        await q.edit_message_text("Ingresá la <b>cantidad a restar</b>.", parse_mode=ParseMode.HTML); return
    if data == "PF:ED:AMT":
        context.user_data["pf_mode"] = "edit_amt"
        await q.edit_message_text("Ingresá el <b>nuevo importe</b>.", parse_mode=ParseMode.HTML); return
    if data == "PF:ED:DEL":
        pf = pf_get(chat_id); idx = context.user_data.get("pf_edit_idx", -1)
        if 0 <= idx < len(pf["items"]):
            pf["items"].pop(idx); save_state()
            await q.edit_message_text("Instrumento eliminado.", reply_markup=kb_pf_main()); return
        await q.edit_message_text("Índice inválido.", reply_markup=kb_pf_main()); return
    if data == "PF:RET":
        kb_back = InlineKeyboardMarkup([[InlineKeyboardButton("Volver", callback_data="PF:BACK")]])
        await q.edit_message_text("Rendimiento del portafolio: (en desarrollo).", reply_markup=kb_back); return
    if data == "PF:PROJ":
        kb_back = InlineKeyboardMarkup([[InlineKeyboardButton("Volver", callback_data="PF:BACK")]])
        await q.edit_message_text("Proyección del portafolio: (en desarrollo).", reply_markup=kb_back); return
    if data == "PF:CLEAR":
        PF[chat_id] = {"base": {"moneda":"ARS","tc":"mep"}, "monto": 0.0, "items": []}; save_state()
        await q.edit_message_text("Portafolio eliminado.", reply_markup=kb_pf_main()); return
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
    text = (update.message.text or "").strip()
    pf = pf_get(chat_id)

    if mode == "set_monto":
        v = _parse_num_text(text)
        if v is None:
            await update.message.reply_text("Ingresá solo número (sin símbolos)."); return
        pf["monto"] = float(v); save_state()
        await update.message.reply_text(f"Monto fijado: {fmt_money_ars(v)}"); context.user_data["pf_mode"]=None; return

    # Alta por cantidad/importe (símbolo ya elegido)
    if mode in ("pf_add_qty","pf_add_amt"):
        v = _parse_num_text(text)
        if v is None:
            await update.message.reply_text("Ingresá solo número (sin símbolos)."); return
        tipo = context.user_data.get("pf_add_tipo"); sym = context.user_data.get("pf_add_simbolo","")
        if mode == "pf_add_qty":
            pf["items"].append({"tipo":tipo, "simbolo": sym, "cantidad": float(v)})
        else:
            pf["items"].append({"tipo":tipo, "simbolo": sym, "importe": float(v)})
        save_state()
        await update.message.reply_text("Agregado ✅"); context.user_data["pf_mode"]=None; context.user_data.pop("pf_add_tipo",None); context.user_data.pop("pf_add_simbolo",None); return

    # Ediciones
    if mode in ("edit_addq","edit_subq","edit_amt"):
        v = _parse_num_text(text)
        if v is None:
            await update.message.reply_text("Ingresá solo número (sin símbolos)."); return
        idx = context.user_data.get("pf_edit_idx", -1)
        if not (0 <= idx < len(pf["items"])):
            await update.message.reply_text("Índice inválido."); context.user_data["pf_mode"]=None; return
        it = pf["items"][idx]
        if mode == "edit_amt":
            it["importe"] = float(v)
        else:
            cur = float(it.get("cantidad") or 0.0)
            it["cantidad"] = cur + (v if mode=="edit_addq" else -v)
            if it["cantidad"] < 0: it["cantidad"] = 0.0
        save_state()
        await update.message.reply_text("Actualizado ✅"); context.user_data["pf_mode"]=None; return

# ============================ MENÚ ECONOMÍA ============================
async def cmd_menu_economia(update: Update, context: ContextTypes.DEFAULT_TYPE):
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
    if data == "ECO:DOLAR":    await cmd_dolar(update, context)
    if data == "ECO:RESERVAS": await cmd_reservas(update, context)
    if data == "ECO:INFLACION":await cmd_inflacion(update, context)
    if data == "ECO:RIESGO":   await cmd_riesgo(update, context)
    if data == "ECO:NOTICIAS": await cmd_noticias(update, context)

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

async def on_startup(app: web.Application):
    load_state()
    await application.initialize()
    await application.start()
    await application.bot.set_webhook(url=WEBHOOK_URL, allowed_updates=["message","callback_query"], drop_pending_updates=True)
    cmds = [
        BotCommand("cedears", "Menú Cedears"),
        BotCommand("acciones", "Menú Acciones"),
        BotCommand("economia", "Menú Economía"),
        BotCommand("alertas_menu", "Menú Alertas"),
        BotCommand("suscripciones", "Suscripción"),
        BotCommand("portafolio", "Menú Portafolio"),
    ]
    try: await application.bot.set_my_commands(cmds)
    except Exception as e: log.warning("set_my_commands error: %s", e)
    log.info("Webhook set: %s", WEBHOOK_URL)
    _schedule_all_subs(application)
    asyncio.create_task(keepalive_loop(application))
    asyncio.create_task(alerts_loop(application))

async def on_cleanup(app: web.Application):
    await application.stop(); await application.shutdown()

async def handle_root(request: web.Request): return web.Response(text="ok", status=200)
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

# ---- Handlers raíz/menús ----
application.add_handler(CommandHandler("acciones", cmd_acciones_menu))
application.add_handler(CommandHandler("cedears", cmd_cedears_menu))
application.add_handler(CallbackQueryHandler(acc_ced_cb, pattern=r"^(ACC:(TOP3|TOP5)|CED:(TOP3|TOP5))$"))

application.add_handler(CommandHandler("economia", cmd_menu_economia))
application.add_handler(CallbackQueryHandler(econ_cb, pattern=r"^ECO:(DOLAR|RESERVAS|INFLACION|RIESGO|NOTICIAS)$"))

application.add_handler(CommandHandler("alertas_menu", cmd_alertas_menu))
# Ojo: NO capturamos AL:ADD acá; lo maneja la conversación como entry point.
application.add_handler(CallbackQueryHandler(alertas_menu_cb, pattern=r"^AL:(LIST|CLEAR|PAUSE|RESUME)$"))

application.add_handler(CommandHandler("portafolio", cmd_portafolio))
application.add_handler(CallbackQueryHandler(pf_menu_cb, pattern=r"^PF:(HELP|SETBASE|SETMONTO|ADD|LIST|EDIT|RET|PROJ|CLEAR|BASE:.*|ADD:.*|PICK:.*|ED:ADDQ|ED:SUBQ|ED:AMT|ED:DEL|BACK|EDIT:\d+|ADDQTY|ADDAMT)$"))
application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, pf_text_input))

# ---- Suscripciones ----
application.add_handler(CommandHandler("suscripciones", cmd_subs))
application.add_handler(CallbackQueryHandler(subs_cb, pattern=r"^SUBS:(T:..:..|OFF|CLOSE)$"))

# ---- Comandos directos macro (para reutilizar) ----
application.add_handler(CommandHandler("dolar", cmd_dolar))
application.add_handler(CommandHandler("reservas", cmd_reservas))
application.add_handler(CommandHandler("inflacion", cmd_inflacion))
application.add_handler(CommandHandler("riesgo", cmd_riesgo))
application.add_handler(CommandHandler("noticias", cmd_noticias))
application.add_handler(CommandHandler("resumen_diario", cmd_resumen_diario))

# ---- Alertas: clear/pause/resume ----
application.add_handler(CallbackQueryHandler(alertas_clear_cb, pattern=r"^CLR:(\d+|ALL|CANCEL)$"))
application.add_handler(CommandHandler("alertas_clear", cmd_alertas_clear))
application.add_handler(CommandHandler("alertas_pause", cmd_alertas_pause))
application.add_handler(CallbackQueryHandler(alerts_pause_cb, pattern=r"^AP:(PAUSE:(INF|1|3|6|12|24)|RESUME|CLOSE)$"))
application.add_handler(CommandHandler("alertas_resume", cmd_alertas_resume))

# ---- Conversación Agregar Alerta ----
conv_alertas = ConversationHandler(
    entry_points=[
        CommandHandler("alertas_add", alertas_add_start),
        CallbackQueryHandler(alertas_add_start, pattern=r"^AL:ADD$"),  # ← ahora el botón "Agregar" inicia el flujo
    ],
    states={
        AL_KIND: [
            CallbackQueryHandler(alertas_add_kind, pattern=r"^KIND:(fx|metric|acciones|cedears)$"),
            CallbackQueryHandler(alertas_back, pattern=r"^BACK:KIND$"),
            CallbackQueryHandler(lambda u,c: c.application.bot.send_message(u.effective_chat.id,"Cancelado."), pattern=r"^CANCEL$"),
        ],
        AL_FX_TYPE: [
            CallbackQueryHandler(alertas_add_fx_type, pattern=r"^FXTYPE:(oficial|mayorista|blue|mep|ccl|tarjeta|cripto)$"),
            CallbackQueryHandler(alertas_back, pattern=r"^BACK:FXTYPE$"),
            CallbackQueryHandler(lambda u,c: ConversationHandler.END, pattern=r"^CANCEL$"),
        ],
        AL_FX_SIDE: [
            CallbackQueryHandler(alertas_add_fx_side, pattern=r"^SIDE:(venta|compra)$"),
            CallbackQueryHandler(alertas_back, pattern=r"^BACK:FXSIDE$"),
            CallbackQueryHandler(lambda u,c: ConversationHandler.END, pattern=r"^CANCEL$"),
        ],
        AL_METRIC_TYPE: [
            CallbackQueryHandler(alertas_add_metric_type, pattern=r"^METRIC:(riesgo|inflacion|reservas)$"),
            CallbackQueryHandler(alertas_back, pattern=r"^BACK:METRIC$"),
            CallbackQueryHandler(lambda u,c: ConversationHandler.END, pattern=r"^CANCEL$"),
        ],
        AL_TICKER: [
            CallbackQueryHandler(alertas_add_ticker_cb, pattern=r"^TICK:[A-Z0-9\.\-]+$"),
            CallbackQueryHandler(alertas_back, pattern=r"^BACK:(TICKERS_ACC|TICKERS_CEDEARS)$"),
            CallbackQueryHandler(lambda u,c: ConversationHandler.END, pattern=r"^CANCEL$"),
        ],
        AL_OP: [
            CallbackQueryHandler(alertas_add_op, pattern=r"^OP:(>|<)$"),
            CallbackQueryHandler(alertas_back, pattern=r"^BACK:OP$"),
            CallbackQueryHandler(alertas_back, pattern=r"^BACK:(TICKERS_ACC|TICKERS_CEDEARS|FXSIDE|METRIC)$"),
            CallbackQueryHandler(lambda u,c: ConversationHandler.END, pattern=r"^CANCEL$"),
        ],
        AL_MODE: [
            CallbackQueryHandler(alertas_add_mode, pattern=r"^MODE:(absolute|percent)$"),
            CallbackQueryHandler(alertas_back, pattern=r"^BACK:MODE$"),
            CallbackQueryHandler(alertas_back, pattern=r"^BACK:OP$"),
            CallbackQueryHandler(lambda u,c: ConversationHandler.END, pattern=r"^CANCEL$"),
        ],
        AL_VALUE: [
            MessageHandler(filters.TEXT & ~filters.COMMAND, alertas_add_value),
        ],
    },
    fallbacks=[CallbackQueryHandler(lambda u,c: ConversationHandler.END, pattern=r"^CANCEL$")],
    name="conv_alertas",
    persistent=False,
)

application.add_handler(conv_alertas)

# ---- Run aiohttp app (Render) ----
if __name__ == "__main__":
    app = build_web_app()
    web.run_app(app, host="0.0.0.0", port=PORT)
