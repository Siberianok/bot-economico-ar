# bot_econ_full_plus_rank_alerts.py
# -*- coding: utf-8 -*-

import os, asyncio, logging, re, html as _html, json, math, io, signal, csv, unicodedata
import copy
import urllib.request
import urllib.error
from time import time
from math import sqrt, floor
from datetime import datetime, timedelta, time as dtime
from zoneinfo import ZoneInfo
from typing import Dict, List, Tuple, Any, Optional, Set, Callable
from urllib.parse import urlparse, quote

# ====== matplotlib opcional (no rompe si no está instalado) ======
HAS_MPL = False
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MPL = True
except Exception:
    plt = None
    np = None

from aiohttp import ClientSession, ClientTimeout, web
from telegram import (
    Update, LinkPreviewOptions, BotCommand, InlineKeyboardMarkup, InlineKeyboardButton,
)
from telegram.constants import ParseMode
from telegram.ext import (
    Application, CommandHandler, ContextTypes, CallbackQueryHandler,
    MessageHandler, ConversationHandler, filters
)

# ============================ CONFIG ============================

TZ = ZoneInfo("America/Argentina/Buenos_Aires")
TELEGRAM_TOKEN = (os.getenv("TELEGRAM_TOKEN") or os.getenv("BOT_TOKEN") or "").strip()
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "tgwebhook").strip().strip("/")
PORT = int(os.getenv("PORT", "10000"))
BASE_URL = os.getenv("BASE_URL", os.getenv("RENDER_EXTERNAL_URL", "http://localhost")).rstrip("/")
ENV_STATE_PATH = os.getenv("STATE_PATH", "state.json")
UPSTASH_URL = (os.getenv("UPSTASH_REDIS_REST_URL") or os.getenv("UPSTASH_URL") or "").strip()
UPSTASH_TOKEN = (os.getenv("UPSTASH_REDIS_REST_TOKEN") or os.getenv("UPSTASH_TOKEN") or "").strip()
UPSTASH_REDIS_URL = (
    os.getenv("UPSTASH_REDIS_URL")
    or os.getenv("REDIS_URL")
    or os.getenv("redis-url")
    or ""
).strip()
UPSTASH_STATE_KEY = os.getenv("UPSTASH_STATE_KEY", "bot-econ-state").strip()

if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN/BOT_TOKEN no configurado.")

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

BINANCE_EXCHANGE_INFO_URL = "https://api.binance.com/api/v3/exchangeInfo"
BINANCE_TICKER_PRICE_URLS = [
    "https://api.binance.com/api/v3/ticker/price",
    "https://data.binance.com/api/v3/ticker/price",
    "https://www.binance.com/api/v3/ticker/price",
    "https://api.binance.us/api/v3/ticker/price",
]
BINANCE_FIAT_QUOTES = {
    "USDT", "USDC", "BUSD", "TUSD", "FDUSD", "DAI", "USD", "EUR", "GBP",
    "ARS", "BRL", "TRY", "RUB", "AUD", "CAD", "JPY", "CHF", "MXN"
}
BINANCE_QUOTE_SYMBOL = {
    "USDT": "US$", "USDC": "US$", "BUSD": "US$", "TUSD": "US$", "FDUSD": "US$",
    "DAI": "US$", "USD": "US$", "EUR": "€", "GBP": "£", "ARS": "$",
    "BRL": "R$", "TRY": "₺", "RUB": "₽", "AUD": "A$", "CAD": "C$",
    "JPY": "¥", "CHF": "Fr.", "MXN": "$"
}
BINANCE_PREFERRED_QUOTES = ["USDT", "USDC", "FDUSD", "BUSD", "TUSD", "DAI", "BTC", "ETH", "BNB"]
MAX_BINANCE_TOP_CRYPTO = 50
BINANCE_TOP_USDT_BASES = [
    "BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "TON", "TRX", "AVAX",
    "SHIB", "DOT", "LINK", "MATIC", "LTC", "BCH", "ICP", "NEAR", "APT", "ARB",
    "FIL", "ATOM", "HBAR", "VET", "GRT", "FTM", "SAND", "AXS", "APE", "OP",
    "RNDR", "INJ", "RUNE", "FLOW", "ALGO", "QNT", "IMX", "MANA", "GALA", "CHZ",
    "LDO", "DYDX", "SUI", "TIA", "SEI", "ETC", "XLM", "EGLD", "STX", "COMP",
    "MKR", "AAVE", "XMR", "UNI", "KAVA", "CRV", "ZIL", "ROSE", "THETA", "IOTA",
    "GMX", "AR", "PYTH", "ARKM", "WIF", "SSV", "JTO", "ENA", "JUP", "NOT",
    "VRA",
]
COINGECKO_MARKETS_URL = "https://api.coingecko.com/api/v3/coins/markets"
CRYPTOCOMPARE_PRICE_URL = "https://min-api.cryptocompare.com/data/price"
_binance_symbols_cache: Dict[str, Dict[str, str]] = {}
_binance_symbols_ts: float = 0.0

RAVA_PERFIL_URL = "https://www.rava.com/perfil/{symbol}"


CUSTOM_CRYPTO_ENTRIES: Dict[str, Dict[str, Any]] = {
    "VRAUSDT": {
        "symbol": "VRAUSDT",
        "base": "VRA",
        "quote": "USDT",
        "display": "Verasity",
        "aliases": ["VERACITYUSDT"],
    },
}


def _build_custom_crypto_map(entries: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    mapping: Dict[str, Dict[str, str]] = {}
    for key, info in entries.items():
        primary = (info.get("symbol") or key or "").upper()
        if not primary:
            continue
        base = (info.get("base") or "").upper()
        quote = (info.get("quote") or "").upper()
        entry: Dict[str, str] = {
            "symbol": primary,
            "base": base,
            "quote": quote,
        }
        display = info.get("display")
        if isinstance(display, str) and display:
            entry["display"] = display
        mapping[primary] = entry
        for alias in info.get("aliases", []):
            alias_up = str(alias).upper()
            if not alias_up:
                continue
            mapping[alias_up] = {**entry}
    return mapping


CUSTOM_CRYPTO_SYMBOLS = _build_custom_crypto_map(CUSTOM_CRYPTO_ENTRIES)


def _merge_custom_crypto_symbols(symbols: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    merged = dict(symbols)
    for key, info in CUSTOM_CRYPTO_SYMBOLS.items():
        merged[key.upper()] = {**info}
    return merged


def _binance_build_fallback() -> Dict[str, Dict[str, str]]:
    fallback: Dict[str, Dict[str, str]] = {}
    for base in BINANCE_TOP_USDT_BASES[:MAX_BINANCE_TOP_CRYPTO]:
        sym = f"{base.upper()}USDT"
        fallback[sym] = {
            "symbol": sym,
            "base": base.upper(),
            "quote": "USDT",
        }
    return _merge_custom_crypto_symbols(fallback)

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

def _sanitize_match(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    stripped = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return re.sub(r"[^A-Z0-9]", "", stripped.upper())

PF_SYMBOL_LOOKUP: Dict[str, Tuple[str, str]] = {}
PF_SYMBOL_SANITIZED_LOOKUP: Dict[str, Tuple[str, str]] = {}
PF_NAME_LOOKUP: Dict[str, Tuple[str, str]] = {}

def _register_pf_symbol(symbol: str, tipo: str):
    key = symbol.upper()
    PF_SYMBOL_LOOKUP[key] = (symbol, tipo)
    PF_SYMBOL_SANITIZED_LOOKUP[_sanitize_match(symbol)] = (symbol, tipo)
    if symbol.endswith(".BA"):
        base = symbol[:-3]
        PF_SYMBOL_LOOKUP[base.upper()] = (symbol, tipo)
        PF_SYMBOL_SANITIZED_LOOKUP[_sanitize_match(base)] = (symbol, tipo)

for sym in ACCIONES_BA:
    _register_pf_symbol(sym, "accion")
for sym in CEDEARS_BA:
    _register_pf_symbol(sym, "cedear")
for sym in BONOS_AR:
    _register_pf_symbol(sym, "bono")
for sym in FCI_LIST:
    _register_pf_symbol(sym, "fci")
for sym in LETES_LIST:
    _register_pf_symbol(sym, "lete")
for cname in CRIPTO_TOP_NAMES:
    csym = _crypto_to_symbol(cname)
    _register_pf_symbol(csym, "cripto")
    PF_SYMBOL_LOOKUP[cname.upper()] = (csym, "cripto")
    PF_SYMBOL_SANITIZED_LOOKUP[_sanitize_match(cname)] = (csym, "cripto")

for sym, name in TICKER_NAME.items():
    entry = PF_SYMBOL_LOOKUP.get(sym.upper())
    if entry:
        PF_NAME_LOOKUP[_sanitize_match(name)] = entry

CEDEARS_SET = {sym.upper() for sym in CEDEARS_BA}

def label_with_currency(sym: str) -> str:
    if sym.endswith(".BA"):
        base_sym = sym[:-3]
        base = f"{TICKER_NAME.get(sym, sym)} ({base_sym})"
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

def pf_guess_symbol(raw: str) -> Optional[Tuple[str, str]]:
    if not raw:
        return None
    query = raw.strip()
    if not query:
        return None
    key = query.upper()
    entry = PF_SYMBOL_LOOKUP.get(key)
    if entry:
        return entry
    sanitized = _sanitize_match(query)
    entry = PF_SYMBOL_SANITIZED_LOOKUP.get(sanitized)
    if entry:
        return entry
    entry = PF_NAME_LOOKUP.get(sanitized)
    if entry:
        return entry
    if key.endswith(".BA"):
        tipo = "cedear" if key in CEDEARS_SET else "accion"
        return key, tipo
    if key.endswith("-USD"):
        return key, "cripto"
    return None

def instrument_currency(sym: str, tipo: str) -> str:
    s = (sym or "").upper()
    t = (tipo or "").lower()
    if s.endswith("-USD"): return "USD"
    if t == "cripto": return "USD"
    if t == "bono": return bono_moneda(sym)
    if t in ("fci", "lete"):
        return "USD" if "USD" in s else "ARS"
    if s.endswith(".BA"): return "ARS"
    if t in ("cedear", "accion"): return "ARS"
    return "ARS"

def price_to_base(price: Optional[float], inst_currency: str, base_currency: str, tc_val: Optional[float]) -> Optional[float]:
    if price is None:
        return None
    if base_currency == inst_currency:
        return float(price)
    if tc_val is None or tc_val <= 0:
        return None
    if base_currency == "ARS" and inst_currency == "USD":
        return float(price) * float(tc_val)
    if base_currency == "USD" and inst_currency == "ARS":
        return float(price) / float(tc_val)
    return None

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

USE_UPSTASH = bool(UPSTASH_URL and UPSTASH_TOKEN)
USE_UPSTASH_REDIS = bool(UPSTASH_REDIS_URL)
STATE_PATH = _writable_path(ENV_STATE_PATH) if not (USE_UPSTASH or USE_UPSTASH_REDIS) else None
ALERTS: Dict[int, List[Dict[str, Any]]] = {}
SUBS: Dict[int, Dict[str, Any]] = {}
PF: Dict[int, Dict[str, Any]] = {}
ALERT_USAGE: Dict[int, Dict[str, Dict[str, Any]]] = {}

_REDIS_CLIENT: Optional[Any] = None
_REDIS_CLIENT_INITIALIZED = False


def _ensure_state_path() -> Optional[str]:
    global STATE_PATH
    if STATE_PATH:
        return STATE_PATH
    STATE_PATH = _writable_path(ENV_STATE_PATH)
    return STATE_PATH


def _get_redis_client():
    global _REDIS_CLIENT, _REDIS_CLIENT_INITIALIZED, USE_UPSTASH_REDIS
    if not USE_UPSTASH_REDIS:
        return None
    if _REDIS_CLIENT_INITIALIZED:
        return _REDIS_CLIENT
    _REDIS_CLIENT_INITIALIZED = True
    try:
        import redis  # type: ignore
    except Exception as e:
        log.warning("Paquete redis no disponible: %s", e)
        USE_UPSTASH_REDIS = False
        _REDIS_CLIENT = None
        return None
    try:
        _REDIS_CLIENT = redis.Redis.from_url(
            UPSTASH_REDIS_URL,
            decode_responses=True,
            socket_timeout=5,
        )
        return _REDIS_CLIENT
    except Exception as e:
        log.warning("No pude inicializar cliente Redis de Upstash: %s", e)
        USE_UPSTASH_REDIS = False
        _REDIS_CLIENT = None
        return None


def _upstash_request(path: str, *, method: str = "GET", data: Optional[str] = None) -> Dict[str, Any]:
    if not USE_UPSTASH:
        raise RuntimeError("Upstash no configurado")
    url = f"{UPSTASH_URL.rstrip('/')}/{path.lstrip('/')}"
    req = urllib.request.Request(url, method=method)
    req.add_header("Authorization", f"Bearer {UPSTASH_TOKEN}")
    if data is not None:
        req.data = data.encode("utf-8")
        req.add_header("Content-Type", "text/plain; charset=utf-8")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", errors="ignore")
        except Exception:
            pass
        raise RuntimeError(f"Upstash HTTP {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Upstash connection error: {e}") from e
    try:
        return json.loads(body)
    except Exception as e:
        raise RuntimeError(f"Upstash invalid response: {body}") from e


def _load_state_from_upstash() -> Optional[Dict[str, Any]]:
    try:
        resp = _upstash_request(f"get/{UPSTASH_STATE_KEY}")
    except Exception as e:
        log.warning("No pude leer estado de Upstash: %s", e)
        return None
    raw = resp.get("result") if isinstance(resp, dict) else None
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception as e:
        log.warning("Estado Upstash inválido: %s", e)
        return None


def _save_state_to_upstash(payload: Dict[str, Any]) -> None:
    data = json.dumps(payload, ensure_ascii=False)
    primary_error: Optional[Exception] = None
    try:
        encoded = quote(data, safe="")
        _upstash_request(f"set/{UPSTASH_STATE_KEY}/{encoded}", method="POST")
        return
    except Exception as e:
        primary_error = e
    try:
        _upstash_request(f"set/{UPSTASH_STATE_KEY}", method="POST", data=data)
    except Exception as e:
        if primary_error:
            log.warning(
                "No pude guardar estado en Upstash (fallback falló tras error primario %s): %s",
                primary_error,
                e,
            )
        else:
            log.warning("No pude guardar estado en Upstash: %s", e)


def _load_state_from_redis() -> Optional[Dict[str, Any]]:
    client = _get_redis_client()
    if not client:
        return None
    try:
        raw = client.get(UPSTASH_STATE_KEY)
    except Exception as e:
        log.warning("No pude leer estado desde Redis Upstash: %s", e)
        return None
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception as e:
        log.warning("Estado Redis Upstash inválido: %s", e)
        return None


def _save_state_to_redis(payload: Dict[str, Any]) -> bool:
    client = _get_redis_client()
    if not client:
        return False
    try:
        client.set(UPSTASH_STATE_KEY, json.dumps(payload, ensure_ascii=False))
        return True
    except Exception as e:
        log.warning("No pude guardar estado en Redis Upstash: %s", e)
        return False


def load_state():
    global ALERTS, SUBS, PF, ALERT_USAGE
    data: Optional[Dict[str, Any]] = None
    if USE_UPSTASH:
        data = _load_state_from_upstash()
    if data is None and USE_UPSTASH_REDIS:
        data = _load_state_from_redis()
    if data is None:
        path = _ensure_state_path()
        if path:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = None
    if data:
        ALERTS = {int(k): v for k, v in data.get("alerts", {}).items()}
        SUBS = {int(k): v for k, v in data.get("subs", {}).items()}
        PF = {int(k): v for k, v in data.get("pf", {}).items()}
        ALERT_USAGE = {int(k): v for k, v in data.get("alert_usage", {}).items()}
        log.info(
            "State loaded. alerts=%d subs=%d pf=%d",
            sum(len(v) for v in ALERTS.values()),
            len(SUBS),
            len(PF),
        )
    else:
        log.info("No previous state found.")


def save_state():
    payload = {"alerts": ALERTS, "subs": SUBS, "pf": PF, "alert_usage": ALERT_USAGE}
    if USE_UPSTASH:
        _save_state_to_upstash(payload)
        return
    if USE_UPSTASH_REDIS and _save_state_to_redis(payload):
        return
    path = _ensure_state_path()
    if not path:
        log.warning("No tengo ruta de estado para guardar en disco.")
        return
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
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

def _fmt_number_generic(n: float, nd: int) -> str:
    s = f"{n:,.{nd}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")

def fmt_crypto_price(n: Optional[float], quote: Optional[str]) -> str:
    if n is None:
        return f"— {quote.upper()}" if quote else "—"
    q = (quote or "").upper()
    symbol = BINANCE_QUOTE_SYMBOL.get(q)
    if symbol:
        nd = 2 if abs(n) >= 1 else (4 if abs(n) >= 0.1 else 6)
        return f"{symbol} {_fmt_number_generic(float(n), nd)}"
    nd = 8 if abs(n) < 1 else 4
    formatted = _fmt_number_generic(float(n), nd)
    return f"{formatted} {q}".strip()

def crypto_display_name(symbol: Optional[str], base: Optional[str], quote: Optional[str]) -> str:
    base_u = (base or "").upper()
    quote_u = (quote or "").upper()
    sym_u = (symbol or "").upper()
    custom = CUSTOM_CRYPTO_SYMBOLS.get(sym_u)
    if custom:
        display = custom.get("display")
        if display and quote_u:
            return f"{display}/{quote_u}"
        if display:
            return display
    if base_u and quote_u:
        return f"{base_u}/{quote_u}"
    if symbol:
        return sym_u
    return "Cripto"

def pct(n: Optional[float], nd: int = 2) -> str:
    try: return f"{n:+.{nd}f}%".replace(".", ",")
    except Exception: return "—"

def pct_plain(n: Optional[float], nd: int = 1) -> str:
    try: return f"{n:.{nd}f}%".replace(".", ",")
    except Exception: return "—"

def format_quantity(sym: str, qty: Optional[float]) -> Optional[str]:
    if qty is None: return None
    try:
        if requires_integer_units(sym):
            return str(int(round(qty)))
        s = f"{qty:.4f}"
        return s.rstrip("0").rstrip(".")
    except Exception:
        return str(qty) if qty is not None else None

def format_added_date(ts: Optional[int]) -> Optional[str]:
    if ts in (None, 0):
        return None
    try:
        dt = datetime.fromtimestamp(int(ts), TZ)
    except Exception:
        return None
    return dt.strftime("%d/%m/%Y")

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


async def get_binance_symbols(session: ClientSession) -> Dict[str, Dict[str, str]]:
    global _binance_symbols_cache, _binance_symbols_ts
    now = time()
    if _binance_symbols_cache and (now - _binance_symbols_ts) < 3600:
        return _binance_symbols_cache
    data = await fetch_json(session, BINANCE_EXCHANGE_INFO_URL)
    if not data:
        if not _binance_symbols_cache:
            _binance_symbols_cache = _binance_build_fallback()
            _binance_symbols_ts = now
        return _binance_symbols_cache
    symbols: Dict[str, Dict[str, str]] = {}
    for entry in data.get("symbols", []):
        try:
            if entry.get("status") != "TRADING":
                continue
            if not entry.get("isSpotTradingAllowed") and "SPOT" not in entry.get("permissions", []):
                continue
            symbol = entry.get("symbol")
            base = entry.get("baseAsset")
            quote = entry.get("quoteAsset")
            if not (symbol and base and quote):
                continue
            symbols[symbol.upper()] = {
                "symbol": symbol.upper(),
                "base": base.upper(),
                "quote": quote.upper(),
            }
        except Exception:
            continue
    if symbols:
        symbols = _merge_custom_crypto_symbols(symbols)
        _binance_symbols_cache = symbols
        _binance_symbols_ts = now
    if not _binance_symbols_cache:
        _binance_symbols_cache = _binance_build_fallback()
        _binance_symbols_ts = now
    return _binance_symbols_cache


async def get_binance_prices(session: ClientSession, symbols: List[str]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    if not symbols:
        return out
    symbol_map: Dict[str, Set[str]] = {}
    norm: List[str] = []
    seen: Set[str] = set()
    for sym in symbols:
        if not sym:
            continue
        sym_u = sym.upper()
        info = CUSTOM_CRYPTO_SYMBOLS.get(sym_u)
        api_symbol = str(info.get("symbol") or sym_u).upper() if info else sym_u
        symbol_map.setdefault(api_symbol, set()).add(sym_u)
        if api_symbol not in seen:
            norm.append(api_symbol)
            seen.add(api_symbol)
    if not norm:
        return out
    chunk_size = 80
    for i in range(0, len(norm), chunk_size):
        chunk = norm[i:i+chunk_size]
        if not chunk:
            continue
        data = None
        for base_url in BINANCE_TICKER_PRICE_URLS:
            url = None
            if len(chunk) == 1:
                url = f"{base_url}?symbol={chunk[0]}"
            else:
                try:
                    payload = quote(json.dumps(chunk))
                    url = f"{base_url}?symbols={payload}"
                except Exception as e:
                    log.warning("binance symbols encode error: %s", e)
                    url = None
            if not url:
                continue
            data = await fetch_json(session, url)
            if data:
                break
        if isinstance(data, dict):
            sym = data.get("symbol")
            price = data.get("price")
            targets = symbol_map.get((sym or "").upper(), {(sym or "").upper()})
            try:
                if sym and price is not None:
                    for target in targets:
                        out[target] = float(price)
            except Exception:
                if sym:
                    for target in targets:
                        out[target] = None
        elif isinstance(data, list):
            for row in data:
                sym = None
                price = None
                if isinstance(row, dict):
                    sym = row.get("symbol")
                    price = row.get("price")
                targets = symbol_map.get((sym or "").upper(), {(sym or "").upper()})
                try:
                    if sym and price is not None:
                        for target in targets:
                            out[target] = float(price)
                except Exception:
                    if sym:
                        for target in targets:
                            out[target] = None
    return out


async def get_binance_price(session: ClientSession, symbol: str) -> Optional[float]:
    prices = await get_binance_prices(session, [symbol])
    return prices.get(symbol.upper())


async def _crypto_price_fallback(
    session: ClientSession,
    *,
    symbol: str,
    base: Optional[str],
    quote: Optional[str],
) -> Optional[float]:
    base_u = (base or "").upper()
    quote_u = (quote or "").upper()
    if not base_u or not quote_u:
        return None
    tsym = quote_u
    if quote_u in {"USDT", "USDC", "BUSD", "FDUSD", "TUSD", "DAI"}:
        tsym = "USD"
    params = {"fsym": base_u, "tsyms": tsym}
    data = await fetch_json(session, CRYPTOCOMPARE_PRICE_URL, params=params)
    if not isinstance(data, dict):
        return None
    if data.get("Response") == "Error":
        return None
    price = data.get(tsym)
    try:
        if price is not None:
            return float(price)
    except Exception:
        return None
    return None


async def get_crypto_prices(
    session: ClientSession,
    symbols: List[str],
    symbols_info: Optional[Dict[str, Dict[str, Optional[str]]]] = None,
) -> Dict[str, Optional[float]]:
    prices: Dict[str, Optional[float]] = {s.upper(): None for s in symbols if s}
    if not prices:
        return {}
    binance_prices = await get_binance_prices(session, list(prices.keys()))
    prices.update(binance_prices)
    missing = [sym for sym, val in prices.items() if val is None]
    if not missing:
        return prices
    info_map = symbols_info or {}
    if not info_map:
        info_map = await get_binance_symbols(session)
    for sym in missing:
        info = info_map.get(sym.upper()) or info_map.get(sym)
        base = info.get("base") if isinstance(info, dict) else None
        quote = info.get("quote") if isinstance(info, dict) else None
        fallback = await _crypto_price_fallback(
            session,
            symbol=sym,
            base=base,
            quote=quote,
        )
        if fallback is not None:
            prices[sym] = fallback
    return prices


async def get_crypto_price(
    session: ClientSession,
    symbol: str,
    *,
    base: Optional[str] = None,
    quote: Optional[str] = None,
) -> Optional[float]:
    info_map: Dict[str, Dict[str, Optional[str]]] = {}
    if base or quote:
        info_map[symbol.upper()] = {"symbol": symbol.upper(), "base": base, "quote": quote}
    prices = await get_crypto_prices(session, [symbol], info_map if info_map else None)
    return prices.get(symbol.upper())

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

# ============================ RAVA ============================

async def _fetch_rava_profile(session: ClientSession, symbol: str) -> Optional[Dict[str, Any]]:
    url = RAVA_PERFIL_URL.format(symbol=quote(symbol))
    try:
        async with session.get(url, headers=REQ_HEADERS, timeout=ClientTimeout(total=12)) as resp:
            if resp.status != 200:
                return None
            html = await resp.text()
    except Exception:
        return None

    match = re.search(r':res="(\{.*?\})"', html, flags=re.S)
    if not match:
        return None
    try:
        data = json.loads(_html.unescape(match.group(1)))
    except Exception:
        return None
    return data

def _rava_history_points(entries: List[Dict[str, Any]]) -> List[Tuple[int, float]]:
    points: List[Tuple[int, float]] = []
    for entry in entries:
        close_raw = entry.get("cierre") if isinstance(entry, dict) else None
        if close_raw is None:
            close_raw = entry.get("ultimo") if isinstance(entry, dict) else None
        try:
            close = float(close_raw)
        except (TypeError, ValueError):
            continue
        ts_raw = entry.get("timestamp") if isinstance(entry, dict) else None
        ts_val: Optional[int] = None
        if ts_raw is not None:
            try:
                ts_val = int(ts_raw)
            except (TypeError, ValueError):
                ts_val = None
        if ts_val is None and isinstance(entry, dict) and entry.get("fecha"):
            try:
                dt = datetime.strptime(entry["fecha"], "%Y-%m-%d").replace(tzinfo=TZ)
                ts_val = int(dt.timestamp())
            except Exception:
                ts_val = None
        if ts_val is None:
            continue
        points.append((ts_val, close))
    points.sort(key=lambda x: x[0])
    return points

def _metrics_from_rava_history(history: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    points = _rava_history_points(history)
    if not points:
        return {"6m": None, "3m": None, "1m": None, "last_ts": None, "vol_ann": None, "dd6m": None,
                "hi52": None, "slope50": None, "trend_flag": None}

    ts = [p[0] for p in points]
    closes = [p[1] for p in points]
    last_ts = ts[-1]
    last = closes[-1]
    prev = closes[-2] if len(closes) >= 2 else None

    def _first_on_or_after(target: int) -> float:
        for t, value in points:
            if t >= target:
                return value
        return points[0][1]

    day = 24 * 3600
    t6 = last_ts - 180 * day
    t3 = last_ts - 90 * day
    t1 = last_ts - 30 * day
    base6 = _first_on_or_after(t6)
    base3 = _first_on_or_after(t3)
    base1 = _first_on_or_after(t1)
    ret6 = ((last / base6) - 1.0) * 100.0 if base6 else None
    ret3 = ((last / base3) - 1.0) * 100.0 if base3 else None
    ret1 = ((last / base1) - 1.0) * 100.0 if base1 else None

    # volatilidad anualizada
    daily_returns: List[float] = []
    for i in range(1, len(closes)):
        if closes[i - 1] and closes[i]:
            try:
                daily_returns.append(closes[i] / closes[i - 1] - 1.0)
            except Exception:
                continue
    window = 60 if len(daily_returns) >= 60 else max(10, len(daily_returns))
    vol_ann = None
    if len(daily_returns) >= 2:
        tail = daily_returns[-window:]
        mu = sum(tail) / len(tail)
        var = sum((r - mu) ** 2 for r in tail) / (len(tail) - 1) if len(tail) > 1 else 0.0
        vol_ann = math.sqrt(var) * math.sqrt(252) * 100.0

    # drawdown últimos 6 meses
    idx_cut = next((i for i, t in enumerate(ts) if t >= t6), 0)
    peak = closes[idx_cut]
    dd_min = 0.0
    for value in closes[idx_cut:]:
        if value > peak:
            peak = value
        draw = value / peak - 1.0
        if draw < dd_min:
            dd_min = draw
    dd6 = abs(dd_min) * 100.0 if dd_min < 0 else 0.0

    hi52 = (last / max(closes) - 1.0) * 100.0 if closes else None

    def _sma(vals: List[float], window: int) -> List[Optional[float]]:
        out: List[Optional[float]] = [None] * len(vals)
        queue: List[float] = []
        acc = 0.0
        for idx, value in enumerate(vals):
            queue.append(value)
            acc += value
            if len(queue) > window:
                acc -= queue.pop(0)
            if len(queue) == window:
                out[idx] = acc / window
        return out

    sma50 = _sma(closes, 50)
    sma200 = _sma(closes, 200)
    slope50 = None
    if sma50[-1] is not None and len(closes) > 20 and sma50[-21] is not None:
        try:
            slope50 = ((sma50[-1] / sma50[-21]) - 1.0) * 100.0
        except Exception:
            slope50 = None
    trend_flag = None
    if sma200[-1] is not None:
        trend_flag = 1.0 if last > sma200[-1] else -1.0

    return {
        "6m": ret6,
        "3m": ret3,
        "1m": ret1,
        "last_ts": int(last_ts),
        "vol_ann": vol_ann,
        "dd6m": dd6,
        "hi52": hi52,
        "slope50": slope50,
        "trend_flag": trend_flag,
        "prev_px": float(prev) if prev is not None else None,
    }

async def _rava_metrics(session: ClientSession, symbol: str) -> Dict[str, Optional[float]]:
    base = {"6m": None, "3m": None, "1m": None, "last_ts": None, "vol_ann": None,
            "dd6m": None, "hi52": None, "slope50": None, "trend_flag": None,
            "last_px": None, "prev_px": None, "last_chg": None}
    data = await _fetch_rava_profile(session, symbol)
    if not data:
        return base

    quotes = data.get("cotizaciones") or []
    history = data.get("coti_hist") or []
    metrics = _metrics_from_rava_history(history)
    base.update(metrics)

    if quotes:
        row = quotes[0]
        last_raw = row.get("ultimo") if isinstance(row, dict) else None
        try:
            last = float(last_raw) if last_raw is not None else None
        except (TypeError, ValueError):
            last = None
        if last is not None:
            base["last_px"] = last
        if row.get("variacion") is not None:
            try:
                base["last_chg"] = float(row.get("variacion"))
            except (TypeError, ValueError):
                base["last_chg"] = None
        if row.get("anterior") is not None:
            try:
                base["prev_px"] = float(row.get("anterior"))
            except (TypeError, ValueError):
                pass
        fecha = row.get("fecha")
        hora = row.get("hora") or "00:00:00"
        if fecha:
            try:
                dt = datetime.strptime(f"{fecha} {hora}", "%Y-%m-%d %H:%M:%S").replace(tzinfo=TZ)
                base["last_ts"] = int(dt.timestamp())
            except Exception:
                pass

    base["currency"] = bono_moneda(symbol)
    return base

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
        prev = closes[idx_last-1] if idx_last >= 1 else None
        last_chg = ((last/prev - 1.0)*100.0) if (prev is not None and prev > 0) else None

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
                "dd6m": dd6, "hi52": hi52, "slope50": slope50, "trend_flag": float(trend_flag),
                "last_px": float(last), "prev_px": float(prev) if prev else None, "last_chg": last_chg}
    except Exception:
        return None

async def _yf_metrics_1y(session: ClientSession, symbol: str) -> Dict[str, Optional[float]]:
    out = {"6m": None, "3m": None, "1m": None, "last_ts": None, "vol_ann": None, "dd6m": None, "hi52": None, "slope50": None,
           "trend_flag": None, "last_px": None, "prev_px": None, "last_chg": None}
    for interval in ("1d", "1wk"):
        res = await _yf_chart_1y(session, symbol, interval)
        if res:
            m = _metrics_from_chart(res)
            if m: out.update(m); break
    return out

async def metrics_for_symbols(session: ClientSession, symbols: List[str]) -> Tuple[Dict[str, Dict[str, Optional[float]]], Optional[int]]:
    out = {s: {"6m": None, "3m": None, "1m": None, "last_ts": None, "vol_ann": None, "dd6m": None, "hi52": None,
               "slope50": None, "trend_flag": None, "last_px": None, "prev_px": None, "last_chg": None} for s in symbols}
    sem = asyncio.Semaphore(4)
    async def work(sym: str):
        async with sem:
            if sym in BONOS_AR:
                out[sym] = await _rava_metrics(session, sym)
            else:
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

def _short_title(text: str, limit: int = 32) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _format_news_item(title: str, link: str) -> str:
    return f"• {anchor(link, title)}\n{_impact_lines(title)}"


def _build_news_layout(news: List[Tuple[str, str]]) -> Tuple[str, Optional[InlineKeyboardMarkup], List[str]]:
    header = "<b>📰 Noticias</b>"
    if not news:
        return header, None, []

    rows: List[List[InlineKeyboardButton]] = []
    current_row: List[InlineKeyboardButton] = []
    body_lines: List[str] = []
    for title, link in news:
        body_lines.append(_format_news_item(title, link))
        btn = InlineKeyboardButton(_short_title(title, 36), url=link)
        current_row.append(btn)
        if len(current_row) == 2:
            rows.append(current_row)
            current_row = []
    if current_row:
        rows.append(current_row)

    markup = InlineKeyboardMarkup(rows) if rows else None
    return header, markup, body_lines


def format_news_block(news: List[Tuple[str, str]]) -> Tuple[str, Optional[InlineKeyboardMarkup]]:
    header, markup, body_lines = _build_news_layout(news)
    body = "\n\n".join(body_lines) if body_lines else "—"
    return f"{header}\n{body}", markup

# ============================ FORMATS & RANKINGS ============================

def _label_long(sym: str) -> str: return label_with_currency(sym)
def _label_short(sym: str) -> str:
    if sym.endswith(".BA"): return f"{NAME_ABBR.get(sym, sym)} ({sym[:-3]})"
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
        # La tabla se muestra desde la perspectiva del usuario que compraría dólares
        # al precio "venta" de la casa y vendería al precio "compra".
        compra = fmt_money_ars(venta_val) if venta_val is not None else "—"
        venta = fmt_money_ars(compra_val) if compra_val is not None else "—"
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

def _expected_daily_return(m: Dict[str, Optional[float]]) -> float:
    components: List[Tuple[float, float]] = []
    for months, key, weight in ((1, "1m", 0.5), (3, "3m", 0.3), (6, "6m", 0.2)):
        val = m.get(key)
        if val is None:
            continue
        try:
            lr = math.log1p(float(val) / 100.0) / (21 * months)
            components.append((lr, weight))
        except Exception:
            continue
    if components:
        wtot = sum(w for _, w in components)
        mu = sum(lr * w for lr, w in components) / wtot if wtot else 0.0
    else:
        mu = 0.0

    vol_ann = m.get("vol_ann")
    if vol_ann is not None:
        try:
            vol_daily = (float(vol_ann) / 100.0) / math.sqrt(252)
            mu -= 0.5 * (vol_daily ** 2)
        except Exception:
            pass

    slope = m.get("slope50")
    if slope is not None:
        try:
            mu += (float(slope) / 100.0) * 0.003
        except Exception:
            pass

    trend_flag = m.get("trend_flag")
    if trend_flag is not None:
        try:
            mu += float(trend_flag) * 0.0005
        except Exception:
            pass

    hi52 = m.get("hi52")
    if hi52 is not None:
        try:
            hi_adj = -float(hi52) / 100.0
            hi_adj = max(-0.25, min(0.25, hi_adj))
            mu += hi_adj * 0.01
        except Exception:
            pass

    dd6 = m.get("dd6m")
    if dd6 is not None:
        try:
            mu -= max(0.0, float(dd6) - 12.0) / 100.0 * 0.003
        except Exception:
            pass

    return mu

def projection_3m(m: Dict[str, Optional[float]]) -> float:
    mu = _expected_daily_return(m)
    return (math.exp(mu * 63) - 1.0) * 100.0

def projection_6m(m: Dict[str, Optional[float]]) -> float:
    mu = _expected_daily_return(m)
    vol_ann = m.get("vol_ann")
    if vol_ann is not None:
        try:
            penalty = (float(vol_ann) / 100.0) ** 2 * 0.0005
            mu -= penalty
        except Exception:
            pass
    return (math.exp(mu * 126) - 1.0) * 100.0

async def _rank_top3(update: Update, symbols: List[str], title: str):
    async with ClientSession() as session:
        mets, last_ts = await metrics_for_symbols(session, symbols)
        fecha = datetime.fromtimestamp(last_ts, TZ).strftime("%d/%m/%Y") if last_ts else None
        pairs = sorted([(sym, m["6m"]) for sym, m in mets.items() if m.get("6m") is not None], key=lambda x: x[1], reverse=True)
        top_syms = [sym for sym, _ in pairs[:3]]
        msg = format_top3_table(title, fecha, top_syms, mets)
        await update.effective_message.reply_text(
            msg,
            parse_mode=ParseMode.HTML,
            link_preview_options=LinkPreviewOptions(is_disabled=True),
        )

        if HAS_MPL and pairs:
            chart_rows: List[Tuple[str, List[Optional[float]]]] = []
            for sym, _ in pairs[:3]:
                metrics = mets.get(sym, {})
                values: List[Optional[float]] = []
                for key in ("1m", "3m", "6m"):
                    raw_val = metrics.get(key)
                    if raw_val is None:
                        values.append(None)
                        continue
                    try:
                        values.append(float(raw_val))
                    except Exception:
                        values.append(None)
                if not any(v is not None for v in values):
                    continue
                chart_rows.append((_label_short(sym), values))
            subtitle = f"Datos al {fecha}" if fecha else None
            img = _bar_image_from_rank(
                chart_rows,
                title=f"{title} — Rendimientos 1/3/6M",
                subtitle=subtitle,
                series_labels=["Rend. 1M", "Rend. 3M", "Rend. 6M"],
            )
            if img:
                await update.effective_message.reply_photo(photo=img)


async def _rank_proj5(update: Update, symbols: List[str], title: str):
    async with ClientSession() as session:
        mets, last_ts = await metrics_for_symbols(session, symbols)
        fecha = datetime.fromtimestamp(last_ts, TZ).strftime("%d/%m/%Y") if last_ts else None
        rows = []
        for sym, m in mets.items():
            if m.get("6m") is None:
                continue
            rows.append((sym, projection_3m(m), projection_6m(m)))
        rows.sort(key=lambda x: x[2], reverse=True)
        top_rows = rows[:5]
        msg = format_proj_dual(title, fecha, top_rows)
        await update.effective_message.reply_text(
            msg,
            parse_mode=ParseMode.HTML,
            link_preview_options=LinkPreviewOptions(is_disabled=True),
        )

        if HAS_MPL and top_rows:
            chart_rows = []
            for sym, p3, p6 in top_rows:
                chart_rows.append((_label_short(sym), [p3, p6]))
            subtitle = f"Datos al {fecha}" if fecha else None
            img = _bar_image_from_rank(
                chart_rows,
                title=f"{title} — Proyecciones",
                subtitle=subtitle,
                series_labels=["Proy. 3M", "Proy. 6M"],
            )
            if img:
                await update.effective_message.reply_photo(photo=img)

# ============================ COMANDOS / MENÚS ============================

def set_menu_counter(context: ContextTypes.DEFAULT_TYPE, name: str, n: int):
    context.user_data.setdefault("menu_counts", {})[name] = n
def dec_and_maybe_show(update: Update, context: ContextTypes.DEFAULT_TYPE, name: str, show_func):
    cnt = context.user_data.get("menu_counts", {}).get(name, 0)
    cnt = max(0, cnt-1)
    context.user_data["menu_counts"][name] = cnt
    if cnt > 0:
        return show_func(update, context)

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    intro = (
        "<b>¡Hola! Soy tu asistente de mercados argentinos.</b>\n"
        "<i>Seguimiento de dólar, bonos, acciones, portafolio y alertas en un mismo lugar.</i>\n\n"
        "Elegí una opción rápida o usá los comandos clásicos:\n"
        "• /economia — Panel macro: dólares, reservas, inflación, riesgo y noticias\n"
        "• /acciones — Rankings y proyecciones de acciones\n"
        "• /cedears — Rankings y proyecciones de cedears\n"
        "• /alertas_menu — Gestioná alertas personalizadas\n"
        "• /portafolio — Armá y analizá tu cartera\n"
        "• /subs — Suscripción al resumen diario\n"
    )

    kb_rows = [
        [
            InlineKeyboardButton("💵 Dólar y Reservas", callback_data="ECO:DOLAR"),
            InlineKeyboardButton("📰 Noticias", callback_data="ECO:NOTICIAS"),
        ],
        [
            InlineKeyboardButton("📈 Acciones Top 3", callback_data="ACC:TOP3"),
            InlineKeyboardButton("🏁 Acciones Proyección", callback_data="ACC:TOP5"),
        ],
        [
            InlineKeyboardButton("🌎 Cedears Top 3", callback_data="CED:TOP3"),
            InlineKeyboardButton("🌐 Cedears Proyección", callback_data="CED:TOP5"),
        ],
        [
            InlineKeyboardButton("🔔 Mis alertas", callback_data="AL:LIST"),
            InlineKeyboardButton("🧾 Resumen diario", callback_data="ST:SUBS"),
        ],
        [InlineKeyboardButton("💼 Portafolio", callback_data="PF:MENU")],
    ]

    await update.effective_message.reply_text(
        intro,
        parse_mode=ParseMode.HTML,
        reply_markup=InlineKeyboardMarkup(kb_rows),
        link_preview_options=LinkPreviewOptions(is_disabled=True),
    )

async def cmd_dolar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    async with ClientSession() as session:
        data = await get_dolares(session)
    msg = format_dolar_message(data) if data else "No pude obtener cotizaciones ahora."
    await update.effective_message.reply_text(msg, parse_mode=ParseMode.HTML, link_preview_options=LinkPreviewOptions(is_disabled=True))

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
    await update.effective_message.reply_text("🌎 Menú cedears", reply_markup=kb_menu)

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
    await update.effective_message.reply_text(txt, parse_mode=ParseMode.HTML)

async def cmd_inflacion(update: Update, context: ContextTypes.DEFAULT_TYPE):
    async with ClientSession() as session:
        tup = await get_inflacion_mensual(session)
    if tup is None:
        txt = "No pude obtener inflación ahora."
    else:
        val, fecha = tup; val_str = str(round(val,1)).replace(".", ",")
        txt = f"<b>📉 Inflación Mensual</b>{f' <i>{fecha}</i>' if fecha else ''}\n<b>{val_str}%</b>\n<i>Fuente: ArgentinaDatos</i>"
    await update.effective_message.reply_text(txt, parse_mode=ParseMode.HTML)

async def cmd_riesgo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    async with ClientSession() as session:
        tup = await get_riesgo_pais(session)
    if tup is None:
        txt = "No pude obtener riesgo país ahora."
    else:
        rp, f = tup; f_str = parse_iso_ddmmyyyy(f)
        txt = f"<b>📈 Riesgo País</b>{f' <i>{f_str}</i>' if f_str else ''}\n<b>{rp} pb</b>\n<i>Fuente: ArgentinaDatos</i>"
    await update.effective_message.reply_text(txt, parse_mode=ParseMode.HTML)

async def cmd_noticias(update: Update, context: ContextTypes.DEFAULT_TYPE):
    async with ClientSession() as session:
        news = await fetch_rss_entries(session, limit=5)
    header, kb, items = _build_news_layout(news or [])
    header_body = f"{header}\n—" if not items else header
    await update.effective_message.reply_text(
        header_body,
        parse_mode=ParseMode.HTML,
        reply_markup=kb,
        link_preview_options=LinkPreviewOptions(is_disabled=True),
    )
    for item in items:
        await update.effective_message.reply_text(
            item,
            parse_mode=ParseMode.HTML,
            link_preview_options=LinkPreviewOptions(prefer_small_media=True),
        )

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

AL_KIND, AL_FX_TYPE, AL_FX_SIDE, AL_OP, AL_MODE, AL_VALUE, AL_METRIC_TYPE, AL_TICKER, AL_CRYPTO = range(9)
ALERTS_SILENT_UNTIL: Dict[int, float] = {}
ALERTS_PAUSED: Set[int] = set()


def _float_equals(a: Any, b: Any, abs_tol: float = 1e-6) -> bool:
    try:
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=abs_tol)
    except Exception:
        return False


def _alerts_match(existing: Dict[str, Any], candidate: Dict[str, Any]) -> bool:
    if existing.get("kind") != candidate.get("kind"):
        return False
    kind = candidate.get("kind")
    if kind == "fx":
        return (
            existing.get("type") == candidate.get("type")
            and existing.get("side") == candidate.get("side")
            and existing.get("op") == candidate.get("op")
            and _float_equals(existing.get("value"), candidate.get("value"))
        )
    if kind == "metric":
        return (
            existing.get("type") == candidate.get("type")
            and existing.get("op") == candidate.get("op")
            and _float_equals(existing.get("value"), candidate.get("value"))
        )
    if kind == "crypto":
        return (
            (existing.get("symbol") or "").upper() == (candidate.get("symbol") or "").upper()
            and (existing.get("base") or "").upper() == (candidate.get("base") or "").upper()
            and (existing.get("quote") or "").upper() == (candidate.get("quote") or "").upper()
            and existing.get("op") == candidate.get("op")
            and _float_equals(existing.get("value"), candidate.get("value"))
        )
    if kind == "ticker":
        return (
            existing.get("symbol") == candidate.get("symbol")
            and existing.get("op") == candidate.get("op")
            and _float_equals(existing.get("value"), candidate.get("value"))
        )
    return False


def _has_duplicate_alert(rules: List[Dict[str, Any]], candidate: Dict[str, Any]) -> bool:
    return any(_alerts_match(r, candidate) for r in rules)

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


def _build_crypto_top_rows(
    symbols_map: Dict[str, Dict[str, str]],
    bases: List[str],
) -> Optional[List[List[Tuple[str, str]]]]:
    rows_data: List[Tuple[str, str]] = []
    for base in bases:
        symbol = f"{base.upper()}USDT"
        info = symbols_map.get(symbol)
        if not info:
            continue
        label = info.get("display") or info.get("base") or base.upper()
        rows_data.append((label, f"CRYPTOSEL:{info.get('symbol')}"))
        if len(rows_data) >= MAX_BINANCE_TOP_CRYPTO:
            break
    if not rows_data:
        return None
    rows: List[List[Tuple[str, str]]] = []
    row: List[Tuple[str, str]] = []
    for item in rows_data:
        row.append(item)
        if len(row) == 2:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    rows.append([("🔍 Buscar manualmente", "CRYPTO:SEARCH")])
    rows.append([("⬅️ Volver", "BACK:KIND"), ("Cancelar", "CANCEL")])
    return rows


async def alertas_prompt_crypto_manual(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    query: Optional["CallbackQuery"] = None,
    prefix: Optional[str] = None,
) -> int:
    msg = (
        "Ingresá la criptomoneda que querés alertar (ej: BTCUSDT, BTC/USDT, BTC).\n"
        "Escribila en el chat."
    )
    if prefix:
        msg = f"{prefix}\n\n{msg}"
    markup = kb([[("⬅️ Volver al listado", "BACK:CRYPTO_LIST"), ("Cancelar", "CANCEL")]])
    if query:
        await query.edit_message_text(msg, reply_markup=markup)
    else:
        await update.effective_message.reply_text(msg, reply_markup=markup)
    return AL_CRYPTO


async def alertas_show_crypto_list(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    query: Optional["CallbackQuery"] = None,
    prefix: Optional[str] = None,
) -> int:
    text = "Elegí la criptomoneda:"
    if prefix:
        text = f"{prefix}\n\n{text}"
    async with ClientSession() as session:
        symbols_map = await get_binance_symbols(session)
        top_bases = await get_top_crypto_bases(session, symbols_map)
    rows = _build_crypto_top_rows(symbols_map, top_bases)
    if rows:
        if query:
            await query.edit_message_text(text, reply_markup=kb(rows))
        else:
            await update.effective_message.reply_text(text, reply_markup=kb(rows))
        return AL_CRYPTO
    fallback_prefix = prefix or "No pude obtener el listado de criptomonedas."
    return await alertas_prompt_crypto_manual(update, context, query=query, prefix=fallback_prefix)

def _parse_float_user_strict(s: str) -> Optional[float]:
    s = (s or "").strip()
    if re.search(r"[^\d\.,\-+]", s): return None
    s = s.replace(".", "").replace(",", ".")
    try: return float(s)
    except Exception: return None

def _fx_display_value(row: Dict[str, Any], side: str) -> Optional[float]:
    if side == "compra": return row.get("venta")
    if side == "venta":  return row.get("compra")
    return None

def _sort_crypto_matches(matches: List[Dict[str, str]]) -> List[Dict[str, str]]:
    def key(info: Dict[str, str]):
        quote = info.get("quote", "")
        try:
            idx = BINANCE_PREFERRED_QUOTES.index(quote)
        except ValueError:
            idx = len(BINANCE_PREFERRED_QUOTES)
        return (idx, info.get("symbol", ""))
    return sorted(matches, key=key)


async def get_top_crypto_bases(
    session: ClientSession,
    symbols_map: Dict[str, Dict[str, str]],
    limit: int = MAX_BINANCE_TOP_CRYPTO,
) -> List[str]:
    available_usdt: Set[str] = {
        (info.get("base") or "").upper()
        for info in symbols_map.values()
        if (info.get("quote") or "").upper() == "USDT"
    }
    bases: List[str] = []
    seen: Set[str] = set()
    if available_usdt and limit > 0:
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": str(min(limit, 250)),
            "page": "1",
            "sparkline": "false",
        }
        data = await fetch_json(session, COINGECKO_MARKETS_URL, params=params)
        if isinstance(data, list):
            for entry in data:
                base = (entry.get("symbol") or "").upper()
                if not base or base in seen or base not in available_usdt:
                    continue
                bases.append(base)
                seen.add(base)
                if len(bases) >= limit:
                    break
    if len(bases) < limit:
        for base in BINANCE_TOP_USDT_BASES:
            up = base.upper()
            if up in seen or up not in available_usdt:
                continue
            bases.append(up)
            seen.add(up)
            if len(bases) >= limit:
                break
    return bases


def kb_alertas_menu() -> InlineKeyboardMarkup:
    return kb([
        [("Listar","AL:LIST"),("Agregar","AL:ADD")],
        [("Borrar","AL:CLEAR")],
        [("Pausar","AL:PAUSE"),("Reanudar","AL:RESUME")],
    ])


async def cmd_alertas_menu(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    prefix: Optional[str] = None,
    *,
    edit: bool = False,
) -> None:
    text = "🔔 Menú Alertas" if not prefix else f"{prefix}\n\n🔔 Menú Alertas"
    markup = kb_alertas_menu()
    if edit and update.callback_query:
        await update.callback_query.edit_message_text(text, reply_markup=markup)
    else:
        await update.effective_message.reply_text(text, reply_markup=markup)

async def alertas_menu_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    data = q.data
    if data == "AL:LIST":
        await cmd_alertas_list(update, context)
        await cmd_alertas_menu(update, context)
    elif data == "AL:CLEAR":
        await cmd_alertas_clear(update, context)
    elif data == "AL:PAUSE":
        await cmd_alertas_pause(update, context)
    elif data == "AL:RESUME":
        await cmd_alertas_resume(update, context)
        await cmd_alertas_menu(update, context)

async def cmd_alertas_list(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    rules = ALERTS.get(chat_id, [])
    now = datetime.now(TZ)
    if not rules:
        txt = "No tenés alertas configuradas.\nUsá /alertas_menu → Agregar."
    else:
        lines = ["<b>🔔 Alertas Configuradas</b>"]
        for i, r in enumerate(rules, 1):
            label = _alert_rule_label(r)
            status = _alert_pause_status(r, now=now)
            if status:
                lines.append(f"{i}. {label} <i>({status})</i>")
            else:
                lines.append(f"{i}. {label}")
        if chat_id in ALERTS_PAUSED:
            lines.append("\n<i>Alertas en pausa (indefinida)</i>")
        elif chat_id in ALERTS_SILENT_UNTIL and ALERTS_SILENT_UNTIL[chat_id] > datetime.now(TZ).timestamp():
            until = datetime.fromtimestamp(ALERTS_SILENT_UNTIL[chat_id], TZ)
            lines.append(f"\n<i>Alertas en pausa hasta {until.strftime('%d/%m %H:%M')}</i>")
        txt = "\n".join(lines)
    await update.effective_message.reply_text(txt, parse_mode=ParseMode.HTML)

async def cmd_alertas_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    rules = ALERTS.get(chat_id, [])
    if not rules:
        await update.effective_message.reply_text("No tenés alertas guardadas.")
        await cmd_alertas_menu(update, context)
        return
    buttons: List[List[Tuple[str,str]]] = []
    for i, r in enumerate(rules, 1):
        if r.get("kind") == "fx":
            label = f"{i}. {r['type'].upper()}({r['side']}) {html_op(r['op'])} {fmt_money_ars(r['value'])}"
        elif r.get("kind") == "metric":
            if r["type"]=="riesgo":     val = f"{r['value']:.0f} pb"
            elif r["type"]=="reservas": val = f"{fmt_number(r['value'],0)} MUS$"
            else:                       val = f"{str(round(r['value'],1)).replace('.',',')}%"
            label = f"{i}. {r['type'].upper()} {html_op(r['op'])} {val}"
        elif r.get("kind") == "crypto":
            label = f"{i}. {crypto_display_name(r.get('symbol'), r.get('base'), r.get('quote'))} {html_op(r['op'])} {fmt_crypto_price(r['value'], r.get('quote'))}"
        else:
            label = f"{i}. {_label_long(r['symbol'])} {html_op(r['op'])} {fmt_money_ars(r['value'])}"
        buttons.append([(label, f"CLR:{i-1}")])
    buttons.append([("Borrar Todas","CLR:ALL")])
    buttons.append([("Volver","CLR:BACK"), ("Cancelar","CLR:CANCEL")])
    await update.effective_message.reply_text("Elegí qué alerta borrar:", reply_markup=InlineKeyboardMarkup(
        [[InlineKeyboardButton(t, callback_data=d) for t,d in row] for row in buttons]
    ))

async def alertas_clear_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    chat_id = q.message.chat_id
    rules = ALERTS.get(chat_id, [])
    data = q.data.split(":",1)[1]
    if data == "CANCEL":
        await cmd_alertas_menu(update, context, prefix="Operación cancelada.", edit=True); return
    if data == "BACK":
        await cmd_alertas_menu(update, context, edit=True); return
    if data == "ALL":
        cnt = len(rules); ALERTS[chat_id] = []; save_state()
        await q.edit_message_text(f"Se eliminaron {cnt} alertas."); return
    try: idx = int(data)
    except Exception:
        await q.edit_message_text("Acción inválida."); return
    if 0 <= idx < len(rules):
        rules.pop(idx); save_state(); await q.edit_message_text("Alerta eliminada.")
    else:
        await q.edit_message_text("Número fuera de rango.")

def _pause_list_markup(chat_id: int) -> InlineKeyboardMarkup:
    rules = ALERTS.get(chat_id, []) or []
    now = datetime.now(TZ)
    rows: List[List[InlineKeyboardButton]] = []
    for i, rule in enumerate(rules, 1):
        status = _alert_pause_status_short(rule, now=now)
        label = f"{i}. {_alert_rule_label(rule)}{status}"[:64]
        rows.append([InlineKeyboardButton(label, callback_data=f"AP:SEL:{i-1}")])
    if rows:
        rows.append([
            InlineKeyboardButton("Pausar todas", callback_data="AP:SEL:ALL"),
            InlineKeyboardButton("Reanudar todas", callback_data="AP:DO:ALL:RESUME"),
        ])
    rows.append([
        InlineKeyboardButton("Volver", callback_data="AP:BACK"),
        InlineKeyboardButton("Cerrar", callback_data="AP:CLOSE"),
    ])
    return InlineKeyboardMarkup(rows)


def _pause_options_markup(target: str, *, is_all: bool = False) -> InlineKeyboardMarkup:
    resume_text = "Reanudar todas" if is_all else "Reanudar"
    rows = [
        [InlineKeyboardButton("Pausar (Indefinida)", callback_data=f"AP:DO:{target}:INF")],
        [
            InlineKeyboardButton("Pausar 1h", callback_data=f"AP:DO:{target}:1"),
            InlineKeyboardButton("Pausar 3h", callback_data=f"AP:DO:{target}:3"),
        ],
        [
            InlineKeyboardButton("Pausar 6h", callback_data=f"AP:DO:{target}:6"),
            InlineKeyboardButton("Pausar 12h", callback_data=f"AP:DO:{target}:12"),
        ],
        [
            InlineKeyboardButton("Pausar 24h", callback_data=f"AP:DO:{target}:24"),
            InlineKeyboardButton(resume_text, callback_data=f"AP:DO:{target}:RESUME"),
        ],
        [
            InlineKeyboardButton("Volver", callback_data="AP:LIST"),
            InlineKeyboardButton("Cancelar", callback_data="AP:CLOSE"),
        ],
    ]
    return InlineKeyboardMarkup(rows)


async def cmd_alertas_pause(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    prefix: Optional[str] = None,
    *,
    edit: bool = False,
) -> None:
    chat_id = update.effective_chat.id
    rules = ALERTS.get(chat_id, []) or []
    if not rules:
        msg = "No tenés alertas guardadas."
        if prefix:
            msg = f"{prefix}\n\n{msg}"
        if edit and update.callback_query:
            await update.callback_query.edit_message_text(msg)
        else:
            await update.effective_message.reply_text(msg)
        await cmd_alertas_menu(update, context)
        return
    text = "Elegí qué alertas pausar:"
    if prefix:
        text = f"{prefix}\n\n{text}"
    markup = _pause_list_markup(chat_id)
    if edit and update.callback_query:
        await update.callback_query.edit_message_text(text, reply_markup=markup)
    else:
        await update.effective_message.reply_text(text, reply_markup=markup)

async def alerts_pause_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    chat_id = q.message.chat_id
    parts = q.data.split(":")
    if len(parts) < 2:
        await q.answer("Acción inválida.", show_alert=True); return
    action = parts[1]
    if action == "CLOSE":
        await q.edit_message_text("Listo."); return
    if action == "BACK":
        await cmd_alertas_menu(update, context, edit=True); return
    if action == "LIST":
        await cmd_alertas_pause(update, context, edit=True); return
    if action == "SEL":
        target = parts[2] if len(parts) > 2 else None
        rules = ALERTS.get(chat_id, []) or []
        if not rules:
            await cmd_alertas_pause(update, context, prefix="No tenés alertas para pausar.", edit=True)
            return
        if target == "ALL":
            text = "Elegí duración para todas las alertas."
            if chat_id in ALERTS_PAUSED:
                text += "\nActualmente están en pausa indefinida."
            elif chat_id in ALERTS_SILENT_UNTIL and ALERTS_SILENT_UNTIL[chat_id] > datetime.now(TZ).timestamp():
                until = datetime.fromtimestamp(ALERTS_SILENT_UNTIL[chat_id], TZ)
                text += f"\nEn pausa hasta {until.strftime('%d/%m %H:%M')}."
            await q.edit_message_text(text, reply_markup=_pause_options_markup("ALL", is_all=True))
            return
        try:
            idx = int(target) if target is not None else -1
        except Exception:
            await q.answer("Alerta inválida.", show_alert=True); return
        if idx < 0 or idx >= len(rules):
            await q.answer("Alerta inexistente.", show_alert=True); return
        rule = rules[idx]
        label = _alert_rule_label(rule)
        status = _alert_pause_status(rule)
        text = f"Elegí duración para la alerta {idx+1}:\n{label[:120]}"
        if status:
            text += f"\n{status}"
        await q.edit_message_text(text, reply_markup=_pause_options_markup(str(idx)))
        return
    if action == "DO":
        if len(parts) < 4:
            await q.answer("Acción inválida.", show_alert=True); return
        target = parts[2]
        op = parts[3]
        rules = ALERTS.get(chat_id, []) or []
        if target == "ALL":
            if op == "RESUME":
                ALERTS_PAUSED.discard(chat_id); ALERTS_SILENT_UNTIL.pop(chat_id, None)
                changed = False
                for rule in rules:
                    if rule.pop("pause_indef", None) is not None:
                        changed = True
                    if rule.pop("pause_until", None) is not None:
                        changed = True
                if changed:
                    save_state()
                await cmd_alertas_pause(update, context, prefix="🔔 Alertas reanudadas.", edit=True)
                return
            if op == "INF":
                ALERTS_PAUSED.add(chat_id); ALERTS_SILENT_UNTIL.pop(chat_id, None)
                save_state()
                await cmd_alertas_pause(update, context, prefix="🔕 Alertas en pausa (indefinida).", edit=True)
                return
            try:
                hrs = int(op)
            except Exception:
                await q.answer("Acción inválida.", show_alert=True); return
            until = datetime.now(TZ) + timedelta(hours=hrs)
            ALERTS_SILENT_UNTIL[chat_id] = until.timestamp(); ALERTS_PAUSED.discard(chat_id)
            save_state()
            await cmd_alertas_pause(
                update,
                context,
                prefix=f"🔕 Alertas en pausa por {hrs}h (hasta {until.strftime('%d/%m %H:%M')}).",
                edit=True,
            )
            return
        try:
            idx = int(target)
        except Exception:
            await q.answer("Acción inválida.", show_alert=True); return
        if idx < 0 or idx >= len(rules):
            await q.answer("Alerta inexistente.", show_alert=True); return
        rule = rules[idx]
        if op == "RESUME":
            changed = False
            if rule.pop("pause_indef", None) is not None:
                changed = True
            if rule.pop("pause_until", None) is not None:
                changed = True
            ALERTS_PAUSED.discard(chat_id); ALERTS_SILENT_UNTIL.pop(chat_id, None)
            if changed:
                save_state()
            await cmd_alertas_pause(update, context, prefix=f"🔔 Alerta {idx+1} reanudada.", edit=True)
            return
        if op == "INF":
            rule["pause_indef"] = True
            rule.pop("pause_until", None)
            ALERTS_PAUSED.discard(chat_id); ALERTS_SILENT_UNTIL.pop(chat_id, None)
            save_state()
            await cmd_alertas_pause(update, context, prefix=f"🔕 Alerta {idx+1} en pausa indefinida.", edit=True)
            return
        try:
            hrs = int(op)
        except Exception:
            await q.answer("Acción inválida.", show_alert=True); return
        until = datetime.now(TZ) + timedelta(hours=hrs)
        rule["pause_until"] = until.timestamp()
        rule.pop("pause_indef", None)
        ALERTS_PAUSED.discard(chat_id); ALERTS_SILENT_UNTIL.pop(chat_id, None)
        save_state()
        await cmd_alertas_pause(
            update,
            context,
            prefix=f"🔕 Alerta {idx+1} en pausa por {hrs}h (hasta {until.strftime('%d/%m %H:%M')}).",
            edit=True,
        )
        return
    await q.answer("Acción inválida.", show_alert=True)

async def cmd_alertas_resume(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ALERTS_PAUSED.discard(chat_id); ALERTS_SILENT_UNTIL.pop(chat_id, None)
    rules = ALERTS.get(chat_id, []) or []
    changed = False
    for rule in rules:
        if rule.pop("pause_indef", None) is not None:
            changed = True
        if rule.pop("pause_until", None) is not None:
            changed = True
    if changed:
        save_state()
    await update.effective_message.reply_text("🔔 Alertas reanudadas.")

# ---- Conversación Agregar Alerta ----

FX_TYPE_LABELS = {
    "oficial": "Dólar Oficial",
    "mayorista": "Dólar Mayorista",
    "blue": "Dólar Blue",
    "mep": "Dólar MEP",
    "ccl": "Dólar CCL",
    "tarjeta": "Dólar Tarjeta",
    "cripto": "Dólar Cripto",
}

METRIC_TYPE_LABELS = {
    "riesgo": "Riesgo País",
    "inflacion": "Inflación Mensual",
    "reservas": "Reservas BCRA",
}

SIDE_LABELS = {"compra": "Compra", "venta": "Venta"}


def _get_rule_pause_until(rule: Dict[str, Any]) -> Optional[float]:
    raw = rule.get("pause_until")
    if raw is None:
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    if math.isnan(val) or math.isinf(val):
        return None
    return val


def _alert_pause_status(rule: Dict[str, Any], now: Optional[datetime] = None) -> Optional[str]:
    if rule.get("pause_indef"):
        return "⏸️ Pausada indefinidamente"
    until = _get_rule_pause_until(rule)
    if until is None:
        return None
    ref = now or datetime.now(TZ)
    if until <= ref.timestamp():
        return None
    until_dt = datetime.fromtimestamp(until, TZ)
    return f"⏸️ Pausada hasta {until_dt.strftime('%d/%m %H:%M')}"


def _alert_pause_status_short(rule: Dict[str, Any], now: Optional[datetime] = None) -> str:
    if rule.get("pause_indef"):
        return " ⏸️"
    until = _get_rule_pause_until(rule)
    if until is None:
        return ""
    ref = now or datetime.now(TZ)
    if until <= ref.timestamp():
        return ""
    until_dt = datetime.fromtimestamp(until, TZ)
    fmt = "%H:%M" if until_dt.date() == ref.date() else "%d/%m %H:%M"
    return f" ⏸️ hasta {until_dt.strftime(fmt)}"


def _alert_rule_label(rule: Dict[str, Any]) -> str:
    kind = rule.get("kind")
    if kind == "fx":
        t = rule.get("type") or ""
        side = rule.get("side") or ""
        label = FX_TYPE_LABELS.get(t, t.upper())
        side_label = SIDE_LABELS.get(side, side)
        return f"{label} ({side_label}) {html_op(rule.get('op', ''))} {fmt_money_ars(rule.get('value'))}"
    if kind == "metric":
        t = rule.get("type") or ""
        op = rule.get("op", "")
        v = rule.get("value")
        if t == "riesgo":
            val = f"{(v or 0):.0f} pb"
        elif t == "reservas":
            val = f"{fmt_number(v, 0)} MUS$"
        else:
            try:
                val = f"{str(round(float(v or 0), 1)).replace('.', ',')}%"
            except Exception:
                val = f"{v}%"
        label = METRIC_TYPE_LABELS.get(t, t.upper())
        return f"{label} {html_op(op)} {val}"
    if kind == "crypto":
        label = crypto_display_name(rule.get("symbol"), rule.get("base"), rule.get("quote"))
        op = rule.get("op", "")
        v = rule.get("value")
        return f"{label} (Precio) {html_op(op)} {fmt_crypto_price(v, rule.get('quote'))}"
    sym = rule.get("symbol")
    op = rule.get("op", "")
    v = rule.get("value")
    return f"{_label_long(sym)} (Precio) {html_op(op)} {fmt_money_ars(v)}"


def _is_rule_paused(rule: Dict[str, Any], now_ts: Optional[float] = None) -> Tuple[bool, bool]:
    if rule.get("pause_indef"):
        return True, False
    until = _get_rule_pause_until(rule)
    if until is None:
        if "pause_until" in rule:
            rule.pop("pause_until", None)
            return False, True
        return False, False
    ref_ts = now_ts if now_ts is not None else datetime.now(TZ).timestamp()
    if until > ref_ts:
        return True, False
    rule.pop("pause_until", None)
    return False, True

def _alert_usage_key(kind: str, meta: Dict[str, Any]) -> Optional[str]:
    if kind == "fx":
        t, side, op = meta.get("type"), meta.get("side"), meta.get("op")
        if t and side and op:
            return f"fx:{t}:{side}:{op}"
        return None
    if kind == "metric":
        t, op = meta.get("type"), meta.get("op")
        if t and op:
            return f"metric:{t}:{op}"
        return None
    if kind == "crypto":
        sym, op = (meta.get("symbol") or "").upper(), meta.get("op")
        if sym and op:
            mode = meta.get("mode") or ""
            return f"crypto:{sym}:{op}:{mode}"
        return None
    if kind == "ticker":
        sym, op = (meta.get("symbol") or "").upper(), meta.get("op")
        if sym and op:
            return f"ticker:{sym}:{op}"
        return None
    return None


def _alert_usage_label(kind: str, meta: Dict[str, Any]) -> Optional[str]:
    op = meta.get("op")
    op_label = "↑ Sube" if op == ">" else "↓ Baja" if op == "<" else None
    if kind == "fx":
        t = meta.get("type")
        side = meta.get("side")
        label = FX_TYPE_LABELS.get(t or "", (t or "").upper())
        side_label = SIDE_LABELS.get(side or "", side or "?")
        if label and side_label and op_label:
            return f"{label} {side_label} {op_label}"
    elif kind == "metric":
        t = meta.get("type")
        label = METRIC_TYPE_LABELS.get(t or "", (t or "").capitalize())
        if label and op_label:
            return f"{label} {op_label}"
    elif kind == "crypto":
        sym = (meta.get("symbol") or "").upper()
        base = (meta.get("base") or "").upper() or None
        quote = (meta.get("quote") or "").upper() or None
        if sym and op_label:
            disp = crypto_display_name(sym, base, quote)
            return f"{disp} {op_label}" if disp else f"{sym} {op_label}"
    elif kind == "ticker":
        sym = (meta.get("symbol") or "").upper()
        if sym and op_label:
            return f"{_label_long(sym)} {op_label}"
    return None


def _record_alert_usage(chat_id: int, al: Dict[str, Any]) -> None:
    kind = al.get("kind")
    if not kind:
        return
    meta: Dict[str, Any] = {}
    if kind == "fx":
        meta = {
            "type": al.get("type"),
            "side": al.get("side"),
            "op": al.get("op"),
            "mode": al.get("mode"),
        }
    elif kind == "metric":
        meta = {
            "type": al.get("type"),
            "op": al.get("op"),
            "mode": al.get("mode"),
        }
    elif kind == "crypto":
        meta = {
            "symbol": (al.get("symbol") or "").upper(),
            "base": (al.get("crypto_base") or "").upper(),
            "quote": (al.get("crypto_quote") or "").upper(),
            "op": al.get("op"),
            "mode": al.get("mode"),
        }
    elif kind == "ticker":
        meta = {
            "symbol": (al.get("symbol") or "").upper(),
            "op": al.get("op"),
            "segment": al.get("segment"),
        }
    key = _alert_usage_key(kind, meta)
    if not key:
        return
    usage = ALERT_USAGE.setdefault(chat_id, {})
    entry = usage.setdefault(key, {"count": 0, "kind": kind, "meta": meta})
    entry["count"] = int(entry.get("count", 0)) + 1
    entry["kind"] = kind
    entry["meta"] = meta
    entry["last"] = time()
    if len(usage) > 30:
        to_remove = sorted(
            usage.items(),
            key=lambda item: (
                int(item[1].get("count", 0)),
                float(item[1].get("last", 0.0)),
            ),
        )
        for drop_key, _ in to_remove[: len(usage) - 30]:
            usage.pop(drop_key, None)


def _get_alert_usage_suggestions(chat_id: int) -> List[Dict[str, Any]]:
    suggestions: List[Dict[str, Any]] = []
    for key, entry in (ALERT_USAGE.get(chat_id) or {}).items():
        count = int(entry.get("count", 0))
        if count < 1:
            continue
        kind = entry.get("kind")
        meta = entry.get("meta") or {}
        label = _alert_usage_label(kind, meta)
        if not label:
            continue
        suggestions.append(
            {
                "key": key,
                "label": label,
                "count": count,
                "last": float(entry.get("last", 0.0)),
                "kind": kind,
                "meta": meta,
            }
        )
    suggestions.sort(key=lambda x: (-x["count"], -x["last"], x["label"]))
    return suggestions[:5]


def _alertas_kind_prompt(chat_id: int) -> Tuple[str, InlineKeyboardMarkup]:
    base_rows: List[List[Tuple[str, str]]] = [
        [("Dólares", "KIND:fx"), ("Economía", "KIND:metric")],
        [("Acciones", "KIND:acciones"), ("Cedears", "KIND:cedears")],
        [("Criptomonedas", "KIND:crypto")],
        [("Volver", "AL:MENU"), ("Cancelar", "CANCEL")],
    ]
    suggestions = _get_alert_usage_suggestions(chat_id)
    rows: List[List[Tuple[str, str]]] = []
    if suggestions:
        for sug in suggestions:
            rows.append([(f"⭐ {sug['label']}", f"SUG:{sug['key']}")])
    rows.extend(base_rows)
    text = "¿Qué querés alertar?"
    if suggestions:
        extra = ["", "Sugerencias frecuentes:"]
        extra.extend([f"• {s['label']} ({s['count']}×)" for s in suggestions])
        text = "\n".join([text] + extra)
    return text, kb(rows)


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
    chat_id = update.effective_chat.id
    text, markup = _alertas_kind_prompt(chat_id)
    if update.callback_query:
        q = update.callback_query; await q.answer()
        await q.edit_message_text(text, reply_markup=markup)
    else:
        await update.effective_message.reply_text(text, reply_markup=markup)
    return AL_KIND


async def alertas_add_exit_to_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    context.user_data.pop("al", None)
    await cmd_alertas_menu(update, context, edit=True)
    return ConversationHandler.END

async def alertas_back(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    target = q.data.split(":",1)[1]
    al = context.user_data.get("al", {})
    if target == "KIND":
        text, markup = _alertas_kind_prompt(update.effective_chat.id)
        await q.edit_message_text(text, reply_markup=markup); return AL_KIND
    if target == "FXTYPE":
        await q.edit_message_text("Elegí el tipo de dólar:", reply_markup=kb_submenu_fx()); return AL_FX_TYPE
    if target == "FXSIDE":
        t = al.get("type","?"); await q.edit_message_text(f"Tipo: {t.upper()}\nElegí lado:", reply_markup=kb_fx_side_for(t)); return AL_FX_SIDE
    if target == "METRIC":
        await q.edit_message_text("Elegí la métrica:", reply_markup=kb_submenu_metric()); return AL_METRIC_TYPE
    if target == "TICKERS_ACC":
        await q.edit_message_text("Elegí el ticker (Acciones):", reply_markup=kb_tickers(ACCIONES_BA, "KIND", "TICK")); return AL_TICKER
    if target == "TICKERS_CEDEARS":
        await q.edit_message_text("Elegí el ticker (cedear):", reply_markup=kb_tickers(CEDEARS_BA, "KIND", "TICK")); return AL_TICKER
    if target in {"CRYPTO", "CRYPTO_LIST"}:
        return await alertas_show_crypto_list(update, context, query=q)
    if target == "CRYPTO_MANUAL":
        return await alertas_prompt_crypto_manual(update, context, query=q)
    if target == "OP":
        kind = al.get("kind")
        if kind == "ticker":
            kb_op = kb([[("↑ Sube", "OP:>"), ("↓ Baja", "OP:<")],[("Volver","BACK:" + ("TICKERS_ACC" if al.get("segment")=="acciones" else "TICKERS_CEDEARS")),("Cancelar","CANCEL")]])
            await q.edit_message_text("Elegí condición:", reply_markup=kb_op)
        elif kind == "fx":
            kb_op = kb([[("↑ Sube", "OP:>"), ("↓ Baja", "OP:<")],[("Volver","BACK:FXSIDE"),("Cancelar","CANCEL")]])
            await q.edit_message_text("Elegí condición:", reply_markup=kb_op)
        elif kind == "crypto":
            kb_op = kb([[("↑ Sube", "OP:>"), ("↓ Baja", "OP:<")],[("Volver","BACK:CRYPTO_LIST"),("Cancelar","CANCEL")]])
            await q.edit_message_text("Elegí condición:", reply_markup=kb_op)
        else:
            kb_op = kb([[("↑ Sube", "OP:>"), ("↓ Baja", "OP:<")],[("Volver","BACK:METRIC"),("Cancelar","CANCEL")]])
            await q.edit_message_text("Elegí condición:", reply_markup=kb_op)
        return AL_OP
    if target == "MODE":
        kb_mode = kb([[("Ingresar Importe", "MODE:absolute"),("Ingresar % vs valor actual", "MODE:percent")], [("Volver","BACK:OP"),("Cancelar","CANCEL")]])
        await q.edit_message_text("¿Cómo querés definir el umbral?", reply_markup=kb_mode); return AL_MODE
    await cmd_alertas_menu(update, context, prefix="Operación cancelada.", edit=True)
    return ConversationHandler.END

async def alertas_add_kind(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    data = q.data
    if data == "CANCEL":
        await cmd_alertas_menu(update, context, prefix="Operación cancelada.", edit=True)
        return ConversationHandler.END
    kind = data.split(":",1)[1]
    context.user_data["al"] = {}
    al = context.user_data["al"]
    if kind == "fx":
        al["kind"] = "fx"; await q.edit_message_text("Elegí el tipo de dólar:", reply_markup=kb_submenu_fx()); return AL_FX_TYPE
    if kind == "metric":
        al["kind"] = "metric"; await q.edit_message_text("Elegí la métrica:", reply_markup=kb_submenu_metric()); return AL_METRIC_TYPE
    if kind == "acciones":
        al["kind"] = "ticker"; al["segment"] = "acciones"
        await q.edit_message_text("Elegí el ticker (Acciones):", reply_markup=kb_tickers(ACCIONES_BA, "KIND", "TICK")); return AL_TICKER
    if kind == "cedears":
        al["kind"] = "ticker"; al["segment"] = "cedears"
        await q.edit_message_text("Elegí el ticker (cedear):", reply_markup=kb_tickers(CEDEARS_BA, "KIND", "TICK")); return AL_TICKER
    if kind == "crypto":
        al["kind"] = "crypto"
        return await alertas_show_crypto_list(update, context, query=q)
    await cmd_alertas_menu(update, context, prefix="Operación cancelada.", edit=True)
    return ConversationHandler.END


async def alertas_add_suggestion(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    data = q.data.split(":", 1)
    if len(data) < 2:
        text, markup = _alertas_kind_prompt(update.effective_chat.id)
        await q.edit_message_text(text, reply_markup=markup)
        return AL_KIND
    key = data[1]
    chat_id = update.effective_chat.id
    entry = (ALERT_USAGE.get(chat_id) or {}).get(key)
    if not entry:
        text, markup = _alertas_kind_prompt(chat_id)
        await q.edit_message_text("La sugerencia ya no está disponible.", reply_markup=markup)
        return AL_KIND
    kind = entry.get("kind")
    meta = entry.get("meta") or {}
    if not kind:
        text, markup = _alertas_kind_prompt(chat_id)
        await q.edit_message_text("No pude usar esa sugerencia.", reply_markup=markup)
        return AL_KIND
    al: Dict[str, Any] = {"kind": kind}
    if kind == "fx":
        al.update({
            "type": meta.get("type"),
            "side": meta.get("side"),
            "op": meta.get("op"),
            "mode": meta.get("mode"),
        })
        context.user_data["al"] = al
        type_raw = al.get("type") or ""
        label = FX_TYPE_LABELS.get(type_raw, type_raw.upper() or "?")
        side_raw = al.get("side") or ""
        side_label = SIDE_LABELS.get(side_raw, side_raw or "?")
        op_val = al.get("op")
        op_text = "↑ Sube" if op_val == ">" else "↓ Baja" if op_val == "<" else "?"
        kb_mode = kb([
            [("Ingresar Importe", "MODE:absolute"), ("Ingresar % vs valor actual", "MODE:percent")],
            [("Volver", "BACK:OP"), ("Cancelar", "CANCEL")],
        ])
        await q.edit_message_text(
            f"Tipo: {label} | Lado: {side_label} | Condición: {op_text}\n¿Cómo querés definir el umbral?",
            reply_markup=kb_mode,
        )
        return AL_MODE
    if kind == "metric":
        al.update({
            "type": meta.get("type"),
            "op": meta.get("op"),
            "mode": meta.get("mode"),
        })
        context.user_data["al"] = al
        type_raw = al.get("type") or ""
        label = METRIC_TYPE_LABELS.get(type_raw, type_raw.capitalize() or "?")
        op_val = al.get("op")
        op_text = "↑ Sube" if op_val == ">" else "↓ Baja" if op_val == "<" else "?"
        kb_mode = kb([
            [("Ingresar Importe", "MODE:absolute"), ("Ingresar % vs valor actual", "MODE:percent")],
            [("Volver", "BACK:OP"), ("Cancelar", "CANCEL")],
        ])
        await q.edit_message_text(
            f"Métrica: {label} | Condición: {op_text}\n¿Cómo querés definir el umbral?",
            reply_markup=kb_mode,
        )
        return AL_MODE
    if kind == "crypto":
        al.update({
            "symbol": (meta.get("symbol") or "").upper(),
            "crypto_base": (meta.get("base") or "").upper(),
            "crypto_quote": (meta.get("quote") or "").upper(),
            "op": meta.get("op"),
            "mode": meta.get("mode"),
        })
        context.user_data["al"] = al
        label = crypto_display_name(al.get("symbol"), al.get("crypto_base"), al.get("crypto_quote"))
        op_val = al.get("op")
        op_text = "↑ Sube" if op_val == ">" else "↓ Baja" if op_val == "<" else "?"
        kb_mode = kb([
            [("Ingresar Importe", "MODE:absolute"), ("Ingresar % vs valor actual", "MODE:percent")],
            [("Volver", "BACK:OP"), ("Cancelar", "CANCEL")],
        ])
        await q.edit_message_text(
            f"Cripto: {label} | Condición: {op_text}\n¿Cómo querés definir el umbral?",
            reply_markup=kb_mode,
        )
        return AL_MODE
    if kind == "ticker":
        al.update({
            "symbol": (meta.get("symbol") or "").upper(),
            "op": meta.get("op"),
            "segment": meta.get("segment"),
        })
        context.user_data["al"] = al
        async with ClientSession() as session:
            metmap, _ = await metrics_for_symbols(session, [al.get("symbol")])
        sym = al.get("symbol")
        last_px = metmap.get(sym, {}).get("last_px") if metmap else None
        price_s = fmt_money_ars(last_px) if last_px is not None else "—"
        op_val = al.get("op")
        op_text = "↑ Sube" if op_val == ">" else "↓ Baja" if op_val == "<" else "?"
        msg = (
            f"Ticker: {_label_long(sym)} | Condición: {op_text}\n"
            f"Actual: Precio {price_s}\n\n"
            "Ingresá el <b>precio objetivo</b> (solo número, sin símbolos ni separadores). Ej: 3500\n"
            "<i>Válidos: 100 | 1000.5 · Inválidos: $100, 1.000,50, 100%</i>"
        )
        await q.edit_message_text(msg, parse_mode=ParseMode.HTML)
        return AL_VALUE
    text, markup = _alertas_kind_prompt(chat_id)
    await q.edit_message_text("No pude usar esa sugerencia.", reply_markup=markup)
    return AL_KIND

async def alertas_add_fx_type(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    if q.data == "CANCEL":
        await cmd_alertas_menu(update, context, prefix="Operación cancelada.", edit=True)
        return ConversationHandler.END
    if q.data.startswith("BACK:"): return await alertas_back(update, context)
    t = q.data.split(":",1)[1]
    context.user_data["al"]["type"] = t
    await q.edit_message_text(f"Tipo: {t.upper()}\nElegí lado:", reply_markup=kb_fx_side_for(t))
    return AL_FX_SIDE

async def alertas_add_fx_side(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    if q.data == "CANCEL":
        await cmd_alertas_menu(update, context, prefix="Operación cancelada.", edit=True)
        return ConversationHandler.END
    if q.data.startswith("BACK:"): return await alertas_back(update, context)
    side = q.data.split(":",1)[1]
    context.user_data["al"]["side"] = side
    kb_op = kb([[("↑ Sube", "OP:>"), ("↓ Baja", "OP:<")],[("Volver","BACK:FXSIDE"),("Cancelar","CANCEL")]])
    await q.edit_message_text(f"Lado: {side}\nElegí condición:", reply_markup=kb_op)
    return AL_OP

async def alertas_add_metric_type(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    if q.data == "CANCEL":
        await cmd_alertas_menu(update, context, prefix="Operación cancelada.", edit=True)
        return ConversationHandler.END
    if q.data.startswith("BACK:"): return await alertas_back(update, context)
    m = q.data.split(":",1)[1]
    context.user_data["al"]["type"] = m
    kb_op = kb([[("↑ Sube", "OP:>"), ("↓ Baja", "OP:<")],[("Volver","BACK:METRIC"),("Cancelar","CANCEL")]])
    await q.edit_message_text(f"Métrica: {m.upper()}\nElegí condición:", reply_markup=kb_op)
    return AL_OP

async def alertas_add_ticker_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    if q.data == "CANCEL":
        await cmd_alertas_menu(update, context, prefix="Operación cancelada.", edit=True)
        return ConversationHandler.END
    if q.data.startswith("BACK:"): return await alertas_back(update, context)
    sym = q.data.split(":",1)[1].upper()
    context.user_data["al"]["symbol"] = sym
    k = kb([[("↑ Sube", "OP:>"), ("↓ Baja", "OP:<")],
            [("Volver","BACK:" + ("TICKERS_ACC" if context.user_data["al"].get("segment")=="acciones" else "TICKERS_CEDEARS")),("Cancelar","CANCEL")]])
    await q.edit_message_text(f"Ticker: {_label_long(sym)}\nElegí condición:", reply_markup=k)
    return AL_OP

async def alertas_crypto_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
    if not text:
        await update.message.reply_text("Ingresá el símbolo de la cripto. Ej: BTCUSDT o BTC/USDT.")
        return AL_CRYPTO
    normalized = unicodedata.normalize("NFKD", text)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    upper = normalized.upper()
    alphanum = re.sub(r"[^A-Z0-9]", "", upper)
    parts = [p for p in re.split(r"[^A-Z0-9]+", upper) if p]
    async with ClientSession() as session:
        symbols_map = await get_binance_symbols(session)
    if not symbols_map:
        await update.message.reply_text("No pude obtener el listado de Binance. Probá más tarde.")
        return AL_CRYPTO
    matches: List[Dict[str, str]] = []
    seen: Set[str] = set()

    def add_match(info: Dict[str, str]):
        sym = info.get("symbol")
        if sym and sym not in seen:
            seen.add(sym)
            matches.append(info)

    if alphanum and alphanum in symbols_map:
        add_match(symbols_map[alphanum])
    if not matches and len(parts) >= 2:
        combined = (parts[0] + parts[1]).upper()
        if combined in symbols_map:
            add_match(symbols_map[combined])
    if not matches and alphanum:
        for info in symbols_map.values():
            if alphanum == info.get("base"):
                add_match(info)
    if not matches and alphanum:
        for info in symbols_map.values():
            if info.get("symbol", "").startswith(alphanum):
                add_match(info)
                if len(matches) >= 12:
                    break
    if not matches and alphanum:
        for info in symbols_map.values():
            if info.get("base", "").startswith(alphanum):
                add_match(info)
                if len(matches) >= 12:
                    break
    if not matches and alphanum:
        for info in symbols_map.values():
            if alphanum in info.get("symbol", ""):
                add_match(info)
                if len(matches) >= 12:
                    break
    matches = _sort_crypto_matches(matches)[:12]
    al = context.user_data.setdefault("al", {})
    if not matches:
        await update.message.reply_text("No encontré esa cripto. Probá con el símbolo completo, por ejemplo BTCUSDT.")
        return AL_CRYPTO
    if len(matches) == 1:
        info = matches[0]
        al["symbol"] = info.get("symbol")
        al["crypto_base"] = info.get("base")
        al["crypto_quote"] = info.get("quote")
        label = crypto_display_name(info.get("symbol"), info.get("base"), info.get("quote"))
        k = kb([[("↑ Sube", "OP:>"), ("↓ Baja", "OP:<")], [("Volver", "BACK:CRYPTO_MANUAL"), ("Cancelar", "CANCEL")]])
        await update.message.reply_text(f"Cripto: {label}\nElegí condición:", reply_markup=k)
        return AL_OP
    buttons: List[List[Tuple[str, str]]] = []
    row: List[Tuple[str, str]] = []
    for info in matches:
        sym = info.get("symbol")
        label = crypto_display_name(sym, info.get("base"), info.get("quote"))
        row.append((label, f"CRYPTOSEL:{sym}"))
        if len(row) == 2:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)
    buttons.append([("Nueva búsqueda", "BACK:CRYPTO_MANUAL"), ("Cancelar", "CANCEL")])
    await update.message.reply_text("Elegí la cripto:", reply_markup=kb(buttons))
    return AL_CRYPTO

async def alertas_crypto_pick_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    data = q.data
    if data == "CANCEL":
        await cmd_alertas_menu(update, context, prefix="Operación cancelada.", edit=True)
        return ConversationHandler.END
    if data == "CRYPTO:SEARCH":
        return await alertas_prompt_crypto_manual(update, context, query=q)
    if data.startswith("BACK:"):
        return await alertas_back(update, context)
    sym = data.split(":", 1)[1].upper()
    async with ClientSession() as session:
        symbols_map = await get_binance_symbols(session)
    info = symbols_map.get(sym)
    if not info:
        await q.edit_message_text("No encontré esa cripto. Probá buscar de nuevo.")
        return AL_CRYPTO
    al = context.user_data.setdefault("al", {})
    al["symbol"] = info.get("symbol")
    al["crypto_base"] = info.get("base")
    al["crypto_quote"] = info.get("quote")
    label = crypto_display_name(info.get("symbol"), info.get("base"), info.get("quote"))
    k = kb([[("↑ Sube", "OP:>"), ("↓ Baja", "OP:<")], [("Volver", "BACK:CRYPTO_LIST"), ("Cancelar", "CANCEL")]])
    await q.edit_message_text(f"Cripto: {label}\nElegí condición:", reply_markup=k)
    return AL_OP

async def alertas_add_op(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    if q.data == "CANCEL":
        await cmd_alertas_menu(update, context, prefix="Operación cancelada.", edit=True)
        return ConversationHandler.END
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
    if q.data == "CANCEL":
        await cmd_alertas_menu(update, context, prefix="Operación cancelada.", edit=True)
        return ConversationHandler.END
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
        if al.get("kind") == "crypto":
            sym = al.get("symbol")
            quote = al.get("crypto_quote")
            base = al.get("crypto_base")
            price = await get_crypto_price(session, sym, base=base, quote=quote) if sym else None
            if price is None:
                await q.edit_message_text("No pude leer el precio actual. Probá más tarde.")
                return ConversationHandler.END
            label = crypto_display_name(sym, base, quote)
            price_s = fmt_crypto_price(price, quote)
            if mode == "percent":
                msg = (f"Cripto: {label} | Condición: {op_text}\nAhora: {price_s}\n\nIngresá el <b>%</b> (solo número). Ej: 10 | 7.5")
            else:
                unidad = (quote or "").upper() or "USDT"
                msg = (f"Cripto: {label} | Condición: {op_text}\nAhora: {price_s}\n\nIngresá el <b>precio objetivo</b> en {unidad} (solo número).")
            await q.edit_message_text(msg, parse_mode=ParseMode.HTML); return AL_VALUE
    await cmd_alertas_menu(update, context, prefix="Operación cancelada.", edit=True)
    return ConversationHandler.END

async def alertas_add_value(update: Update, context: ContextTypes.DEFAULT_TYPE):
    al = context.user_data.get("al", {})
    val = _parse_float_user_strict(update.message.text)
    if val is None:
        await update.message.reply_text("Ingresá solo número (sin $ ni % ni separadores)."); return AL_VALUE
    chat_id = update.effective_chat.id
    rules = ALERTS.setdefault(chat_id, [])
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
            candidate = {"kind":"fx","type":al["type"],"side":al["side"],"op":al["op"],"value":float(thr)}
            if _has_duplicate_alert(rules, candidate):
                await update.message.reply_text("Ya tenés una alerta igual configurada. Probá con otro valor.")
                return AL_VALUE
            rules.append(candidate)
            _record_alert_usage(chat_id, al)
            save_state()
            await update.message.reply_text("Listo. Alerta agregada ✅")
            await cmd_alertas_menu(update, context)
            return ConversationHandler.END

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
            candidate = {"kind":"metric","type":al["type"],"op":al["op"],"value":float(thr)}
            if _has_duplicate_alert(rules, candidate):
                await update.message.reply_text("Ya tenés una alerta igual configurada. Probá con otro valor.")
                return AL_VALUE
            rules.append(candidate)
            _record_alert_usage(chat_id, al)
            save_state()
            await update.message.reply_text("Listo. Alerta agregada ✅")
            await cmd_alertas_menu(update, context)
            return ConversationHandler.END

        if al.get("kind") == "crypto":
            sym = al.get("symbol")
            op = al.get("op")
            quote = al.get("crypto_quote")
            base = al.get("crypto_base")
            price = await get_crypto_price(session, sym, base=base, quote=quote) if sym else None
            if price is None:
                await update.message.reply_text("No pude leer el precio actual."); return ConversationHandler.END
            thr = price*(1 + (val/100.0)) if al.get("mode") == "percent" and op == ">" else \
                  price*(1 - (val/100.0)) if al.get("mode") == "percent" else val
            if al.get("mode") == "absolute":
                if (op == ">" and thr <= price) or (op == "<" and thr >= price):
                    await update.message.reply_text(
                        f"El precio objetivo debe ser {'mayor' if op=='>' else 'menor'} que {fmt_crypto_price(price, quote)}."
                    ); return AL_VALUE
            candidate = {
                "kind": "crypto",
                "symbol": (sym or "").upper(),
                "op": op,
                "value": float(thr),
                "mode": al.get("mode"),
                "base": (base or "").upper(),
                "quote": (quote or "").upper(),
            }
            if _has_duplicate_alert(rules, candidate):
                await update.message.reply_text("Ya tenés una alerta igual configurada. Probá con otro valor.")
                return AL_VALUE
            rules.append(candidate)
            _record_alert_usage(chat_id, al)
            save_state()
            target_s = fmt_crypto_price(thr, quote)
            direction = "sube a" if op == ">" else "baja a"
            await update.message.reply_text(
                f"Listo. Alerta agregada ✅\nSe disparará si el precio {direction} {target_s}."
            )
            await cmd_alertas_menu(update, context)
            return ConversationHandler.END

        # ticker
        sym, op = al.get("symbol"), al.get("op")
        metmap, _ = await metrics_for_symbols(session, [sym])
        last_px = metmap.get(sym, {}).get("last_px")
        if last_px is None:
            await update.message.reply_text("No pude leer el precio actual."); return ConversationHandler.END
        thr = val
        if (op == ">" and thr <= last_px) or (op == "<" and thr >= last_px):
            await update.message.reply_text(f"El precio objetivo debe ser {'mayor' if op=='>' else 'menor'} que {fmt_money_ars(last_px)}."); return AL_VALUE
        candidate = {"kind":"ticker","symbol":sym,"op":op,"value":float(thr),"mode":"absolute"}
        if _has_duplicate_alert(rules, candidate):
            await update.message.reply_text("Ya tenés una alerta igual configurada. Probá con otro valor.")
            return AL_VALUE
        rules.append(candidate)
        _record_alert_usage(chat_id, al)
        save_state()
        await update.message.reply_text("Listo. Alerta agregada ✅")
        await cmd_alertas_menu(update, context)
        return ConversationHandler.END

# ============================ LOOP ALERTAS ============================

async def alerts_loop(app: Application):
    try:
        await asyncio.sleep(5)
        timeout = ClientTimeout(total=12)
        while True:
            try:
                now_ts = datetime.now(TZ).timestamp()
                state_dirty = False
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
                        crypto_list = {(r.get("symbol") or "").upper() for cid in active_chats for r in ALERTS.get(cid, []) if r.get("kind")=="crypto" and r.get("symbol")}
                        crypto_info_map: Dict[str, Dict[str, Optional[str]]] = {}
                        for cid in active_chats:
                            for rule in ALERTS.get(cid, []) or []:
                                if rule.get("kind") != "crypto":
                                    continue
                                sym = (rule.get("symbol") or "").upper()
                                if not sym or sym in crypto_info_map:
                                    continue
                                crypto_info_map[sym] = {
                                    "symbol": sym,
                                    "base": rule.get("base"),
                                    "quote": rule.get("quote"),
                                }
                        metmap, _ = (await metrics_for_symbols(session, sorted(sym_list))) if sym_list else ({}, None)
                        crypto_prices = await get_crypto_prices(session, sorted(crypto_list), crypto_info_map) if crypto_list else {}
                        for chat_id in active_chats:
                            rules = ALERTS.get(chat_id, [])
                            if not rules: continue
                            trig = []
                            for r in rules:
                                paused, changed = _is_rule_paused(r, now_ts)
                                if changed:
                                    state_dirty = True
                                if paused:
                                    continue
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
                                    sym = r["symbol"]; m = metmap.get(sym, {}); cur = m.get("last_px")
                                    if cur is None: continue
                                    ok = (cur > r["value"]) if r["op"] == ">" else (cur < r["value"])
                                    if ok: trig.append(("ticker_px", sym, r["op"], r["value"], cur))
                                elif r.get("kind") == "crypto":
                                    sym = (r.get("symbol") or "").upper()
                                    cur = crypto_prices.get(sym)
                                    if cur is None: continue
                                    ok = (cur > r["value"]) if r["op"] == ">" else (cur < r["value"])
                                    if ok:
                                        trig.append(("crypto_px", sym, r.get("op"), r.get("value"), cur, r.get("base"), r.get("quote")))
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
                                    elif t == "crypto_px":
                                        sym, op, v, cur, base, quote = rest
                                        label = crypto_display_name(sym, base, quote)
                                        lines.append(f"{label}: {fmt_crypto_price(cur, quote)} ({html_op(op)} {fmt_crypto_price(v, quote)})")
                                    else:
                                        sym, op, v, cur = rest
                                        lines.append(f"{_label_long(sym)} (Precio): {fmt_money_ars(cur)} ({html_op(op)} {fmt_money_ars(v)})")
                                try:
                                    await app.bot.send_message(chat_id, "\n".join(lines), parse_mode=ParseMode.HTML)
                                except Exception as e:
                                    log.warning("send alert failed %s: %s", chat_id, e)
                    if state_dirty:
                        save_state()
                await asyncio.sleep(600)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning("alerts_loop error: %s", e)
                await asyncio.sleep(30)
    except asyncio.CancelledError:
        log.info("alerts_loop cancelado")
        raise

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
    await update.effective_message.reply_text(txt, reply_markup=kb_times_full(), parse_mode=ParseMode.HTML)
    return SUBS_SET_TIME


async def subs_start_from_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    return await cmd_subs(update, context)

async def subs_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    chat_id = q.message.chat_id; data = q.data
    if data == "SUBS:CLOSE": await q.edit_message_text("Listo."); return ConversationHandler.END
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

# ============================ PORTAFOLIO (salida debajo del menú + torta) ============================

def pf_get(chat_id: int) -> Dict[str, Any]:
    pf = PF.setdefault(chat_id, {"base": {}, "monto": 0.0, "items": []})
    base_conf = pf.get("base")
    if not isinstance(base_conf, dict):
        base_conf = {}
        pf["base"] = base_conf
    base_conf.setdefault("moneda", "ARS")
    base_conf.setdefault("tc", "mep")
    base_conf.setdefault("tc_valor", None)
    base_conf.setdefault("tc_timestamp", None)
    if not isinstance(pf.get("items"), list):
        pf["items"] = []
    return pf

def kb_pf_main() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Ayuda", callback_data="PF:HELP")],
        [InlineKeyboardButton("Fijar base", callback_data="PF:SETBASE"), InlineKeyboardButton("Fijar monto", callback_data="PF:SETMONTO")],
        [InlineKeyboardButton("Agregar instrumento", callback_data="PF:ADD")],
        [InlineKeyboardButton("Ver composición", callback_data="PF:LIST"), InlineKeyboardButton("Editar instrumento", callback_data="PF:EDIT")],
        [InlineKeyboardButton("Rendimiento", callback_data="PF:RET"), InlineKeyboardButton("Proyección", callback_data="PF:PROJ")],
        [InlineKeyboardButton("Exportar", callback_data="PF:EXPORT")],
        [InlineKeyboardButton("Eliminar portafolio", callback_data="PF:CLEAR")],
    ])

def kb_pf_add_methods() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Por cantidad", callback_data="PF:ADDQTY"), InlineKeyboardButton("Por importe", callback_data="PF:ADDAMT")],
        [InlineKeyboardButton("Por % del monto", callback_data="PF:ADDPCT")],
        [InlineKeyboardButton("Volver", callback_data="PF:ADD")],
    ])

async def pf_main_menu_text(chat_id: int) -> str:
    pf = pf_get(chat_id)
    base_conf = pf.get("base", {})
    base = (base_conf.get("moneda") or "ARS").upper()
    tc = (base_conf.get("tc") or "oficial").upper()
    monto = float(pf.get("monto") or 0.0)
    f_money = fmt_money_ars if base == "ARS" else fmt_money_usd
    _, _, total_invertido, total_actual, tc_val, tc_ts = await pf_market_snapshot(pf)
    restante = max(0.0, monto - total_invertido)
    lines = ["<b>📦 Menú Portafolio</b>"]
    lines.append(f"Base: {base} / {tc}")
    lines.append(f"Monto objetivo: {f_money(monto)}")
    lines.append(f"Valor invertido: {f_money(total_invertido)}")
    lines.append(f"Restante: {f_money(restante)}")
    if pf.get("items"):
        lines.append(f"Valor actual estimado: {f_money(total_actual)}")
    lines.append(f"Instrumentos cargados: {len(pf.get('items', []))}")
    if tc_val is not None:
        tc_line = f"Tipo de cambio ref. ({tc}): {fmt_money_ars(tc_val)} por USD"
        if tc_ts:
            dt = datetime.fromtimestamp(tc_ts, TZ)
            tc_line += f" (al {dt.strftime('%d/%m/%Y %H:%M')})"
        lines.append(tc_line)
    return "\n".join(lines)

async def pf_refresh_menu(context: ContextTypes.DEFAULT_TYPE, chat_id: int):
    text = await pf_main_menu_text(chat_id)
    kb_main = kb_pf_main()
    msg_id = context.user_data.get("pf_menu_msg_id") if isinstance(context.user_data, dict) else None
    if msg_id:
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=msg_id,
                text=text,
                parse_mode=ParseMode.HTML,
                reply_markup=kb_main,
            )
        except Exception:
            pass
        else:
            context.user_data["pf_menu_msg_id"] = msg_id
            return
    sent = await context.bot.send_message(
        chat_id,
        text,
        parse_mode=ParseMode.HTML,
        reply_markup=kb_main,
        disable_web_page_preview=True,
    )
    context.user_data["pf_menu_msg_id"] = sent.message_id

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
    chat_id = update.effective_chat.id
    text = await pf_main_menu_text(chat_id)
    msg = await update.effective_message.reply_text(
        text,
        reply_markup=kb_pf_main(),
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=True,
    )
    context.user_data["pf_menu_msg_id"] = msg.message_id

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
    context.user_data["pf_menu_msg_id"] = q.message.message_id

    if data == "PF:MENU":
        await cmd_portafolio(update, context)
        return

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
        base_conf = pf.get("base", {})
        tc_val = None
        tc_ts: Optional[int] = None
        if tc:
            async with ClientSession() as session:
                tc_val = await get_tc_value(session, tc)
            tc_ts = int(time()) if tc_val is not None else None
        base_conf.update({
            "moneda": mon,
            "tc": tc,
            "tc_valor": tc_val,
            "tc_timestamp": tc_ts,
        })
        pf["base"] = base_conf
        save_state()
        msg = f"Base fijada: {mon.upper()} / {tc.upper()}"
        await pf_refresh_menu(context, chat_id)
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
            [InlineKeyboardButton("Buscar ticker", callback_data="PF:SEARCH")],
            [InlineKeyboardButton("Volver", callback_data="PF:BACK")]
        ])
        if q.message and (q.message.text or "").startswith("📦 Menú Portafolio"):
            await _send_below_menu(context, chat_id, text="¿Qué querés agregar?", reply_markup=kb_add)
        else:
            await q.edit_message_text("¿Qué querés agregar?", reply_markup=kb_add)
        return

    if data == "PF:SEARCH":
        context.user_data["pf_mode"] = "pf_search_symbol"
        context.user_data["pf_add_message_id"] = q.message.message_id
        await _send_below_menu(context, chat_id, text="Ingresá el <b>ticker o nombre</b> del instrumento.")
        return

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
        context.user_data["pf_add_message_id"] = q.message.message_id
        return

    if data.startswith("PF:PICK:"):
        sym = data.split(":")[2]
        if sym in CRIPTO_TOP_NAMES:
            context.user_data["pf_add_simbolo"] = _crypto_to_symbol(sym)
            sel_label = _label_long(sym)
        else:
            context.user_data["pf_add_simbolo"] = sym
            sel_label = _label_long(sym)
        kb_ask = kb_pf_add_methods()
        await _send_below_menu(context, chat_id, text=f"Seleccionado: {sel_label}\n¿Cómo cargar?")
        await q.edit_message_reply_markup(reply_markup=kb_ask)
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
            pf["items"].pop(idx); save_state()
            await _send_below_menu(context, chat_id, text="Instrumento eliminado.")
            await pf_refresh_menu(context, chat_id)
            return
        await _send_below_menu(context, chat_id, text="Índice inválido."); return

    if data == "PF:RET":
        await pf_show_return_below(context, chat_id)
        return
    if data == "PF:PROJ":
        await pf_show_projection_below(context, chat_id)
        return

    if data == "PF:EXPORT":
        pf = pf_get(chat_id)
        if not pf.get("items"):
            await _send_below_menu(context, chat_id, text="Tu portafolio está vacío. No hay datos para exportar.")
            return
        snapshot, last_ts, total_invertido, total_actual, tc_val, tc_ts = await pf_market_snapshot(pf)
        base_conf = pf.get("base", {})
        base = (base_conf.get("moneda") or "ARS").upper()
        tc_name = (base_conf.get("tc") or "oficial").upper()
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow([
            "symbol",
            "nombre",
            "tipo",
            "cantidad",
            "precio_base",
            "importe_base",
            "valor_actual_estimado",
            "moneda_base",
            "tc_nombre",
            "tc_valor",
            "tc_timestamp",
            "fecha_valuacion",
            "item_fx_rate",
            "item_fx_timestamp",
            "fecha_alta",
            "added_timestamp",
        ])
        fecha_val = datetime.fromtimestamp(last_ts, TZ).strftime("%Y-%m-%d") if last_ts else ""
        tc_fecha = datetime.fromtimestamp(tc_ts, TZ).strftime("%Y-%m-%d %H:%M") if tc_ts else ""
        for entry in snapshot:
            sym = entry.get("symbol") or ""
            qty = entry.get("cantidad")
            item_fx_rate = entry.get("fx_rate")
            item_fx_ts = entry.get("fx_ts")
            item_fx_fecha = datetime.fromtimestamp(item_fx_ts, TZ).strftime("%Y-%m-%d %H:%M") if item_fx_ts else ""
            added_ts_raw = entry.get("raw", {}).get("added_ts") if entry.get("raw") else entry.get("added_ts")
            try:
                added_ts = int(added_ts_raw) if added_ts_raw is not None else None
            except (TypeError, ValueError):
                added_ts = None
            added_fecha = datetime.fromtimestamp(added_ts, TZ).strftime("%Y-%m-%d %H:%M") if added_ts else ""
            writer.writerow([
                sym,
                entry.get("label") or sym or entry.get("tipo") or "",
                entry.get("tipo") or "",
                qty if qty is not None else "",
                entry.get("precio_base") if entry.get("precio_base") is not None else "",
                entry.get("invertido"),
                entry.get("valor_actual"),
                base,
                tc_name,
                tc_val if tc_val is not None else "",
                tc_fecha,
                fecha_val,
                item_fx_rate if item_fx_rate is not None else "",
                item_fx_fecha,
                added_fecha,
                added_ts if added_ts is not None else "",
            ])
        csv_bytes = buf.getvalue().encode("utf-8")
        filename = f"portafolio_{datetime.now(TZ).strftime('%Y%m%d_%H%M')}.csv"
        f_money = fmt_money_ars if base == "ARS" else fmt_money_usd
        caption_lines = [f"Base {base}/{tc_name}"]
        caption_lines.append(f"Valor invertido: {f_money(total_invertido)}")
        caption_lines.append(f"Valor actual estimado: {f_money(total_actual)}")
        caption_lines.append(f"Instrumentos: {len(snapshot)}")
        if tc_val is not None:
            tc_caption = f"TC ref.: {fmt_money_ars(tc_val)} por USD"
            if tc_ts:
                tc_caption += f" (al {datetime.fromtimestamp(tc_ts, TZ).strftime('%d/%m/%Y %H:%M')})"
            caption_lines.append(tc_caption)
        if fecha_val:
            caption_lines.append(f"Datos al {fecha_val}")
        await context.bot.send_document(
            chat_id=chat_id,
            document=io.BytesIO(csv_bytes),
            filename=filename,
            caption="\n".join(caption_lines),
        )
        return

    if data == "PF:CLEAR":
        pf = pf_get(chat_id)
        items = pf.get("items", [])

        def _pf_clear_keyboard() -> InlineKeyboardMarkup:
            rows = []
            for i, it in enumerate(items, 1):
                sym = it.get("simbolo")
                label = _label_long(sym) if sym else it.get("tipo", "Instrumento").upper()
                rows.append([InlineKeyboardButton(f"{i}. {label}", callback_data=f"PF:CLEAR:{i-1}")])
            rows.append([
                InlineKeyboardButton("Eliminar todo", callback_data="PF:CLEAR:ALL"),
                InlineKeyboardButton("Deshacer", callback_data="PF:CLEAR:UNDO"),
            ])
            rows.append([InlineKeyboardButton("Cancelar", callback_data="PF:CLEAR:CANCEL")])
            return InlineKeyboardMarkup(rows)

        if not items:
            await _send_below_menu(
                context,
                chat_id,
                text="Tu portafolio está vacío. Podés usar <b>Deshacer</b> si eliminaste algo recientemente.",
                reply_markup=_pf_clear_keyboard(),
            )
        else:
            await _send_below_menu(
                context,
                chat_id,
                text="Elegí qué instrumento eliminar:",
                reply_markup=_pf_clear_keyboard(),
            )
        return

    if data.startswith("PF:CLEAR:"):
        pf = pf_get(chat_id)
        stack: List[Dict[str, Any]] = context.user_data.setdefault("pf_deleted_stack", [])
        action = data.split(":", 2)[2]

        try:
            await q.edit_message_reply_markup(reply_markup=None)
        except Exception:
            pass

        if action == "ALL":
            items = pf.get("items", [])
            if not items:
                await _send_below_menu(context, chat_id, text="No hay instrumentos para eliminar.")
                return
            kb_confirm = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("Sí, eliminar todo", callback_data="PF:CLEAR:ALLCONFIRM"),
                    InlineKeyboardButton("Cancelar", callback_data="PF:CLEAR:CANCEL"),
                ]
            ])
            await _send_below_menu(
                context,
                chat_id,
                text="¿Seguro que querés eliminar todos los instrumentos? Podés deshacer luego.",
                reply_markup=kb_confirm,
            )
            return

        if action == "ALLCONFIRM":
            items = pf.get("items", [])
            if not items:
                await _send_below_menu(context, chat_id, text="No había instrumentos para eliminar.")
                return
            for idx, entry in reversed(list(enumerate(items))):
                stack.append({"index": idx, "item": copy.deepcopy(entry)})
            cnt = len(items)
            pf["items"].clear()
            save_state()
            await pf_refresh_menu(context, chat_id)
            await _send_below_menu(
                context,
                chat_id,
                text=f"Se eliminaron {cnt} instrumentos. Podés usar <b>Deshacer</b> para recuperarlos.",
            )
            return

        if action == "UNDO":
            if not stack:
                await _send_below_menu(context, chat_id, text="No hay eliminaciones para deshacer.")
                return
            last = stack.pop()
            idx = max(0, min(int(last.get("index", 0)), len(pf.get("items", []))))
            item = copy.deepcopy(last.get("item"))
            pf.setdefault("items", [])
            pf["items"].insert(idx, item)
            save_state()
            await pf_refresh_menu(context, chat_id)
            sym = item.get("simbolo") if isinstance(item, dict) else None
            label = _label_long(sym) if sym else (item.get("tipo", "Instrumento").upper() if isinstance(item, dict) else "Instrumento")
            await _send_below_menu(context, chat_id, text=f"Se restauró: {label}.")
            return

        if action == "CANCEL":
            await _send_below_menu(context, chat_id, text="Operación cancelada.")
            return

        try:
            idx = int(action)
        except Exception:
            await _send_below_menu(context, chat_id, text="Acción inválida.")
            return

        items = pf.get("items", [])
        if 0 <= idx < len(items):
            removed = items.pop(idx)
            stack.append({"index": idx, "item": copy.deepcopy(removed)})
            save_state()
            await pf_refresh_menu(context, chat_id)
            sym = removed.get("simbolo") if isinstance(removed, dict) else None
            label = _label_long(sym) if sym else (removed.get("tipo", "Instrumento").upper() if isinstance(removed, dict) else "Instrumento")
            await _send_below_menu(
                context,
                chat_id,
                text=f"Instrumento eliminado: {label}. Podés usar <b>Deshacer</b> para revertir.",
            )
        else:
            await _send_below_menu(context, chat_id, text="Índice inválido.")
        return

    if data == "PF:BACK":
        try:
            await q.delete_message()
        except Exception:
            await q.edit_message_reply_markup(reply_markup=None)
        return

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

    if mode == "pf_search_symbol":
        guess = pf_guess_symbol(text)
        if not guess:
            await update.message.reply_text("No encontré el instrumento. Probá con el ticker completo (ej. GGAL.BA).")
            return
        sym, tipo = guess
        context.user_data["pf_add_simbolo"] = sym
        context.user_data["pf_add_tipo"] = tipo
        label_sym = sym
        if sym.endswith("-USD"):
            base_sym = sym[:-4]
            if base_sym in CRIPTO_TOP_NAMES:
                label_sym = base_sym
        sel_label = _label_long(label_sym)
        kb_ask = kb_pf_add_methods()
        await _send_below_menu(context, chat_id, text=f"Seleccionado: {sel_label}\n¿Cómo cargar?")
        msg_id = context.user_data.get("pf_add_message_id")
        if msg_id:
            try:
                await context.bot.edit_message_reply_markup(chat_id=chat_id, message_id=msg_id, reply_markup=kb_ask)
            except Exception:
                pass
        context.user_data["pf_mode"] = None
        return

    def _restante_str(usado: float) -> str:
        pf_base = pf["base"]["moneda"].upper()
        return (fmt_money_ars if pf_base=="ARS" else fmt_money_usd)(max(0.0, pf["monto"] - usado))

    # Monto total
    if mode == "set_monto":
        v = _parse_num_text(text)
        if v is None:
            await update.message.reply_text("Ingresá solo número (sin símbolos)."); return
        pf["monto"] = float(v); save_state()
        usado = await _pf_total_usado(chat_id)
        pf_base = pf["base"]["moneda"].upper()
        f_money = fmt_money_ars if pf_base=="ARS" else fmt_money_usd
        await update.message.reply_text(f"Monto fijado: {f_money(v)} · Restante: {_restante_str(usado)}")
        await pf_refresh_menu(context, chat_id)
        context.user_data["pf_mode"]=None; return

    # Alta por cantidad/importe/% (símbolo ya elegido)
    if mode in ("pf_add_qty","pf_add_amt","pf_add_pct"):
        v = _parse_num_text(text)
        if v is None:
            await update.message.reply_text("Ingresá solo número (sin símbolos)."); return

        tipo = context.user_data.get("pf_add_tipo")
        sym = context.user_data.get("pf_add_simbolo","")
        yfsym = sym

        base_conf = pf.get("base", {})
        pf_base = (base_conf.get("moneda") or "ARS").upper()  # ARS o USD
        inst_moneda = instrument_currency(yfsym, tipo) if yfsym else pf_base
        needs_fx = pf_base != inst_moneda
        tc_key = (base_conf.get("tc") or "oficial").lower()
        tc_val = base_conf.get("tc_valor")
        tc_ts = base_conf.get("tc_timestamp")

        price_native = None  # precio en MONEDA NATIVA
        async with ClientSession() as session:
            if yfsym.endswith(".BA") or yfsym.endswith("-USD"):
                mets, _ = await metrics_for_symbols(session, [yfsym])
                price_native = mets.get(yfsym,{}).get("last_px")
            if needs_fx and (not tc_val or tc_val <= 0):
                tc_val = await get_tc_value(session, tc_key)
                tc_ts = int(time()) if tc_val is not None else None
                base_conf["tc_valor"] = tc_val
                base_conf["tc_timestamp"] = tc_ts

        # Precio expresado en MONEDA BASE
        price_base = None
        fx_rate_used: Optional[float] = None
        fx_ts_used: Optional[int] = tc_ts if isinstance(tc_ts, int) else None
        if price_native is not None:
            if pf_base == inst_moneda:
                price_base = float(price_native)
            else:
                if tc_val and tc_val > 0:
                    fx_rate_used = float(tc_val)
                    if pf_base == "ARS" and inst_moneda == "USD":
                        price_base = float(price_native) * float(tc_val)
                    elif pf_base == "USD" and inst_moneda == "ARS":
                        price_base = float(price_native) / float(tc_val)
                else:
                    fx_rate_used = None

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
        if needs_fx:
            item["fx_rate"] = float(fx_rate_used) if fx_rate_used is not None else tc_val if tc_val is not None else None
            item["fx_ts"] = fx_ts_used
        item["added_ts"] = int(time())
        pf["items"].append(item); save_state()

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
        await pf_refresh_menu(context, chat_id)
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
        base_conf = pf.get("base", {})
        pf_base = (base_conf.get("moneda") or "ARS").upper()
        inst_moneda = instrument_currency(yfsym or "", it.get("tipo")) if yfsym else pf_base
        needs_fx = pf_base != inst_moneda
        tc_key = (base_conf.get("tc") or "oficial").lower()
        tc_val = base_conf.get("tc_valor")
        tc_ts = base_conf.get("tc_timestamp")
        item_fx_rate = it.get("fx_rate")
        item_fx_ts = it.get("fx_ts")
        effective_tc = item_fx_rate if item_fx_rate else tc_val

        async with ClientSession() as session:
            if yfsym and (yfsym.endswith(".BA") or yfsym.endswith("-USD")):
                mets, _ = await metrics_for_symbols(session, [yfsym])
                px = mets.get(yfsym,{}).get("last_px")
            else:
                px = None
            if needs_fx and (not effective_tc or effective_tc <= 0):
                fetched_tc = await get_tc_value(session, tc_key)
                tc_ts = int(time()) if fetched_tc is not None else None
                base_conf["tc_valor"] = fetched_tc
                base_conf["tc_timestamp"] = tc_ts
                effective_tc = fetched_tc
                tc_val = fetched_tc

        price_base = None
        if px is not None:
            if pf_base == inst_moneda:
                price_base = float(px)
            else:
                if effective_tc and effective_tc > 0:
                    if pf_base == "ARS" and inst_moneda == "USD":
                        price_base = float(px) * float(effective_tc)
                    elif pf_base == "USD" and inst_moneda == "ARS":
                        price_base = float(px) / float(effective_tc)

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

        if needs_fx and effective_tc and effective_tc > 0:
            it["fx_rate"] = float(effective_tc)
            it["fx_ts"] = item_fx_ts if item_fx_rate else tc_ts

        save_state()
        usado = await _pf_total_usado(chat_id)
        f_money = fmt_money_ars if pf_base=="ARS" else fmt_money_usd
        await update.message.reply_text("Actualizado ✅ · Restante: " + f_money(max(0.0, pf["monto"]-usado)))
        await pf_refresh_menu(context, chat_id)
        context.user_data["pf_mode"]=None; return

# --- Composición: texto + torta (debajo del menú) ---

async def pf_market_snapshot(pf: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Optional[int], float, float, Optional[float], Optional[int]]:
    items = pf.get("items", [])
    base_conf = pf.get("base", {})
    base_currency = (base_conf.get("moneda") or "ARS").upper()
    tc_name = (base_conf.get("tc") or "").lower()
    tc_val_raw = base_conf.get("tc_valor")
    try:
        tc_val = float(tc_val_raw) if tc_val_raw is not None else None
    except (TypeError, ValueError):
        tc_val = None
    tc_ts_raw = base_conf.get("tc_timestamp")
    try:
        tc_ts = int(tc_ts_raw) if tc_ts_raw is not None else None
    except (TypeError, ValueError):
        tc_ts = None
    state_updated = False
    async with ClientSession() as session:
        if tc_name and (tc_val is None or tc_val <= 0):
            fetched_tc = await get_tc_value(session, tc_name)
            if fetched_tc is not None:
                tc_val = float(fetched_tc)
                tc_ts = int(time())
                base_conf["tc_valor"] = tc_val
                base_conf["tc_timestamp"] = tc_ts
                state_updated = True
        symbols = sorted({it.get("simbolo") for it in items if it.get("simbolo")})
        mets, last_ts = await metrics_for_symbols(session, symbols) if symbols else ({}, None)
    if state_updated:
        save_state()

    enriched: List[Dict[str, Any]] = []
    total_invertido = 0.0
    total_actual = 0.0
    for it in items:
        sym = it.get("simbolo", "")
        tipo = it.get("tipo", "")
        qty = float(it["cantidad"]) if it.get("cantidad") is not None else None
        invertido = float(it.get("importe") or 0.0)
        met = mets.get(sym, {}) if sym in mets else {}
        if met and met.get("last_px") is None:
            met = {}
        met_currency = met.get("currency") if met else None
        inst_cur = instrument_currency(sym, tipo) if sym else base_currency
        if met_currency:
            inst_cur = str(met_currency).upper()
        fx_rate_raw = it.get("fx_rate")
        try:
            fx_rate_item = float(fx_rate_raw) if fx_rate_raw is not None else None
        except (TypeError, ValueError):
            fx_rate_item = None
        fx_ts_raw = it.get("fx_ts")
        try:
            fx_ts_item = int(fx_ts_raw) if fx_ts_raw is not None else None
        except (TypeError, ValueError):
            fx_ts_item = None
        added_ts_raw = it.get("added_ts")
        try:
            added_ts = int(added_ts_raw) if added_ts_raw is not None else None
        except (TypeError, ValueError):
            added_ts = None
        effective_tc = fx_rate_item if fx_rate_item is not None else tc_val
        effective_ts = fx_ts_item if fx_rate_item is not None else tc_ts
        price_native = met.get("last_px") if met else None
        price_base = price_to_base(price_native, inst_cur, base_currency, effective_tc) if price_native is not None else None
        derived_qty = False
        if qty is None and price_base and price_base > 0 and invertido > 0:
            qty = invertido / price_base
            derived_qty = True
        valor_actual = invertido
        if qty is not None and price_base is not None:
            valor_actual = float(qty) * float(price_base)
        total_invertido += invertido
        total_actual += valor_actual
        label = _label_long(sym) if sym else (tipo.upper() if tipo else "Instrumento")
        enriched.append({
            "raw": it,
            "symbol": sym,
            "tipo": tipo,
            "label": label,
            "cantidad": qty,
            "cantidad_derivada": derived_qty,
            "invertido": invertido,
            "valor_actual": valor_actual,
            "precio_base": price_base,
            "metrics": met,
            "inst_currency": inst_cur,
            "daily_change": met.get("last_chg") if met else None,
            "fx_rate": effective_tc,
            "fx_ts": effective_ts,
            "added_ts": added_ts,
        })

    for entry in enriched:
        entry["peso"] = (entry["valor_actual"] / total_actual) if total_actual > 0 else 0.0

    return enriched, last_ts, total_invertido, total_actual, tc_val, tc_ts

def _bar_image_from_rank(
    rows: List[Tuple[str, List[Optional[float]]]],
    title: str,
    subtitle: Optional[str] = None,
    series_labels: Optional[List[str]] = None,
) -> Optional[bytes]:
    if not HAS_MPL:
        return None

    clean_rows: List[Tuple[str, List[Optional[float]]]] = []
    for label, values in rows:
        vals = [float(v) if v is not None else None for v in values]
        if any(v is not None for v in vals):
            clean_rows.append((label, vals))

    if not clean_rows:
        return None

    n_series = len(clean_rows[0][1]) if clean_rows[0][1] else 0
    if n_series == 0:
        return None

    for _, vals in clean_rows:
        if len(vals) != n_series:
            return None

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    fig, ax = plt.subplots(figsize=(6, 4), dpi=160)
    x_positions = list(range(len(clean_rows)))
    width = 0.75 / max(1, n_series)

    for idx in range(n_series):
        heights = [vals[idx] if vals[idx] is not None else 0.0 for _, vals in clean_rows]
        present_flags = [vals[idx] is not None for _, vals in clean_rows]
        offsets = [x + (idx - (n_series - 1) / 2) * width for x in x_positions]
        bars = ax.bar(
            offsets,
            heights,
            width=width,
            color=palette[idx % len(palette)],
            label=series_labels[idx] if series_labels and idx < len(series_labels) else None,
        )

        for bar, val, present in zip(bars, heights, present_flags):
            if not present:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    0.2,
                    "s/d",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#666666",
                )
                continue

            y = bar.get_height()
            va = "bottom" if y >= 0 else "top"
            offset = 0.6 if y >= 0 else -0.6
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y + offset,
                f"{val:.1f}%",
                ha="center",
                va=va,
                fontsize=8,
                color="#1a1a1a",
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([label for label, _ in clean_rows], rotation=15, ha="right")
    ax.axhline(0, color="#444444", linewidth=0.8)
    ax.set_ylabel("Variación %")
    ax.set_title(title + (f"\n{subtitle}" if subtitle else ""))
    if series_labels:
        ax.legend()
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _pie_image_from_items(pf: Dict[str, Any], snapshot: Optional[List[Dict[str, Any]]] = None) -> Optional[bytes]:
    if not HAS_MPL:
        return None

    pair_details: List[Dict[str, Any]] = []
    if snapshot:
        for entry in snapshot:
            val = float(entry.get("valor_actual") or 0.0)
            if val > 0:
                label = entry.get("label") or entry.get("symbol") or "Instrumento"
                pair_details.append(
                    {
                        "label": label,
                        "valor_actual": val,
                        "invertido": float(entry.get("invertido") or 0.0),
                    }
                )
    else:
        for it in pf.get("items", []):
            val = float(it.get("importe") or 0.0)
            if val > 0:
                sym = it.get("simbolo", "")
                label = _label_short(sym) if sym else (it.get("tipo", "").upper() or "Instrumento")
                pair_details.append(
                    {
                        "label": label,
                        "valor_actual": val,
                        "invertido": val,
                    }
                )

    pair_details = [detail for detail in pair_details if detail["valor_actual"] > 0]
    if not pair_details:
        return None

    pair_details.sort(key=lambda x: x["valor_actual"], reverse=True)
    total = sum(detail["valor_actual"] for detail in pair_details)
    if total <= 0:
        return None

    base_currency = pf.get("base", {}).get("moneda", "ARS").upper()
    f_money = fmt_money_ars if base_currency == "ARS" else fmt_money_usd

    selected_details: List[Dict[str, Any]] = []
    otros_bucket: List[Dict[str, Any]] = []
    if len(pair_details) > 6:
        for detail in pair_details:
            fraction = detail["valor_actual"] / total if total else 0.0
            if fraction < 0.03:
                otros_bucket.append(detail)
            else:
                selected_details.append(detail)
    else:
        selected_details = list(pair_details)

    if otros_bucket:
        selected_details.append(
            {
                "label": "Otros",
                "valor_actual": sum(d["valor_actual"] for d in otros_bucket),
                "invertido": sum(d.get("invertido", 0.0) for d in otros_bucket),
            }
        )

    labels2 = [detail["label"] for detail in selected_details]
    vals2 = [detail["valor_actual"] for detail in selected_details]

    fig, (ax_pie, ax_info) = plt.subplots(
        1,
        2,
        figsize=(10, 6),
        dpi=200,
        gridspec_kw={"width_ratios": [3, 2]},
    )

    cmap = plt.get_cmap("tab20c")
    color_positions = np.linspace(0, 1, len(vals2)) if vals2 else []
    colors = [cmap(pos) for pos in color_positions]

    def autopct_fmt(pct: float) -> str:
        value = total * pct / 100.0
        return f"{pct_plain(pct, 1)}\n{f_money(value)}"

    _, _, autotexts = ax_pie.pie(
        vals2,
        labels=labels2,
        autopct=autopct_fmt,
        pctdistance=0.75,
        startangle=90,
        colors=colors,
        wedgeprops=dict(width=0.35, edgecolor="white"),
    )

    for text in autotexts:
        text.set_color("#1a1a1a")
        text.set_fontsize(9)

    ax_pie.text(
        0,
        0,
        f"{f_money(total)}\n{base_currency}",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#1a1a1a",
    )
    ax_pie.set_aspect("equal")
    ax_pie.axis("off")

    ax_info.axis("off")
    ax_info.set_xlim(0, 1.05)
    ax_info.set_ylim(0, 1)

    ax_info.text(0.02, 0.95, "Detalle por instrumento", fontsize=12, fontweight="bold", va="center")
    headers = ["Instrumento", "%", "Actual", "Invertido", "Variación"]
    col_x = [0.08, 0.4, 0.66, 0.86, 1.02]
    header_y = 0.85
    for x, header in zip(col_x, headers):
        ax_info.text(x, header_y, header, fontsize=10, fontweight="bold", va="center")

    n_rows = len(selected_details)
    row_spacing = 0.65 / max(1, n_rows)
    start_y = header_y - row_spacing

    for color, detail in zip(colors, selected_details):
        pct_value = detail["valor_actual"] / total * 100.0 if total else 0.0
        invertido = detail.get("invertido", 0.0)
        variacion = detail["valor_actual"] - invertido
        ax_info.scatter(0.03, start_y, color=color, s=80, marker="s")
        ax_info.text(0.08, start_y, detail["label"], fontsize=9, va="center")
        ax_info.text(0.4, start_y, pct_plain(pct_value, 1), fontsize=9, va="center")
        ax_info.text(0.66, start_y, f_money(detail["valor_actual"]), fontsize=9, va="center")
        ax_info.text(0.86, start_y, f_money(invertido), fontsize=9, va="center")
        ax_info.text(1.02, start_y, f_money(variacion), fontsize=9, va="center")
        start_y -= row_spacing

    ax_info.text(
        0.02,
        max(0.05, start_y),
        f"Total: {f_money(total)} {base_currency}",
        fontsize=10,
        fontweight="bold",
        va="center",
    )

    fig.subplots_adjust(wspace=0.15)

    fig.suptitle("Composición del Portafolio", fontsize=14, fontweight="bold")
    buf = io.BytesIO()
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _projection_bar_image(
    points: List[Tuple[str, Optional[float]]],
    formatter: Callable[[Optional[float]], str],
    title: str,
    subtitle: Optional[str] = None,
) -> Optional[bytes]:
    if not HAS_MPL:
        return None

    cleaned: List[Tuple[str, float]] = []
    for label, value in points:
        if value is None:
            continue
        try:
            cleaned.append((label, float(value)))
        except (TypeError, ValueError):
            continue

    if len(cleaned) < 2:
        return None

    labels = [label for label, _ in cleaned]
    values = [val for _, val in cleaned]
    max_val = max(values)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=160)
    palette = ["#3478bc", "#34a853", "#fbbc04", "#a142f4", "#f26f5e"]
    bars = ax.bar(range(len(values)), values, color=palette[: len(values)])

    max_display = max_val if max_val else 1.0
    inner_margin = max_display * 0.04
    top_margin = max_display * 0.08
    for bar, val in zip(bars, values):
        label = formatter(val)
        text_x = bar.get_x() + bar.get_width() / 2
        height = bar.get_height()
        rotation = 90 if len(label) > 12 else 0

        if height <= 0:
            ax.text(
                text_x,
                height + top_margin,
                label,
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=rotation,
                color="#1a1a1a",
            )
            continue

        if height >= max_display * 0.18:
            text_y = height - inner_margin
            va = "top"
            color = "#ffffff"
        else:
            text_y = height + inner_margin
            va = "bottom"
            color = "#1a1a1a"

        ax.text(
            text_x,
            text_y,
            label,
            ha="center",
            va=va,
            fontsize=8,
            rotation=rotation,
            color=color,
        )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Monto estimado")
    ax.set_title(title + (f"\n{subtitle}" if subtitle else ""))
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax.set_ylim(0, max_display * 1.1)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _format_signed_money(value: float, money_formatter: Callable[[Optional[float]], str]) -> str:
    formatted = money_formatter(abs(value))
    if value > 0:
        return f"+{formatted}"
    if value < 0:
        return f"-{formatted}"
    return money_formatter(0.0)


def _projection_vs_return_image(
    entries: List[Dict[str, Optional[float]]],
    title: str,
    subtitle: Optional[str] = None,
    value_formatter: Optional[Callable[[float], str]] = None,
    ylabel: str = "Variación %",
    line_label: str = "Rendimiento actual",
) -> Optional[bytes]:
    if not HAS_MPL:
        return None

    cleaned: List[Dict[str, float]] = []
    for entry in entries:
        label = entry.get("label")
        if not label:
            continue
        try:
            proj3 = float(entry["proj3"]) if entry.get("proj3") is not None else math.nan
        except (TypeError, ValueError):
            proj3 = math.nan
        try:
            proj6 = float(entry["proj6"]) if entry.get("proj6") is not None else math.nan
        except (TypeError, ValueError):
            proj6 = math.nan
        try:
            actual = float(entry["actual"]) if entry.get("actual") is not None else math.nan
        except (TypeError, ValueError):
            actual = math.nan

        if math.isnan(actual) and math.isnan(proj3) and math.isnan(proj6):
            continue
        cleaned.append(
            {
                "label": str(label),
                "proj3": proj3,
                "proj6": proj6,
                "actual": actual,
            }
        )

    if not cleaned:
        return None

    labels = [entry["label"] for entry in cleaned]
    proj3_vals = [entry["proj3"] for entry in cleaned]
    proj6_vals = [entry["proj6"] for entry in cleaned]
    actual_vals = [entry["actual"] for entry in cleaned]

    has_proj3 = any(not math.isnan(val) for val in proj3_vals)
    has_proj6 = any(not math.isnan(val) for val in proj6_vals)

    x = np.arange(len(labels))
    width = 0.34 if has_proj3 and has_proj6 else 0.44

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=180)
    fig.patch.set_facecolor("#f8f9fb")
    ax.set_facecolor("#f8f9fb")

    bar_series: List[Tuple[Any, np.ndarray]] = []

    formatter = value_formatter or (lambda v: f"{v:+.1f}%")

    if has_proj3:
        offset = -width / 2 if has_proj6 else 0.0
        heights = np.array(proj3_vals, dtype=float)
        bars = ax.bar(
            x + offset,
            heights,
            width=width,
            label="Proyección 3M",
            color="#3b82f6",
            alpha=0.85,
        )
        bar_series.append((bars, heights))

    if has_proj6:
        offset = width / 2 if has_proj3 else 0.0
        heights = np.array(proj6_vals, dtype=float)
        bars = ax.bar(
            x + offset,
            heights,
            width=width,
            label="Proyección 6M",
            color="#60a5fa",
            alpha=0.85,
        )
        bar_series.append((bars, heights))

    actual_arr = np.array(actual_vals, dtype=float)
    masked_actual = np.ma.masked_invalid(actual_arr)
    ax.plot(
        x,
        masked_actual,
        color="#1d4ed8",
        marker="o",
        markersize=6,
        linewidth=2.2,
        label=line_label,
        markerfacecolor="#ffffff",
        markeredgewidth=1.6,
        zorder=3,
    )

    for bars, heights in bar_series:
        for rect, val in zip(bars, heights):
            if math.isnan(val):
                continue
            height = rect.get_height()
            offset = max(1.0, abs(height) * 0.08)
            if height >= 0:
                y = height + offset
                va = "bottom"
            else:
                y = height - offset
                va = "top"
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                y,
                formatter(val),
                ha="center",
                va=va,
                fontsize=8,
                color="#1a1a1a",
            )

    for xi, val in zip(x, actual_arr):
        if math.isnan(val):
            continue
        offset = max(1.0, abs(val) * 0.08)
        if val >= 0:
            y = val + offset
            va = "bottom"
        else:
            y = val - offset
            va = "top"
        ax.text(
            xi,
            y,
            formatter(val),
            ha="center",
            va=va,
            fontsize=8,
            color="#1a1a1a",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title + (f"\n{subtitle}" if subtitle else ""))

    ax.legend()
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.axhline(0, color="#d1d5db", linewidth=0.9)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _return_bar_image(
    points: List[Tuple[str, Optional[float]]],
    title: str,
    subtitle: Optional[str] = None,
    formatter: Optional[Callable[[float], str]] = None,
) -> Optional[bytes]:
    if not HAS_MPL:
        return None

    cleaned: List[Tuple[str, float]] = []
    for label, value in points:
        if value is None:
            continue
        try:
            cleaned.append((label, float(value)))
        except (TypeError, ValueError):
            continue

    if not cleaned:
        return None

    labels = [label for label, _ in cleaned]
    values = [val for _, val in cleaned]
    max_abs = max(abs(val) for val in values) if values else 0.0
    if max_abs <= 0:
        max_abs = 1.0

    fig, ax = plt.subplots(figsize=(6, 4), dpi=160)
    colors = []
    for val in values:
        if val > 0:
            colors.append("#34a853")
        elif val < 0:
            colors.append("#ea4335")
        else:
            colors.append("#9aa0a6")

    bars = ax.bar(range(len(values)), values, color=colors)

    inner_offset = max_abs * 0.06
    outer_offset = max_abs * 0.08
    for bar, val in zip(bars, values):
        text = formatter(val) if formatter else f"{val:+.1f}%"
        text_x = bar.get_x() + bar.get_width() / 2
        height = bar.get_height()

        if val > 0:
            if height > max_abs * 0.18:
                y = height - inner_offset
                va = "top"
                color = "#ffffff"
            else:
                y = height + outer_offset
                va = "bottom"
                color = "#1a1a1a"
        elif val < 0:
            if abs(height) > max_abs * 0.18:
                y = height + inner_offset
                va = "bottom"
                color = "#ffffff"
            else:
                y = height - outer_offset
                va = "top"
                color = "#1a1a1a"
        else:
            y = height + outer_offset
            va = "bottom"
            color = "#1a1a1a"

        ax.text(
            text_x,
            y,
            text,
            ha="center",
            va=va,
            fontsize=8,
            color=color,
        )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.axhline(0, color="#4a4a4a", linewidth=0.8)
    ax.set_ylabel("Variación %")
    ax.set_title(title + (f"\n{subtitle}" if subtitle else ""))
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ylim = max_abs * 1.25
    ax.set_ylim(-ylim, ylim)

    fig.tight_layout()
    buf = io.BytesIO()
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
    snapshot, last_ts, total_invertido, total_actual, tc_val, tc_ts = await pf_market_snapshot(pf)
    fecha = datetime.fromtimestamp(last_ts, TZ).strftime("%d/%m/%Y") if last_ts else None
    header = f"<b>Portafolio</b> — Base: {pf['base']['moneda'].upper()}/{pf['base']['tc'].upper()}"
    if fecha:
        header += f" <i>Datos al {fecha}</i>"
    lines = [header, f"Monto objetivo: {f_money(pf['monto'])}"]
    lines.append(f"Valor invertido: {f_money(total_invertido)}")
    lines.append(f"Valor actual estimado: {f_money(total_actual)}")
    delta = total_actual - total_invertido
    if total_invertido > 0:
        lines.append(f"Variación estimada: {f_money(delta)} ({pct(delta/total_invertido*100.0,2)})")
    restante = max(0.0, pf['monto'] - total_invertido)
    lines.append(f"Restante del objetivo: {f_money(restante)}")
    if tc_val is not None:
        tc_line = f"Tipo de cambio ref. ({pf['base']['tc'].upper()}): {fmt_money_ars(tc_val)} por USD"
        if tc_ts:
            tc_line += f" (al {datetime.fromtimestamp(tc_ts, TZ).strftime('%d/%m/%Y %H:%M')})"
        lines.append(tc_line)
    lines.append("")
    for i, entry in enumerate(snapshot, 1):
        if i > 1:
            lines.append("")
        linea = f"{i}. {entry['label']}"
        linea += f" · Valor: {f_money(entry['valor_actual'])}"
        if entry['invertido'] > 0:
            r_ind = (entry['valor_actual']/entry['invertido']-1.0)*100.0
            linea += f" ({pct(r_ind,2)} vs {f_money(entry['invertido'])})"
        qty_txt = format_quantity(entry['symbol'], entry.get('cantidad'))
        if qty_txt:
            linea += f" · Cant: {qty_txt}"
        if entry.get('peso'):
            linea += f" · Peso: {pct_plain(entry['peso']*100.0,1)}"
        added_str = format_added_date(entry.get('added_ts'))
        if added_str:
            linea += f" · Desde: {added_str}"
        lines.append(linea)
    if not HAS_MPL:
        lines.append("")
        lines.append("ℹ️ Instalá matplotlib para ver la composición en gráficos.")
    await _send_below_menu(context, chat_id, text="\n".join(lines))
    # torta
    img = _pie_image_from_items(pf, snapshot)
    if img:
        await _send_below_menu(context, chat_id, photo_bytes=img)

# --- Rendimiento (debajo del menú) ---

async def pf_show_return_below(context: ContextTypes.DEFAULT_TYPE, chat_id: int):
    pf = pf_get(chat_id)
    if not pf["items"]:
        await _send_below_menu(context, chat_id, text="Tu portafolio está vacío. Agregá instrumentos primero."); return
    pf_base = pf["base"]["moneda"].upper()
    f_money = fmt_money_ars if pf_base=="ARS" else fmt_money_usd
    snapshot, last_ts, total_invertido, total_actual, tc_val, tc_ts = await pf_market_snapshot(pf)
    fecha = datetime.fromtimestamp(last_ts, TZ).strftime("%d/%m/%Y") if last_ts else None
    header = "<b>📈 Rendimiento del portafolio</b>"
    if fecha:
        header += f" <i>Datos al {fecha}</i>"
    lines = [header]
    if tc_val is not None:
        tc_line = f"Tipo de cambio ref. ({pf['base']['tc'].upper()}): {fmt_money_ars(tc_val)} por USD"
        if tc_ts:
            tc_line += f" (al {datetime.fromtimestamp(tc_ts, TZ).strftime('%d/%m/%Y %H:%M')})"
        lines.append(tc_line)

    port_daily_vals = [entry['peso'] * entry['daily_change'] for entry in snapshot if entry.get('daily_change') is not None]
    daily_sum: Optional[float] = None
    if port_daily_vals:
        daily_sum = sum(port_daily_vals)
        lines.append(f"Variación diaria estimada: {pct(daily_sum,2)}")

    has_daily_data = any(entry.get('daily_change') is not None for entry in snapshot)
    return_points: List[Tuple[str, Optional[float]]] = []
    daily_points: List[Tuple[str, Optional[float]]] = []
    for idx, entry in enumerate(snapshot):
        label = entry['label']
        valor_actual = entry['valor_actual']
        invertido = entry['invertido']
        delta = valor_actual - invertido
        ret_pct = (delta / invertido * 100.0) if invertido > 0 else None
        if idx > 0:
            lines.append("")
        detail = f"• {label}: {f_money(valor_actual)}"
        if ret_pct is not None:
            detail += f" ({pct(ret_pct,2)} | Δ {f_money(delta)})"
        elif invertido > 0:
            detail += f" (Δ {f_money(delta)})"
        qty_txt = format_quantity(entry['symbol'], entry.get('cantidad'))
        if qty_txt:
            detail += f" · Cant: {qty_txt}"
        if entry.get('precio_base') is not None:
            detail += f" · Px: {f_money(entry['precio_base'])}"
        daily = entry.get('daily_change')
        if daily is not None:
            detail += f" · Día: {pct(daily,2)}"
        if entry.get('peso'):
            detail += f" · Peso: {pct_plain(entry['peso']*100.0,1)}"
        added_str = format_added_date(entry.get('added_ts'))
        if added_str:
            detail += f" · Desde: {added_str}"
        lines.append(detail)

        short_label = _label_short(entry['symbol']) if entry.get('symbol') else label
        if ret_pct is not None:
            return_points.append((short_label, ret_pct))
        if has_daily_data:
            daily_points.append((short_label, daily if daily is not None else None))

    delta_t = total_actual - total_invertido
    lines.append("")
    lines.append(f"Invertido: {f_money(total_invertido)}")
    lines.append(f"Valor actual estimado: {f_money(total_actual)}")
    if total_invertido > 0:
        lines.append(f"Variación total: {f_money(delta_t)} ({pct((delta_t/total_invertido)*100.0,2)})")
    else:
        lines.append(f"Variación total: {f_money(delta_t)}")

    sin_datos = [entry['label'] for entry in snapshot if not entry.get('metrics')]
    if sin_datos:
        lines.append("")
        lines.append("Sin datos recientes para: " + ", ".join(sin_datos) + ". Se mantiene el valor cargado.")

    if not HAS_MPL:
        lines.append("")
        lines.append("ℹ️ Instalá matplotlib para ver el gráfico de rendimiento.")

    await _send_below_menu(context, chat_id, text="\n".join(lines))

    total_pct = (delta_t / total_invertido * 100.0) if total_invertido > 0 else None
    if total_pct is not None:
        return_points.append(("Portafolio", total_pct))
    if has_daily_data:
        daily_points.append(("Portafolio", daily_sum if daily_sum is not None else None))

    if return_points:
        img = _return_bar_image(
            return_points,
            "Rendimiento por instrumento",
            "Variación acumulada vs. invertido",
            formatter=lambda v: f"{v:+.1f}%",
        )
        if img:
            await _send_below_menu(context, chat_id, photo_bytes=img)

    cleaned_daily = [pt for pt in daily_points if pt[1] is not None]
    if cleaned_daily:
        daily_img = _return_bar_image(
            cleaned_daily,
            "Variación diaria",
            "Cambios porcentuales del día",
            formatter=lambda v: f"{v:+.2f}%",
        )
        if daily_img:
            await _send_below_menu(context, chat_id, photo_bytes=daily_img)

# --- Proyección (debajo del menú) ---

async def pf_show_projection_below(context: ContextTypes.DEFAULT_TYPE, chat_id: int):
    pf = pf_get(chat_id)
    if not pf["items"]:
        await _send_below_menu(context, chat_id, text="Tu portafolio está vacío. Agregá instrumentos primero."); return
    snapshot, last_ts, total_invertido, total_actual, tc_val, tc_ts = await pf_market_snapshot(pf)
    if total_actual <= 0:
        await _send_below_menu(context, chat_id, text="Sin valores suficientes para proyectar."); return

    fecha = datetime.fromtimestamp(last_ts, TZ).strftime("%d/%m/%Y") if last_ts else None
    pf_base = pf["base"]["moneda"].upper()
    f_money = fmt_money_ars if pf_base=="ARS" else fmt_money_usd

    w3 = 0.0
    w6 = 0.0
    detail: List[str] = []
    comparison_points: List[Dict[str, Optional[float]]] = []
    profit_points: List[Dict[str, Optional[float]]] = []
    has_projection_data = False
    for entry in snapshot:
        metrics = entry.get('metrics') or {}
        weight = entry.get('peso') or 0.0
        if not metrics:
            continue
        has_projection_data = True
        p3 = projection_3m(metrics)
        p6 = projection_6m(metrics)
        w3 += weight * p3
        w6 += weight * p6
        short_label = _label_short(entry['symbol']) if entry.get('symbol') else entry['label']
        if detail:
            detail.append("")
        extras = [f"peso {pct_plain(weight*100.0,1)}"]
        added_str = format_added_date(entry.get('added_ts'))
        if added_str:
            extras.append(f"desde {added_str}")
        detail.append(
            f"• {short_label} → 3M {pct(p3,2)} | 6M {pct(p6,2)} (" + " · ".join(extras) + ")"
        )

        invertido = float(entry.get('invertido') or 0.0)
        valor_actual = float(entry.get('valor_actual') or 0.0)
        actual_pct = ((valor_actual / invertido) - 1.0) * 100.0 if invertido > 0 else None
        comparison_points.append(
            {
                "label": short_label,
                "proj3": p3,
                "proj6": p6,
                "actual": actual_pct,
            }
        )

        actual_profit = valor_actual - invertido
        proj_profit3 = valor_actual * (p3 / 100.0)
        proj_profit6 = valor_actual * (p6 / 100.0)
        profit_points.append(
            {
                "label": short_label,
                "proj3": proj_profit3,
                "proj6": proj_profit6,
                "actual": actual_profit,
            }
        )

    forecast3 = total_actual * (1.0 + w3/100.0)
    forecast6 = total_actual * (1.0 + w6/100.0)
    total_pct = ((total_actual / total_invertido) - 1.0) * 100.0 if total_invertido > 0 else math.nan

    header = "<b>🔮 Proyección del Portafolio</b>"
    if fecha:
        header += f" <i>Datos al {fecha}</i>"
    lines = [header, f"Valor actual estimado: {f_money(total_actual)}"]
    lines.append(f"Proyección 3M: {pct(w3,2)} → {f_money(forecast3)}")
    lines.append(f"Proyección 6M: {pct(w6,2)} → {f_money(forecast6)}")
    if tc_val is not None:
        tc_line = f"Tipo de cambio ref. ({pf['base']['tc'].upper()}): {fmt_money_ars(tc_val)} por USD"
        if tc_ts:
            tc_line += f" (al {datetime.fromtimestamp(tc_ts, TZ).strftime('%d/%m/%Y %H:%M')})"
        lines.append(tc_line)

    if detail:
        lines.append("")
        lines.extend(detail)

    sin_datos = [entry['label'] for entry in snapshot if not entry.get('metrics')]
    if sin_datos:
        lines.append("")
        lines.append("Sin datos de mercado para: " + ", ".join(sin_datos) + ". Se asumió variación 0%.")

    if not HAS_MPL:
        lines.append("")
        lines.append("ℹ️ Instalá matplotlib para ver la proyección en gráficos.")

    await _send_below_menu(context, chat_id, text="\n".join(lines))

    img = _projection_bar_image(
        [
            ("Actual", total_actual),
            ("3M", forecast3),
            ("6M", forecast6),
        ],
        f_money,
        "Proyección del portafolio",
        "Valores estimados",
    )
    if img:
        await _send_below_menu(context, chat_id, photo_bytes=img)

    comparison_clean: List[Dict[str, Optional[float]]] = []
    for pt in comparison_points:
        raw_actual = pt.get("actual")
        try:
            actual_val = float(raw_actual) if raw_actual is not None else math.nan
        except (TypeError, ValueError):
            actual_val = math.nan
        if math.isnan(actual_val):
            continue
        if pt.get("proj3") is None and pt.get("proj6") is None:
            continue
        cleaned_entry = dict(pt)
        cleaned_entry["actual"] = actual_val
        comparison_clean.append(cleaned_entry)
    if comparison_clean:
        comp_img = _projection_vs_return_image(
            comparison_clean,
            "Proyección vs. rendimiento por instrumento",
            "Variaciones estimadas vs. reales",
        )
        if comp_img:
            await _send_below_menu(context, chat_id, photo_bytes=comp_img)

    money_formatter = lambda v: _format_signed_money(v, f_money)

    profit_clean: List[Dict[str, float]] = []
    for pt in profit_points:
        try:
            actual_val = float(pt.get("actual")) if pt.get("actual") is not None else math.nan
        except (TypeError, ValueError):
            actual_val = math.nan
        if math.isnan(actual_val):
            continue
        entry = {"label": pt.get("label", ""), "actual": actual_val}
        for key in ("proj3", "proj6"):
            try:
                entry[key] = float(pt.get(key)) if pt.get(key) is not None else math.nan
            except (TypeError, ValueError):
                entry[key] = math.nan
        profit_clean.append(entry)

    if profit_clean:
        profit_img = _projection_vs_return_image(
            profit_clean,
            "Ganancia proyectada vs. actual por instrumento",
            f"Montos estimados en {pf_base}",
            value_formatter=money_formatter,
            ylabel=f"Ganancias / pérdidas ({pf_base})",
            line_label="Ganancia actual",
        )
        if profit_img:
            await _send_below_menu(context, chat_id, photo_bytes=profit_img)

    if has_projection_data and not math.isnan(total_pct):
        total_profit = total_actual - total_invertido
        total_proj3 = total_actual * (w3 / 100.0)
        total_proj6 = total_actual * (w6 / 100.0)
        overall_img = _projection_vs_return_image(
            [
                {
                    "label": "Portafolio",
                    "proj3": total_proj3,
                    "proj6": total_proj6,
                    "actual": total_profit,
                }
            ],
            "Ganancia proyectada vs. actual del portafolio",
            f"Montos estimados en {pf_base}",
            value_formatter=money_formatter,
            ylabel=f"Ganancias / pérdidas ({pf_base})",
            line_label="Ganancia actual",
        )
        if overall_img:
            await _send_below_menu(context, chat_id, photo_bytes=overall_img)

# ============================ RESUMEN DIARIO ============================

async def cmd_resumen_diario(update: Update, context: ContextTypes.DEFAULT_TYPE):
    async with ClientSession() as session:
        fx = await get_dolares(session)
        rp = await get_riesgo_pais(session)
        infl = await get_inflacion_mensual(session)
        rv = await get_reservas_lamacro(session)
        news = await fetch_rss_entries(session, limit=3)

    partes = []
    if fx:
        partes.append(format_dolar_message(fx))
    if rp:
        partes.append(f"<b>📈 Riesgo País</b> {rp[0]} pb" + (f" <i>({parse_iso_ddmmyyyy(rp[1])})</i>" if rp[1] else ""))
    if infl:
        partes.append(f"<b>📉 Inflación Mensual</b> {str(round(infl[0],1)).replace('.',',')}%" + (f" <i>({infl[1]})</i>" if infl[1] else ""))
    if rv:
        partes.append(f"<b>🏦 Reservas</b> {fmt_number(rv[0],0)} MUS$" + (f" <i>({rv[1]})</i>" if rv[1] else ""))
    if news:
        partes.append(format_news_block(news)[0])

    txt = "\n\n".join(partes) if partes else "Sin datos para el resumen ahora."
    await update.effective_message.reply_text(txt, parse_mode=ParseMode.HTML, link_preview_options=LinkPreviewOptions(is_disabled=True))

# ============================ WEBHOOK / APP ============================

async def keepalive_loop():
    try:
        await asyncio.sleep(5)
        url = f"{BASE_URL}/"; timeout = ClientTimeout(total=6)
        async with ClientSession(timeout=timeout) as session:
            while True:
                try:
                    async with session.get(url) as resp:
                        logging.info("Keepalive %s -> %s", url, resp.status)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logging.warning("Keepalive error: %s", e)
                await asyncio.sleep(300)
    except asyncio.CancelledError:
        logging.info("keepalive_loop cancelado")
        raise


def setup_health_routes(application: Application) -> None:
    updater = getattr(application, "updater", None)
    if not updater:
        logging.warning("No updater disponible para configurar healthchecks")
        return

    webhook_app = getattr(updater, "webhook_app", None)
    if webhook_app is None:
        logging.warning("Webhook app no disponible para configurar healthchecks")
        return

    if hasattr(webhook_app, "app"):
        inner_app = getattr(webhook_app, "app", None)
    else:
        inner_app = webhook_app

    if not isinstance(inner_app, web.Application):
        logging.warning("Webhook app no disponible para configurar healthchecks")
        return

    async def _health(_: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    router = inner_app.router

    for path in ("/", "/healthz"):
        already_registered = False
        for route in router.routes():
            resource = getattr(route, "resource", None)
            canonical = getattr(resource, "canonical", None) if resource else None
            if canonical != path:
                continue
            method = getattr(route, "method", None)
            if method in (None, "*", "GET"):
                already_registered = True
                break
        if already_registered:
            continue
        try:
            router.add_get(path, _health)
        except (RuntimeError, ValueError) as exc:
            logging.debug("No se pudo registrar ruta %s: %s", path, exc)

BOT_COMMANDS = [
    BotCommand("economia","Menú de economía"),
    BotCommand("acciones","Menú acciones"),
    BotCommand("cedears","Menú cedears"),
    BotCommand("alertas_menu","Configurar alertas"),
    BotCommand("portafolio","Menú portafolio"),
    BotCommand("subs","Suscripción a resumen diario"),
]


def build_application() -> Application:
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    # Comandos
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("economia", cmd_menu_economia))
    app.add_handler(CommandHandler("dolar", cmd_dolar))
    app.add_handler(CommandHandler("reservas", cmd_reservas))
    app.add_handler(CommandHandler("inflacion", cmd_inflacion))
    app.add_handler(CommandHandler("riesgo", cmd_riesgo))
    app.add_handler(CommandHandler("noticias", cmd_noticias))
    app.add_handler(CommandHandler("acciones", cmd_acciones_menu))
    app.add_handler(CommandHandler("cedears", cmd_cedears_menu))
    app.add_handler(CallbackQueryHandler(acc_ced_cb, pattern="^(ACC:|CED:)"))
    app.add_handler(CallbackQueryHandler(econ_cb, pattern="^ECO:"))

    # Alertas - menú simple
    app.add_handler(CommandHandler("alertas_menu", cmd_alertas_menu))
    app.add_handler(CallbackQueryHandler(alertas_menu_cb, pattern="^AL:(LIST|CLEAR|PAUSE|RESUME)$"))
    app.add_handler(CallbackQueryHandler(alertas_clear_cb, pattern="^CLR:"))
    app.add_handler(CommandHandler("alertas_pause", cmd_alertas_pause))
    app.add_handler(CallbackQueryHandler(alerts_pause_cb, pattern="^AP:"))

    # Alertas - conversación Agregar
    alert_conv = ConversationHandler(
        entry_points=[CallbackQueryHandler(alertas_add_start, pattern="^AL:ADD$")],
        states={
            AL_KIND: [
                CallbackQueryHandler(alertas_add_suggestion, pattern="^SUG:"),
                CallbackQueryHandler(alertas_add_kind, pattern="^(KIND:|CANCEL$)"),
            ],
            AL_FX_TYPE: [CallbackQueryHandler(alertas_add_fx_type, pattern="^(FXTYPE:|BACK:|CANCEL$)")],
            AL_FX_SIDE: [CallbackQueryHandler(alertas_add_fx_side, pattern="^(SIDE:|BACK:|CANCEL$)")],
            AL_METRIC_TYPE: [CallbackQueryHandler(alertas_add_metric_type, pattern="^(METRIC:|BACK:|CANCEL$)")],
            AL_TICKER: [CallbackQueryHandler(alertas_add_ticker_cb, pattern="^(TICK:|BACK:|CANCEL$)")],
            AL_CRYPTO: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, alertas_crypto_query),
                CallbackQueryHandler(alertas_crypto_pick_cb, pattern="^(CRYPTOSEL:|CRYPTO:SEARCH|BACK:|CANCEL$)"),
            ],
            AL_OP: [CallbackQueryHandler(alertas_add_op, pattern="^(OP:|BACK:|CANCEL$)")],
            AL_MODE: [CallbackQueryHandler(alertas_add_mode, pattern="^(MODE:|BACK:|CANCEL$)")],
            AL_VALUE: [MessageHandler(filters.TEXT & ~filters.COMMAND, alertas_add_value)],
        },
        fallbacks=[
            CallbackQueryHandler(alertas_back, pattern="^BACK:"),
            CallbackQueryHandler(alertas_add_start, pattern="^AL:ADD$"),
            CallbackQueryHandler(alertas_add_exit_to_menu, pattern="^AL:MENU$"),
        ],
        per_chat=True,
        per_user=True,
        per_message=False,
    )
    app.add_handler(alert_conv)

    # Suscripciones
    subs_conv = ConversationHandler(
        entry_points=[
            CommandHandler("subs", cmd_subs),
            CallbackQueryHandler(subs_start_from_cb, pattern="^ST:SUBS$"),
        ],
        states={SUBS_SET_TIME: [CallbackQueryHandler(subs_cb, pattern="^SUBS:")]},
        fallbacks=[],
        per_chat=True,
        per_user=True,
        per_message=False,
    )
    app.add_handler(subs_conv)

    # Portafolio
    app.add_handler(CommandHandler("portafolio", cmd_portafolio))
    app.add_handler(CallbackQueryHandler(pf_menu_cb, pattern="^PF:"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, pf_text_input))

    # Resumen diario on-demand
    app.add_handler(CommandHandler("resumen", cmd_resumen_diario))

    return app

async def main():
    load_state()
    application = build_application()
    _schedule_all_subs(application)

    alerts_task = None
    keepalive_task = None
    updater_started = False
    app_started = False

    async with application:
        alerts_task = asyncio.create_task(alerts_loop(application))
        keepalive_task = asyncio.create_task(keepalive_loop())
        try:
            await application.bot.set_my_commands(BOT_COMMANDS)
            await application.updater.start_webhook(
                listen="0.0.0.0",
                port=PORT,
                url_path=WEBHOOK_SECRET,
                webhook_url=WEBHOOK_URL,
                drop_pending_updates=True,
            )
            setup_health_routes(application)
            updater_started = True

            await application.start()
            app_started = True

            loop = asyncio.get_running_loop()
            stop_future = loop.create_future()
            stop_signals = (signal.SIGINT, signal.SIGTERM)
            for sig in stop_signals:
                try:
                    loop.add_signal_handler(sig, stop_future.cancel)
                except NotImplementedError:
                    pass
            try:
                await stop_future
            except asyncio.CancelledError:
                pass
            finally:
                for sig in stop_signals:
                    try:
                        loop.remove_signal_handler(sig)
                    except (NotImplementedError, RuntimeError, ValueError):
                        pass
        except asyncio.CancelledError:
            pass
        finally:
            if alerts_task:
                alerts_task.cancel()
            if keepalive_task:
                keepalive_task.cancel()
            if alerts_task or keepalive_task:
                await asyncio.gather(
                    *(t for t in (alerts_task, keepalive_task) if t),
                    return_exceptions=True,
                )
            if updater_started:
                await application.updater.stop()
            if app_started:
                await application.stop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
