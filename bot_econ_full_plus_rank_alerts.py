# bot_econ_full_plus_rank_alerts.py
# -*- coding: utf-8 -*-

import os, asyncio, logging, re, html as _html, json, math, io, signal, csv, unicodedata, textwrap
import copy
import urllib.request
import urllib.error
from time import time
from math import sqrt, floor
from datetime import datetime, timedelta, time as dtime, date
from zoneinfo import ZoneInfo
from typing import Dict, List, Tuple, Any, Optional, Set, Callable, Awaitable, Iterable
from urllib.parse import urlparse, quote, parse_qs, urljoin
import httpx
import certifi

from bot.config import config
from bot.constants import WINDOW_DAYS, WINDOW_MONTHS
from bot.services.cache import RateLimiter, ShortCache
from bot.services.http import SourceSuspendedError, http_service

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

from aiohttp import ClientError, ClientSession, ClientTimeout, web
from telegram import (
    Update, LinkPreviewOptions, BotCommand, InlineKeyboardMarkup, InlineKeyboardButton,
)
from telegram.constants import ParseMode
from telegram.ext import (
    Application, CommandHandler, ContextTypes, CallbackQueryHandler,
    MessageHandler, ConversationHandler, filters
)
from metrics import metrics

from bot.persistence.state import (
    CURRENT_STATE_VERSION,
    JsonFileStore,
    RedisStore,
    StateStore,
    deserialize_state_payload,
    ensure_writable_path,
    serialize_state_payload,
)

# ============================ CONFIG ============================

TZ = ZoneInfo("America/Argentina/Buenos_Aires")

LINK_PREVIEWS_ENABLED = config.link_previews_enabled
LINK_PREVIEWS_PREFER_SMALL = config.link_previews_prefer_small
ALERTS_PAGE_SIZE = config.alerts_page_size
RANK_TOP_LIMIT = config.rank_top_limit
RANK_PROJ_LIMIT = config.rank_proj_limit
TELEGRAM_TOKEN = config.telegram_token
WEBHOOK_SECRET = config.webhook_secret
PORT = config.port
BASE_URL = config.base_url
UPSTASH_URL = config.upstash.rest_url
UPSTASH_TOKEN = config.upstash.rest_token
UPSTASH_REDIS_URL = config.upstash.redis_url
UPSTASH_STATE_KEY = config.upstash.state_key

WEBHOOK_PATH = f"/{WEBHOOK_SECRET}"
WEBHOOK_URL = f"{BASE_URL}{WEBHOOK_PATH}"

CRYPTOYA_DOLAR_URL = "https://criptoya.com/api/dolar"
DOLARAPI_BASE = "https://dolarapi.com/v1"
BANDAS_CAMBIARIAS_URL = f"{DOLARAPI_BASE}/bandas-cambiarias"
DOLARITO_BANDAS_HTML = "https://dolarito.ar/dolar/bandas-cambiarias"
DOLARITO_BANDAS_JSON = "https://api.dolarito.ar/api/frontend/bandas-cambiarias"

AMBITO_RIESGO_URL = "https://mercados.ambito.com//riesgo-pais/variacion"
ARG_DATOS_BASES = [
    "https://api.argentinadatos.com/v1/finanzas/indices",
    "https://argentinadatos.com/v1/finanzas/indices",
]

ARG_DATOS_RIESGO_URL = "https://api.argentinadatos.com/v1/finanzas/indices/riesgo-pais"
DOLARITO_RIESGO_API = "https://api.dolarito.ar/api/frontend/indices/riesgo-pais"
DOLARITO_RIESGO_HTML = "https://www.dolarito.ar/indices/riesgo-pais"
CRIPTOYA_RIESGO_URL = "https://criptoya.com/charts/riesgo-pais"

LAMACRO_RESERVAS_URL = "https://www.lamacro.ar/variables/1"

YF_URLS = [
    "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
    "https://query2.finance.yahoo.com/v8/finance/chart/{symbol}",
]
YF_HEADERS = {"User-Agent": "Mozilla/5.0"}
REQ_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
    "Accept": "text/html,application/json;q=0.9,*/*;q=0.8",
    "Accept-Language": "es-AR,es;q=0.9,en;q=0.8",
    "Connection": "keep-alive",
    "Cache-Control": "no-cache",
}
HTTPX_CLIENT_KEY = "httpx_client"

# ============================ LISTADOS ============================

ACCIONES_BA = [
    "GGAL.BA","YPFD.BA","PAMP.BA","CEPU.BA","ALUA.BA","TXAR.BA","TGSU2.BA","BYMA.BA","SUPV.BA","BMA.BA",
    "EDN.BA","CRES.BA","COME.BA","VALO.BA","TGNO4.BA","TRAN.BA","LOMA.BA","HARG.BA","CVH.BA","TECO2.BA"
]
CEDEARS_BA = [
    "AAPL.BA","MSFT.BA","NVDA.BA","AMZN.BA","GOOGL.BA","TSLA.BA","META.BA","JNJ.BA","KO.BA","NFLX.BA",
    "BRKB.BA","PG.BA","DISN.BA","AMD.BA","INTC.BA","NKE.BA","V.BA","MA.BA","PFE.BA","XOM.BA"
]
CEDEAR_UNDERLYING_OVERRIDES = {
    "BRKB.BA": "BRK-B",
    "DISN.BA": "DIS",
}
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

BINANCE_EXCHANGE_INFO_URLS = [
    "https://data-api.binance.vision/api/v3/exchangeInfo",
    "https://api.binance.com/api/v3/exchangeInfo",
]
BINANCE_TICKER_PRICE_URLS = [
    "https://data-api.binance.vision/api/v3/ticker/price",
    "https://api.binance.com/api/v3/ticker/price",
    "https://api1.binance.com/api/v3/ticker/price",
    "https://api2.binance.com/api/v3/ticker/price",
    "https://api3.binance.com/api/v3/ticker/price",
    "https://api-gcp.binance.com/api/v3/ticker/price",
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
SCREENERMATIC_BONDS_URL = "https://www.screenermatic.com/bondsdescriptive.php"


CUSTOM_CRYPTO_ENTRIES: Dict[str, Dict[str, Any]] = {
    "VRAUSDT": {
        "symbol": "VRAUSDT",
        "base": "VRA",
        "quote": "USDT",
        "display": "Verasity",
        "aliases": ["VERACITYUSDT"],
    },
}


FCI_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "fci_data.json")
_fci_series_cache: Optional[Dict[str, List[Tuple[int, float]]]] = None
FONDOSONLINE_FUNDS_URL = "https://fondosonline.com/Operations/Funds/GetFundsProducts"


def _load_fci_series() -> Dict[str, List[Tuple[int, float]]]:
    global _fci_series_cache
    if _fci_series_cache is not None:
        return _fci_series_cache

    series: Dict[str, List[Tuple[int, float]]] = {}
    try:
        with open(FCI_DATA_PATH, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except FileNotFoundError:
        log.warning("No se encontró archivo de series FCI en %s", FCI_DATA_PATH)
        _fci_series_cache = {}
        return _fci_series_cache
    except Exception as exc:
        log.warning("Error cargando series FCI (%s): %s", FCI_DATA_PATH, exc)
        _fci_series_cache = {}
        return _fci_series_cache

    for symbol, rows in raw.items():
        points: List[Tuple[int, float]] = []
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, (list, tuple)) or len(row) < 2:
                    continue
                date_raw, value_raw = row[0], row[1]
                try:
                    if isinstance(date_raw, (int, float)):
                        dt = datetime.fromtimestamp(float(date_raw), tz=TZ)
                    else:
                        dt = datetime.fromisoformat(str(date_raw))
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=TZ)
                except Exception:
                    continue
                try:
                    val = float(value_raw)
                except (TypeError, ValueError):
                    continue
                ts = int(datetime.combine(dt.date(), dtime(17, 0), tzinfo=TZ).timestamp())
                points.append((ts, val))
        if points:
            points.sort(key=lambda x: x[0])
            series[symbol] = points

    _fci_series_cache = series
    return _fci_series_cache


def _fci_metrics_from_series(symbol: str) -> Dict[str, Optional[float]]:
    base = {
        "6m": None,
        "3m": None,
        "1m": None,
        "last_ts": None,
        "vol_ann": None,
        "dd6m": None,
        "hi52": None,
        "slope50": None,
        "trend_flag": None,
        "last_px": None,
        "prev_px": None,
        "last_chg": None,
        "currency": "USD" if "USD" in symbol.upper() else "ARS",
    }

    series = _load_fci_series().get(symbol)
    if not series:
        return base

    last_ts, last_val = series[-1]
    base["last_ts"] = last_ts
    base["last_px"] = float(last_val)

    prev_val: Optional[float] = None
    for ts, value in reversed(series[:-1]):
        if value is not None:
            prev_val = float(value)
            break
    if prev_val is not None and prev_val > 0:
        base["prev_px"] = prev_val
        base["last_chg"] = (last_val / prev_val - 1.0) * 100.0

    window_map = {f"{months}m": WINDOW_DAYS[months] for months in WINDOW_MONTHS}
    for label in ("6m", "3m", "1m"):
        window = window_map[label]
        if len(series) >= window:
            ref = float(series[-window][1])
        else:
            ref = None
        if ref and ref > 0:
            base[label] = (last_val / ref - 1.0) * 100.0

    closes = [float(val) for _, val in series]
    if len(closes) >= 2:
        rets = []
        for i in range(1, len(closes)):
            if closes[i - 1] > 0:
                rets.append(closes[i] / closes[i - 1] - 1.0)
        if len(rets) >= 10:
            window_sample = rets[-60:] if len(rets) >= 60 else list(rets)
            mu = sum(window_sample) / len(window_sample)
            if len(window_sample) > 1:
                var = sum((r - mu) ** 2 for r in window_sample) / (len(window_sample) - 1)
                base["vol_ann"] = math.sqrt(max(var, 0.0)) * math.sqrt(252) * 100.0

        day = 24 * 3600
        look_back_ts = last_ts - 180 * day
        peak = closes[0]
        dd_min = 0.0
        for ts, value in series:
            if ts < look_back_ts:
                continue
            if value > peak:
                peak = value
            drawdown = value / peak - 1.0
            if drawdown < dd_min:
                dd_min = drawdown
        base["dd6m"] = abs(dd_min) * 100.0 if dd_min < 0 else 0.0

        max_val = max(closes)
        if max_val:
            base["hi52"] = (last_val / max_val - 1.0) * 100.0

        window = 50 if len(closes) >= 50 else len(closes)
        if window >= 5:
            sma = sum(closes[-window:]) / window
            prev_window = closes[-window - 5 : -5] if len(closes) >= window + 5 else closes[:-5]
            if prev_window:
                sma_prev = sum(prev_window) / len(prev_window)
                if sma_prev:
                    base["slope50"] = (sma / sma_prev - 1.0) * 100.0

        long_window = 100 if len(closes) >= 100 else len(closes)
        if long_window >= 10:
            sma_long = sum(closes[-long_window:]) / long_window
            base["trend_flag"] = 1.0 if last_val > sma_long else (-1.0 if last_val < sma_long else 0.0)

    return base


FCI_KEYWORDS = {
    "FCI-MoneyMarket": ["money market", "cash management", "liquidez", "money"],
    "FCI-BonosUSD": ["dólar", "usd", "dolar"],
    "FCI-AccionesArg": ["acciones argentinas"],
    "FCI-Corporativos": ["corporativo", "corporativa"],
    "FCI-Liquidez": ["liquidez"],
    "FCI-Balanceado": ["balanceado"],
    "FCI-RentaMixta": ["renta mixta"],
    "FCI-RealEstate": ["real estate", "inmobiliario"],
    "FCI-Commodity": ["commodity"],
    "FCI-Tech": ["tecnología", "technology", "tech"],
    "FCI-BonosCER": ["cer", "uva"],
    "FCI-DurationCorta": ["corto", "short"],
    "FCI-DurationMedia": ["mediano", "medio"],
    "FCI-DurationLarga": ["largo"],
    "FCI-HighYield": ["high yield", "alto retorno"],
    "FCI-BlueChips": ["blue chip", "bluechips"],
    "FCI-Growth": ["growth", "crecimiento"],
    "FCI-Value": ["value", "valor"],
    "FCI-Latam": ["latam", "latino"],
    "FCI-Global": ["global"],
}


def _matches_keyword(record: Dict[str, Any], keyword: str) -> bool:
    haystack = " ".join(
        str(part or "")
        for part in (
            record.get("fundName"),
            record.get("fundStrategy"),
            record.get("fundFocus"),
        )
    ).lower()
    return keyword.lower() in haystack


def _pick_fci_record(symbol: str, records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    keywords = FCI_KEYWORDS.get(symbol, [])
    for kw in keywords:
        for rec in records:
            if _matches_keyword(rec, kw):
                return rec

    currency_pref = "USD" if "USD" in symbol.upper() else "ARS"
    for rec in records:
        if str(rec.get("fundCurrency", "")).upper() == currency_pref:
            return rec
    return records[0] if records else None


def _fci_metrics_from_record(record: Dict[str, Any]) -> Dict[str, Optional[float]]:
    base = {
        "6m": None,
        "3m": None,
        "1m": None,
        "last_ts": None,
        "vol_ann": None,
        "dd6m": None,
        "hi52": None,
        "slope50": None,
        "trend_flag": None,
        "last_px": None,
        "prev_px": None,
        "last_chg": None,
    }

    try:
        base["last_px"] = float(record.get("lastPrice"))
    except (TypeError, ValueError):
        base["last_px"] = None

    try:
        base["last_chg"] = float(record.get("dayPercent"))
    except (TypeError, ValueError):
        base["last_chg"] = None

    try:
        base["1m"] = float(record.get("monthPercent"))
    except (TypeError, ValueError):
        base["1m"] = None

    try:
        base["6m"] = float(record.get("yearPercent"))
    except (TypeError, ValueError):
        base["6m"] = None

    last_ts_raw = record.get("lastPriceDate")
    if isinstance(last_ts_raw, str):
        try:
            dt = datetime.fromisoformat(last_ts_raw.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=TZ)
            base["last_ts"] = int(dt.timestamp())
        except Exception:
            pass

    base["currency"] = "USD" if str(record.get("fundCurrency", "")).upper() == "USD" else "ARS"
    return base


async def _fondosonline_records(session: ClientSession) -> List[Dict[str, Any]]:
    cache_key = "fondosonline:funds:records"
    cached = SHORT_CACHE.get(cache_key)
    if isinstance(cached, list):
        return cached  # type: ignore[return-value]

    payload = await fetch_json(
        session,
        FONDOSONLINE_FUNDS_URL,
        params={"PageSize": 500, "sortColumn": "FundName", "isAscending": True},
        timeout=ClientTimeout(total=12),
        source=urlparse(FONDOSONLINE_FUNDS_URL).netloc,
    )
    records = payload.get("records") if isinstance(payload, dict) else None
    if isinstance(records, list):
        SHORT_CACHE.set(cache_key, records, ttl=300)
        return records  # type: ignore[return-value]
    return []


async def _fci_metrics(session: ClientSession, symbol: str) -> Dict[str, Optional[float]]:
    records = await _fondosonline_records(session)
    if records:
        picked = _pick_fci_record(symbol, records)
        if picked:
            return _fci_metrics_from_record(picked)
    return _fci_metrics_from_series(symbol)


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

def cedear_underlying_symbol(symbol: str) -> Optional[str]:
    if not symbol:
        return None
    key = symbol.upper()
    if not key.endswith(".BA"):
        return None
    if key in CEDEAR_UNDERLYING_OVERRIDES:
        return CEDEAR_UNDERLYING_OVERRIDES[key]
    return key[:-3]

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


def metric_last_price(metrics: Dict[str, Any]) -> Optional[float]:
    """Return the last available native price contained in a metrics dict.

    Prefers explicit ``last_px`` values, but can infer the value from
    ``prev_px`` and ``last_chg`` (variation %) when intraday quotes are
    unavailable. As a last resort it falls back to ``prev_px`` alone so the
    instrument still contributes to portfolio valuations with the most recent
    close on record.
    """

    if not metrics:
        return None

    last_raw = metrics.get("last_px")
    if last_raw is not None:
        try:
            return float(last_raw)
        except (TypeError, ValueError):
            pass

    prev_raw = metrics.get("prev_px")
    chg_raw = metrics.get("last_chg")

    if prev_raw is not None and chg_raw is not None:
        try:
            prev_val = float(prev_raw)
            chg_val = float(chg_raw)
            return prev_val * (1.0 + chg_val / 100.0)
        except (TypeError, ValueError):
            pass

    if prev_raw is not None:
        try:
            return float(prev_raw)
        except (TypeError, ValueError):
            pass

    return None

# ============================ LOGGING ============================

log = logging.getLogger(__name__)


def instrument_command(name: str, handler: Callable[..., Awaitable[Any]]):
    async def _wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        metrics.increment(f"command.{name}.total")
        started = time()
        try:
            return await handler(update, context)
        except Exception:
            metrics.increment(f"command.{name}.errors")
            raise
        finally:
            metrics.observe_latency_ms(f"command.{name}.latency_ms", (time() - started) * 1000)

    return _wrapper


def _record_http_metrics(host: str, duration_ms: float, *, success: bool, timeout: bool = False) -> None:
    base_key = f"http.{host}" if host else "http.unknown"
    metrics.observe_latency_ms(f"{base_key}.latency_ms", duration_ms)
    metrics.increment(f"{base_key}.success" if success else f"{base_key}.errors")
    if timeout:
        metrics.increment(f"{base_key}.timeouts")


# ============================ ERRORES ============================

async def handle_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    error = context.error

    if isinstance(error, asyncio.CancelledError):
        return

    try:
        update_repr = repr(update)
    except Exception:
        update_repr = "<update no serializable>"

    log.error("Excepción no manejada (update=%s)", update_repr, exc_info=error)

    try:
        chat = getattr(update, "effective_chat", None)
        if chat:
            await context.bot.send_message(
                chat_id=chat.id,
                text="Ocurrió un error inesperado. Intentalo de nuevo en unos minutos.",
            )
    except Exception as notify_error:
        log.debug("No se pudo notificar el error al chat: %s", notify_error)

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
            log.error(
                "STATE_PATH no escribible, usando fallback.",
                extra={"event": "persistence_failure", "candidate": candidate, "fallback": fallback},
            )
            return fallback
        except Exception as e:
            log.error(
                "No puedo escribir estado",
                extra={"event": "persistence_failure", "candidate": candidate, "error": str(e)},
            )
            return fallback


def _ensure_state_path() -> Optional[str]:
    try:
        return _writable_path(STATE_PATH)
    except Exception:
        return None

USE_UPSTASH_REST = config.upstash.use_rest
USE_UPSTASH_REDIS = config.upstash.use_redis
USE_UPSTASH = USE_UPSTASH_REST or USE_UPSTASH_REDIS
STATE_PATH = ensure_writable_path(str(config.state_path), log)
STATE_STORE: StateStore
FALLBACK_STATE_STORE: Optional[StateStore] = None

if USE_UPSTASH_REST or USE_UPSTASH_REDIS:
    STATE_STORE = RedisStore(
        state_key=UPSTASH_STATE_KEY,
        rest_url=UPSTASH_URL if USE_UPSTASH_REST else "",
        rest_token=UPSTASH_TOKEN if USE_UPSTASH_REST else "",
        redis_url=UPSTASH_REDIS_URL if USE_UPSTASH_REDIS else "",
    )
    FALLBACK_STATE_STORE = JsonFileStore(STATE_PATH)
else:
    STATE_STORE = JsonFileStore(STATE_PATH)

ALERTS: Dict[int, List[Dict[str, Any]]] = {}
SUBS: Dict[int, Dict[str, Any]] = {}
PF: Dict[int, Dict[str, Any]] = {}
ALERT_USAGE: Dict[int, Dict[str, Dict[str, Any]]] = {}
PROJECTION_RECORDS: List[Dict[str, Any]] = []
PROJECTION_BATCHES: List[Dict[str, Any]] = []
NEWS_HISTORY: List[Tuple[str, float]] = []
NEWS_CACHE: Dict[str, Any] = {"date": "", "items": []}
RIESGO_CACHE: Dict[str, Any] = {}
RESERVAS_CACHE: Dict[str, Any] = {}
DOLAR_CACHE: Dict[str, Any] = {}
SHORT_CACHE = ShortCache(default_ttl=45, redis_url=UPSTASH_REDIS_URL, namespace="bot-econ:short")
CMD_THROTTLER = RateLimiter(redis_url=UPSTASH_REDIS_URL, namespace="bot-econ:throttle")
PROJ_HISTORY: List[Dict[str, Any]] = []
PROJ_CALIBRATION: Dict[str, Dict[str, float]] = {}
PROJ_HORIZON_DAYS = {"3m": 90, "6m": 180}
PROJ_HISTORY_TOLERANCE_DAYS = WINDOW_DAYS[1]
PROJ_HISTORY_MAX_AGE_DAYS = 400
PROJ_HISTORY_MAX_ENTRIES = 2000
PROJ_CALIBRATION_MIN_POINTS = 25


def invalidate_rankings_cache() -> None:
    SHORT_CACHE.invalidate_prefix("rankings:")


def invalidate_alerts_cache() -> None:
    SHORT_CACHE.invalidate_prefix("alerts:")


def is_throttled(command: str, chat_id: Optional[int], user_id: Optional[int], ttl: int = 45) -> bool:
    buckets = []
    if chat_id is not None:
        buckets.append(f"{command}:chat:{chat_id}")
    if user_id is not None:
        buckets.append(f"{command}:user:{user_id}")
    for bucket in buckets:
        if CMD_THROTTLER.hit(bucket, ttl):
            return True
    return False


async def load_state():
    global ALERTS, SUBS, PF, ALERT_USAGE, NEWS_HISTORY, NEWS_CACHE, RIESGO_CACHE, RESERVAS_CACHE, DOLAR_CACHE
    global PROJ_HISTORY, PROJ_CALIBRATION
    data: Optional[Dict[str, Any]] = None
    try:
        data = await STATE_STORE.load()
    except Exception as exc:
        log.error(
            "No pude leer estado del store principal",
            extra={"event": "persistence_failure", "error": str(exc)},
        )

    if data is None and FALLBACK_STATE_STORE is not None:
        try:
            data = await FALLBACK_STATE_STORE.load()
        except Exception as exc:
            log.error(
                "No pude leer estado del store de respaldo",
                extra={"event": "persistence_failure", "error": str(exc)},
            )

    if data is None:
        path = _ensure_state_path()
        if path:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                log.error(
                    "No pude leer estado local",
                    extra={"event": "persistence_failure", "path": path, "error": str(e)},
                )
                data = None

    version = CURRENT_STATE_VERSION
    if data:
        data = deserialize_state_payload(data)
        version = data.get("version", CURRENT_STATE_VERSION)
        ALERTS = {int(k): v for k, v in data.get("alerts", {}).items()}
        SUBS = {int(k): v for k, v in data.get("subs", {}).items()}
        PF = {int(k): v for k, v in data.get("pf", {}).items()}
        ALERT_USAGE = {int(k): v for k, v in data.get("alert_usage", {}).items()}
        projection_records = data.get("projection_records", [])
        projection_batches = data.get("projection_batches", [])
        PROJECTION_RECORDS = projection_records if isinstance(projection_records, list) else []
        PROJECTION_BATCHES = projection_batches if isinstance(projection_batches, list) else []

        raw_history = data.get("news_history", [])
        if isinstance(raw_history, list):
            NEWS_HISTORY = []
            for item in raw_history:
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    continue
                stem, ts = item[0], item[1]
                try:
                    stem_str = str(stem)
                    ts_val = float(ts)
                except Exception:
                    continue
                NEWS_HISTORY.append((stem_str, ts_val))

        cache_date = data.get("news_cache_date")
        cache_items = data.get("news_cache_items")
        if isinstance(cache_items, list) and isinstance(cache_date, str):
            NEWS_CACHE = {"date": cache_date, "items": cache_items}
        else:
            NEWS_CACHE = {"date": "", "items": []}

        cache_riesgo = data.get("riesgo_cache")
        if isinstance(cache_riesgo, dict) and cache_riesgo.get("val") is not None:
            try:
                cache_val = int(cache_riesgo.get("val"))
                cache_fecha = (
                    str(cache_riesgo.get("fecha")) if cache_riesgo.get("fecha") is not None else None
                )
                cache_var = (
                    float(cache_riesgo.get("variation")) if cache_riesgo.get("variation") is not None else None
                )
                cache_ts_raw = cache_riesgo.get("updated_at")
                cache_ts = float(cache_ts_raw) if cache_ts_raw is not None else None
                RIESGO_CACHE = {
                    "val": cache_val,
                    "fecha": cache_fecha,
                    "variation": cache_var,
                    "updated_at": cache_ts,
                }
            except Exception:
                RIESGO_CACHE = {}
        else:
            RIESGO_CACHE = {}

        cache_reservas = data.get("reservas_cache")
        if isinstance(cache_reservas, dict) and cache_reservas.get("val") is not None:
            try:
                cache_val = float(cache_reservas.get("val"))
                cache_prev = cache_reservas.get("prev_val")
                cache_prev_val = float(cache_prev) if cache_prev is not None else None
                cache_fecha = (
                    str(cache_reservas.get("fecha")) if cache_reservas.get("fecha") is not None else None
                )
                cache_ts_raw = cache_reservas.get("updated_at")
                cache_ts = float(cache_ts_raw) if cache_ts_raw is not None else None
                RESERVAS_CACHE = {
                    "val": cache_val,
                    "prev_val": cache_prev_val,
                    "fecha": cache_fecha,
                    "updated_at": cache_ts,
                }
            except Exception:
                RESERVAS_CACHE = {}
        else:
            RESERVAS_CACHE = {}

        cache_dolar = data.get("dolar_cache")
        if isinstance(cache_dolar, dict):
            cached_data = cache_dolar.get("data") if isinstance(cache_dolar.get("data"), dict) else None
            ts_raw = cache_dolar.get("updated_at")
            try:
                ts_val = float(ts_raw) if ts_raw is not None else None
            except Exception:
                ts_val = None

            if cached_data and any(
                isinstance(v, dict) and (v.get("compra") is not None or v.get("venta") is not None)
                for v in cached_data.values()
            ):
                DOLAR_CACHE = {"data": cached_data, "updated_at": ts_val}
            else:
                DOLAR_CACHE = {}
        else:
            DOLAR_CACHE = {}

        raw_proj_history = data.get("proj_history")
        PROJ_HISTORY = _clean_proj_history(raw_proj_history)

        raw_proj_cal = data.get("proj_calibration")
        PROJ_CALIBRATION = _clean_proj_calibration(raw_proj_cal)

        log.info(
            "State loaded (v%s). alerts=%d subs=%d pf=%d",
            version,
            sum(len(v) for v in ALERTS.values()),
            len(SUBS),
            len(PF),
        )
    else:
        log.info("No previous state found (v%s).", version)


def _prune_news_history(now: Optional[float] = None, window_hours: int = 72) -> None:
    global NEWS_HISTORY
    ts_now = now if now is not None else time()
    cutoff = ts_now - (window_hours * 3600)
    NEWS_HISTORY = [(stem, ts) for stem, ts in NEWS_HISTORY if ts >= cutoff]


def _clean_proj_history(raw: Any) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    if not isinstance(raw, list):
        return cleaned
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        symbol = entry.get("symbol")
        horizon = entry.get("horizon")
        raw_val = entry.get("raw")
        ts = entry.get("ts")
        last_px = entry.get("last_px")
        if not isinstance(symbol, str) or horizon not in PROJ_HORIZON_DAYS:
            continue
        try:
            raw_f = float(raw_val)
            ts_f = float(ts)
            last_px_f = float(last_px)
        except Exception:
            continue
        cleaned.append(
            {
                "symbol": symbol,
                "horizon": horizon,
                "raw": raw_f,
                "ts": ts_f,
                "last_px": last_px_f,
            }
        )
    return cleaned


def _clean_proj_calibration(raw: Any) -> Dict[str, Dict[str, float]]:
    cleaned: Dict[str, Dict[str, float]] = {}
    if not isinstance(raw, dict):
        return cleaned
    for horizon, entry in raw.items():
        if horizon not in PROJ_HORIZON_DAYS or not isinstance(entry, dict):
            continue
        try:
            a = float(entry.get("a"))
            b = float(entry.get("b"))
        except Exception:
            continue
        cleaned[horizon] = {
            "a": a,
            "b": b,
            "n": float(entry.get("n")) if entry.get("n") is not None else 0.0,
            "updated_at": float(entry.get("updated_at")) if entry.get("updated_at") is not None else 0.0,
        }
    return cleaned


def _prune_proj_history(now_ts: Optional[float] = None) -> None:
    global PROJ_HISTORY
    ts_now = now_ts if now_ts is not None else time()
    cutoff = ts_now - (PROJ_HISTORY_MAX_AGE_DAYS * 86400)
    PROJ_HISTORY = [e for e in PROJ_HISTORY if e.get("ts") is not None and e["ts"] >= cutoff]
    if len(PROJ_HISTORY) > PROJ_HISTORY_MAX_ENTRIES:
        PROJ_HISTORY = sorted(PROJ_HISTORY, key=lambda e: e.get("ts", 0.0), reverse=True)[
            :PROJ_HISTORY_MAX_ENTRIES
        ]


def register_projection_history(symbol: str, horizon: str, raw: float, metrics: Dict[str, Optional[float]]) -> None:
    if horizon not in PROJ_HORIZON_DAYS:
        return
    if not symbol:
        return
    last_px = metrics.get("last_px")
    last_ts = metrics.get("last_ts")
    if last_px is None or last_ts is None:
        return
    try:
        entry = {
            "symbol": str(symbol),
            "horizon": horizon,
            "raw": float(raw),
            "ts": float(last_ts),
            "last_px": float(last_px),
        }
    except Exception:
        return
    PROJ_HISTORY.append(entry)


async def recalibrate_projection_coeffs() -> None:
    if not PROJ_HISTORY:
        return
    now_ts = time()
    _prune_proj_history(now_ts)
    by_horizon: Dict[str, List[Dict[str, Any]]] = {h: [] for h in PROJ_HORIZON_DAYS}
    for entry in PROJ_HISTORY:
        horizon = entry.get("horizon")
        if horizon in by_horizon:
            by_horizon[horizon].append(entry)

    symbols: Set[str] = set()
    for entries in by_horizon.values():
        for entry in entries:
            symbol = entry.get("symbol")
            if isinstance(symbol, str):
                symbols.add(symbol)

    if not symbols:
        return

    async with ClientSession() as session:
        metrics_map, _ = await metrics_for_symbols(session, sorted(symbols))

    updated = False
    for horizon, entries in by_horizon.items():
        horizon_days = PROJ_HORIZON_DAYS[horizon]
        pairs: List[Tuple[float, float]] = []
        for entry in entries:
            symbol = entry.get("symbol")
            if symbol not in metrics_map:
                continue
            metrics = metrics_map[symbol]
            now_px = metrics.get("last_px")
            now_ts_entry = metrics.get("last_ts")
            if now_px is None or now_ts_entry is None:
                continue
            entry_ts = entry.get("ts")
            entry_px = entry.get("last_px")
            if entry_ts is None or entry_px is None:
                continue
            try:
                age_days = (float(now_ts_entry) - float(entry_ts)) / 86400.0
            except Exception:
                continue
            if age_days < (horizon_days - PROJ_HISTORY_TOLERANCE_DAYS):
                continue
            if age_days > (horizon_days + PROJ_HISTORY_TOLERANCE_DAYS):
                continue
            try:
                actual = (float(now_px) / float(entry_px) - 1.0) * 100.0
                raw_val = float(entry.get("raw"))
            except Exception:
                continue
            pairs.append((raw_val, actual))

        if len(pairs) < PROJ_CALIBRATION_MIN_POINTS:
            continue
        mean_x = sum(p[0] for p in pairs) / len(pairs)
        mean_y = sum(p[1] for p in pairs) / len(pairs)
        var_x = sum((p[0] - mean_x) ** 2 for p in pairs)
        if var_x <= 1e-6:
            b = 1.0
            a = 0.0
        else:
            cov = sum((p[0] - mean_x) * (p[1] - mean_y) for p in pairs)
            b = cov / var_x
            a = mean_y - b * mean_x
        PROJ_CALIBRATION[horizon] = {
            "a": a,
            "b": b,
            "n": float(len(pairs)),
            "updated_at": now_ts,
        }
        updated = True

    if updated:
        await save_state()


async def save_state():
    _prune_news_history()
    _prune_proj_history()
    payload = serialize_state_payload(
        {
            "alerts": ALERTS,
            "subs": SUBS,
            "pf": PF,
            "alert_usage": ALERT_USAGE,
            "projection_records": PROJECTION_RECORDS,
            "projection_batches": PROJECTION_BATCHES,
            "news_history": NEWS_HISTORY,
            "news_cache_date": NEWS_CACHE.get("date", ""),
            "news_cache_items": NEWS_CACHE.get("items", []),
            "riesgo_cache": RIESGO_CACHE,
            "reservas_cache": RESERVAS_CACHE,
            "dolar_cache": DOLAR_CACHE,
            "proj_history": PROJ_HISTORY,
            "proj_calibration": PROJ_CALIBRATION,
        }
    )
    stored = await STATE_STORE.save(payload)
    if stored or FALLBACK_STATE_STORE is None:
        return
    path = _ensure_state_path()
    if not path:
        log.error("No path available to persist state locally", extra={"event": "persistence_failure"})
        return
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
    except Exception as e:
        log.error(
            "save_state error",
            extra={"event": "persistence_failure", "path": path, "error": str(e)},
        )

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
    started = time()
    host = urlparse(url).netloc
    timeout = kwargs.pop("timeout", ClientTimeout(total=12))
    headers = kwargs.pop("headers", {})
    source = kwargs.pop("source", None)
    http_timeout = timeout.total if isinstance(timeout, ClientTimeout) else timeout
    try:
        return await http_service.get_json(
            url, source=source, headers={**REQ_HEADERS, **headers}, timeout=http_timeout, **kwargs
        )
    except SourceSuspendedError as exc:
        _record_http_metrics(host, (time() - started) * 1000, success=False)
        log.warning(
            "source_suspended source=%s resume_at=%s url=%s",
            exc.source,
            exc.resume_at,
            url,
        )
        return None
    except Exception as exc:
        _record_http_metrics(host, (time() - started) * 1000, success=False)
        log.warning("fetch_json http_service error %s: %s", url, exc)
    try:
        async with session.get(
            url, timeout=timeout, headers={**REQ_HEADERS, **headers}, **kwargs
        ) as resp:
            if 200 <= resp.status < 300:
                payload = await resp.json(content_type=None)
                _record_http_metrics(host, (time() - started) * 1000, success=True)
                return payload
            log.warning("GET %s -> %s", url, resp.status)
    except asyncio.TimeoutError:
        duration_ms = (time() - started) * 1000
        _record_http_metrics(host, duration_ms, success=False, timeout=True)
        log.error("fetch_json timeout", extra={"url": url, "event": "api_timeout", "duration_ms": duration_ms})
        return None
    except Exception as e:
        _record_http_metrics(host, (time() - started) * 1000, success=False)
        log.warning("fetch_json error %s: %s", url, e)
        return None
    _record_http_metrics(host, (time() - started) * 1000, success=False)
    return None


def build_httpx_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=15,
        headers=REQ_HEADERS,
        follow_redirects=True,
        verify=certifi.where(),
    )


def get_httpx_client(bot_data: Dict[str, Any]) -> Optional[httpx.AsyncClient]:
    client = bot_data.get(HTTPX_CLIENT_KEY)
    return client if isinstance(client, httpx.AsyncClient) else None


async def fetch_json_httpx(client: httpx.AsyncClient, url: str, **kwargs) -> Optional[Dict[str, Any]]:
    started = time()
    host = urlparse(url).netloc
    try:
        timeout = kwargs.pop("timeout", None)
        headers = kwargs.pop("headers", {})
        request_kwargs = kwargs
        if timeout is not None:
            request_kwargs["timeout"] = timeout
        if headers:
            request_kwargs["headers"] = {**dict(client.headers), **headers}
        resp = await client.get(url, **request_kwargs)
        if 200 <= resp.status_code < 300:
            payload = resp.json()
            _record_http_metrics(host, (time() - started) * 1000, success=True)
            return payload
        log.warning("httpx GET %s -> %s", url, resp.status_code)
    except httpx.TimeoutException:
        duration_ms = (time() - started) * 1000
        _record_http_metrics(host, duration_ms, success=False, timeout=True)
        log.error("fetch_json_httpx timeout", extra={"url": url, "event": "api_timeout", "duration_ms": duration_ms})
        return None
    except Exception as e:
        _record_http_metrics(host, (time() - started) * 1000, success=False)
        log.warning("fetch_json_httpx error %s: %s", url, e)
        return None
    _record_http_metrics(host, (time() - started) * 1000, success=False)
    return None

async def fetch_text(session: ClientSession, url: str, **kwargs) -> Optional[str]:
    started = time()
    host = urlparse(url).netloc
    timeout = kwargs.pop("timeout", ClientTimeout(total=15))
    headers = kwargs.pop("headers", {})

    try:
        http_timeout = timeout.total if isinstance(timeout, ClientTimeout) else timeout
        return await http_service.get_text(
            url,
            headers={**REQ_HEADERS, **headers},
            timeout=http_timeout,
            **kwargs,
        )
    except SourceSuspendedError as exc:
        _record_http_metrics(host, (time() - started) * 1000, success=False)
        log.warning(
            "source_suspended source=%s resume_at=%s url=%s",
            exc.source,
            exc.resume_at,
            url,
        )
        return None
    try:
        async with session.get(url, timeout=timeout, headers={**REQ_HEADERS, **headers}, **kwargs) as resp:
            if resp.status == 200:
                body = await resp.text()
                _record_http_metrics(host, (time() - started) * 1000, success=True)
                return body
            log.warning("GET %s -> %s", url, resp.status)
    except asyncio.TimeoutError:
        duration_ms = (time() - started) * 1000
        _record_http_metrics(host, duration_ms, success=False, timeout=True)
        log.error("fetch_text timeout", extra={"url": url, "event": "api_timeout", "duration_ms": duration_ms})
        return None
    except Exception as e:
        _record_http_metrics(host, (time() - started) * 1000, success=False)
        log.warning("fetch_text error %s: %s", url, e)
        return None
    _record_http_metrics(host, (time() - started) * 1000, success=False)
    return None


async def get_binance_symbols(session: ClientSession) -> Dict[str, Dict[str, str]]:
    global _binance_symbols_cache, _binance_symbols_ts
    now = time()
    if _binance_symbols_cache and (now - _binance_symbols_ts) < 3600:
        return _binance_symbols_cache
    data = None
    for url in BINANCE_EXCHANGE_INFO_URLS:
        data = await fetch_json(session, url)
        if data:
            break
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


def _has_fx_data(rows: Dict[str, Dict[str, Any]]) -> bool:
    for row in rows.values():
        if not isinstance(row, dict):
            continue
        if row.get("compra") is not None or row.get("venta") is not None:
            return True
    return False


def _save_dolar_cache(payload: Dict[str, Dict[str, Any]]) -> None:
    global DOLAR_CACHE
    new_cache = {"data": payload, "updated_at": time()}
    if DOLAR_CACHE != new_cache:
        DOLAR_CACHE = new_cache
        try:
            asyncio.create_task(save_state())
        except Exception:
            pass

async def get_dolares(session: ClientSession) -> Dict[str, Dict[str, Any]]:
    global DOLAR_CACHE
    data: Dict[str, Dict[str, Any]] = {}
    variations: Dict[str, float] = {}
    cache_key = "quotes:dolares"
    cached_short = SHORT_CACHE.get(cache_key)
    if isinstance(cached_short, dict) and _has_fx_data(cached_short):
        return cached_short

    def _safe_float(val: Any) -> Optional[float]:
        try:
            num = float(val) if val is not None else None
            if num is None or not math.isfinite(num):
                return None
            return num
        except Exception:
            return None

    def _pick_variation(block: Any) -> Optional[float]:
        if isinstance(block, dict):
            if "variation" in block:
                v = _safe_float(block.get("variation"))
                if v is not None:
                    return v
            for key in ["24hs", "ccb", "usdt", "usdc", "al30", "gd30", "bpo27", "letras", "ci"]:
                v = _pick_variation(block.get(key))
                if v is not None:
                    return v
        return None

    def _safe(block: Dict[str, Any]):
        if not isinstance(block, dict):
            return (None, None, None)
        c = block.get("compra") or block.get("buy") or block.get("bid")
        v = block.get("venta") or block.get("sell") or block.get("ask")
        if v is None and "price" in block:
            v = block.get("price")
            if c is None:
                c = v
        var = _pick_variation(block)
        try:
            return (
                float(c) if c is not None else None,
                float(v) if v is not None else None,
                float(var) if var is not None else None,
            )
        except Exception:
            return (None, None, var if isinstance(var, float) else None)

    if not http_service.is_suspended("criptoya"):
        cj = await fetch_json(session, CRYPTOYA_DOLAR_URL, source="criptoya")
        if cj:
            for k in ["oficial", "mayorista", "blue", "mep", "ccl", "cripto", "tarjeta", "ahorro"]:
                c, v, var = _safe(cj.get(k, {}))
                if c is not None or v is not None:
                    data[k] = {"compra": c, "venta": v, "fuente": "CriptoYa"}
                if var is not None:
                    variations[k] = var
        else:
            log.warning("fallback_to_dolarapi source=criptoya")
    else:
        log.info("source_skip_due_to_suspension source=criptoya")

    async def dolarapi(path: str, date: Optional[str] = None):
        url = f"{DOLARAPI_BASE}{path}"
        if date:
            url = f"{url}?fecha={date}"
        j = await fetch_json(session, url, source="dolarapi")
        if not j:
            return (None, None, None)
        c, v, fecha = j.get("compra"), j.get("venta"), j.get("fechaActualizacion") or j.get("fecha")
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

    today = datetime.now(TZ).date()
    prev_dates = [(today - timedelta(days=delta)).isoformat() for delta in range(1, 6)]

    for k, path in mapping.items():
        needs_fallback = (
            k not in data
            or (data[k].get("compra") is None and data[k].get("venta") is None)
            or not data[k].get("fecha")
        )
        async def _fetch_with_history() -> Tuple[Optional[float], Optional[float], Optional[str]]:
            c, v, fecha = await dolarapi(path)
            if c is not None or v is not None:
                return c, v, fecha

            for date_str in prev_dates:
                c_prev, v_prev, fecha_prev = await dolarapi(path, date_str)
                if c_prev is not None or v_prev is not None:
                    return c_prev, v_prev, fecha_prev

            return None, None, None

        if needs_fallback:
            c, v, fecha = await _fetch_with_history()
            if c is not None or v is not None:
                data[k] = {"compra": c, "venta": v, "fuente": "DolarAPI", "fecha": fecha}
            elif k in data and fecha:
                data[k]["fecha"] = fecha
        if k in data and k in variations:
            data[k]["variation"] = variations[k]

    def _current_val(row: Dict[str, Any]) -> Optional[float]:
        venta = row.get("venta")
        compra = row.get("compra")
        try:
            if venta is not None:
                return float(venta)
            if compra is not None:
                return float(compra)
        except Exception:
            return None
        return None

    async def _update_variation(key: str, path: str):
        row = data.get(key)
        if not row:
            return

        # Si ya tenemos variación (provista por CriptoYa u otra fuente), no la
        # pisamos con el cálculo basado en DolarAPI que hoy siempre devuelve el
        # mismo valor histórico y terminaba dejando la variación en 0 para
        # todos los tipos de cambio.
        if row.get("variation") is not None:
            return

        cur_val = _current_val(row)
        if cur_val is None:
            return

        for date_str in prev_dates:
            prev_c, prev_v, _ = await dolarapi(path, date_str)
            prev_val = prev_v if prev_v is not None else prev_c
            if prev_val is None or prev_val == 0:
                continue
            try:
                row["variation"] = ((cur_val - prev_val) / prev_val) * 100.0
                row["prev_fecha"] = date_str
                return
            except Exception:
                continue

    for k, path in mapping.items():
        await _update_variation(k, path)

    def _normalize_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        compra = _safe_float(row.get("compra"))
        venta = _safe_float(row.get("venta"))
        variation = _safe_float(row.get("variation"))
        fecha = row.get("fecha")

        if compra is None and venta is not None:
            compra = venta
        if venta is None and compra is not None:
            venta = compra
        if compra is None and venta is None:
            return None

        normalized = {
            "compra": compra,
            "venta": venta,
            "fuente": row.get("fuente"),
            "fecha": fecha,
        }
        if variation is not None:
            normalized["variation"] = variation
        return normalized

    cached_data = DOLAR_CACHE.get("data") if isinstance(DOLAR_CACHE, dict) else None
    merged: Dict[str, Dict[str, Any]] = {}
    keys = set(data.keys()) | (set(cached_data.keys()) if isinstance(cached_data, dict) else set())

    for k in keys:
        fresh = _normalize_row(data.get(k, {})) if data.get(k) else None
        cached = _normalize_row(cached_data.get(k, {})) if isinstance(cached_data, dict) else None

        if fresh is None and cached is None:
            continue
        if fresh is None:
            merged[k] = cached  # type: ignore[assignment]
            continue
        row = fresh.copy()
        if cached:
            if row.get("compra") is None and cached.get("compra") is not None:
                row["compra"] = cached["compra"]
            if row.get("venta") is None and cached.get("venta") is not None:
                row["venta"] = cached["venta"]
            if row.get("fecha") is None and cached.get("fecha") is not None:
                row["fecha"] = cached["fecha"]
            if row.get("variation") is None and cached.get("variation") is not None:
                row["variation"] = cached["variation"]
            if row.get("fuente") is None and cached.get("fuente") is not None:
                row["fuente"] = cached["fuente"]
        merged[k] = row

    # Tipos de cambio especiales (promedio bancos, Qatar) ya no se calculan
    # aquí para simplificar el panel principal.
    if _has_fx_data(merged):
        _save_dolar_cache(merged)
        SHORT_CACHE.set(cache_key, merged)
        return merged

    if isinstance(cached_data, dict) and _has_fx_data(cached_data):
        SHORT_CACHE.set(cache_key, cached_data)
        return cached_data

    SHORT_CACHE.set(cache_key, merged)
    return merged

FX_DOLARAPI_PATHS = {
    "mep": "/dolares/bolsa",
    "ccl": "/dolares/contadoconliqui",
}
FX_YAHOO_FALLBACK = {
    "mep": "USDARS=X",
    "ccl": "USDARS=X",
}

def _pick_fx_value(compra: Optional[float], venta: Optional[float]) -> Optional[float]:
    if venta is not None:
        return float(venta)
    if compra is not None:
        return float(compra)
    return None

async def _dolarapi_quote(
    session: ClientSession,
    path: str,
    date: Optional[str] = None,
) -> Optional[Tuple[Optional[float], Optional[float], Optional[str]]]:
    url = f"{DOLARAPI_BASE}{path}"
    if date:
        url = f"{url}?fecha={date}"
    j = await fetch_json(session, url, source="dolarapi")
    if not j:
        return None
    compra = j.get("compra")
    venta = j.get("venta")
    fecha = j.get("fechaActualizacion") or j.get("fecha")
    try:
        return (
            float(compra) if compra is not None else None,
            float(venta) if venta is not None else None,
            fecha,
        )
    except Exception:
        return None

async def _fx_value_at(
    session: ClientSession,
    fx_type: str,
    target_date: date,
    max_back: int = 7,
) -> Optional[float]:
    path = FX_DOLARAPI_PATHS.get(fx_type)
    if not path:
        return None
    for delta in range(max_back):
        date_str = (target_date - timedelta(days=delta)).isoformat()
        quote = await _dolarapi_quote(session, path, date_str)
        if not quote:
            continue
        compra, venta, _ = quote
        val = _pick_fx_value(compra, venta)
        if val is not None:
            return val
    return None

async def _fx_metrics_series(
    session: ClientSession,
    fx_type: str,
) -> Dict[str, Optional[float]]:
    out = {
        "6m": None,
        "3m": None,
        "1m": None,
        "last_ts": None,
        "last_px": None,
        "prev_px": None,
        "last_chg": None,
    }
    fx = await get_dolares(session)
    row = fx.get(fx_type.lower(), {}) if fx_type else {}
    compra = row.get("compra")
    venta = row.get("venta")
    cur_val = _pick_fx_value(compra, venta)
    if cur_val is None:
        return out

    out["last_px"] = cur_val
    out["last_ts"] = int(datetime.now(TZ).timestamp())
    try:
        if row.get("variation") is not None:
            out["last_chg"] = float(row.get("variation"))
    except Exception:
        pass

    today = datetime.now(TZ).date()
    horizons = [(30, "1m"), (90, "3m"), (180, "6m")]
    for days, key in horizons:
        prev_val = await _fx_value_at(session, fx_type, today - timedelta(days=days))
        if prev_val and prev_val > 0:
            out[key] = (cur_val / prev_val - 1.0) * 100.0
    if _fx_returns_look_empty(out):
        fallback_symbol = FX_YAHOO_FALLBACK.get(fx_type)
        if fallback_symbol:
            fallback = await _yf_metrics_1y(session, fallback_symbol)
            if fallback:
                for horizon in ("1m", "3m", "6m"):
                    if fallback.get(horizon) is not None:
                        out[horizon] = fallback.get(horizon)
                for key in ("last_ts", "last_px", "prev_px", "last_chg"):
                    if fallback.get(key) is not None:
                        out[key] = fallback.get(key)
    return out

def _fx_returns_look_empty(metrics: Dict[str, Optional[float]]) -> bool:
    values = [metrics.get("1m"), metrics.get("3m"), metrics.get("6m")]
    if not any(v is not None for v in values):
        return True
    non_zero = [v for v in values if v is not None and abs(float(v)) > 0.01]
    return len(non_zero) == 0

async def get_tc_value(session: ClientSession, tc_name: Optional[str]) -> Optional[float]:
    if not tc_name: return None
    fx = await get_dolares(session)
    row = fx.get(tc_name.lower(), {})
    v = row.get("venta")
    try: return float(v) if v is not None else None
    except: return None

async def get_riesgo_pais(
    session: ClientSession, httpx_client: Optional[httpx.AsyncClient] = None
) -> Optional[Tuple[int, Optional[str], Optional[float], bool]]:
    global RIESGO_CACHE
    val: Optional[float] = None
    fecha: Optional[str] = None
    variation: Optional[float] = None
    from_cache = False
    cache_key = "macro:riesgo"
    cached_short = SHORT_CACHE.get(cache_key)
    if isinstance(cached_short, (list, tuple)):
        cached_tuple = tuple(cached_short)
        if len(cached_tuple) >= 3:
            return cached_tuple  # type: ignore[return-value]

    value_keys = {"valor", "value", "ultimo", "last", "riesgo", "embi", "current", "close", "latest"}
    date_keys = {
        "fecha",
        "updatedat",
        "lastupdated",
        "last_update",
        "fechaactualizacion",
        "fechaactualizada",
        "date",
        "timestamp",
    }
    prev_keys = {"anterior", "previous", "prev", "valoranterior", "prev_value", "valorprevio", "previo"}
    variation_keys = {
        "variacion",
        "variation",
        "cambio",
        "cambioporcentual",
        "porcentaje",
        "porcentual",
        "delta",
        "change",
        "pctchange",
        "variaciondiaria",
    }

    def _save_riesgo_cache(v: int, f: Optional[str], var: Optional[float]) -> None:
        global RIESGO_CACHE
        new_cache = {
            "val": v,
            "fecha": f,
            "variation": var,
            "updated_at": time(),
        }
        if RIESGO_CACHE != new_cache:
            RIESGO_CACHE = new_cache
            try:
                asyncio.create_task(save_state())
            except Exception:
                pass

    def _safe_number(raw: Any) -> Optional[float]:
        try:
            if isinstance(raw, str):
                cleaned = raw.strip().replace("%", "").replace("+", "").replace(",", ".")
                if cleaned.startswith("–"):
                    cleaned = "-" + cleaned[1:]
                raw = cleaned
            num = float(raw)
            return num if math.isfinite(num) else None
        except (TypeError, ValueError):
            return None

    def _extract_field(data: Any, keys: Iterable[str]) -> Optional[Any]:
        normalized_keys = {re.sub(r"[^a-z0-9]", "", str(k).lower()) for k in keys}

        if isinstance(data, dict):
            for k, v in data.items():
                key_norm = re.sub(r"[^a-z0-9]", "", str(k).lower())
                if key_norm in normalized_keys and v is not None:
                    return v
                res = _extract_field(v, keys)
                if res is not None:
                    return res
        elif isinstance(data, list):
            for item in data:
                res = _extract_field(item, keys)
                if res is not None:
                    return res
        return None

    def _parse_json_payload(payload: Any) -> None:
        nonlocal val, fecha, variation
        parsed_val = _safe_number(_extract_field(payload, value_keys))
        if parsed_val is not None:
            val = parsed_val
        fetched_fecha = _extract_field(payload, date_keys)
        if fetched_fecha:
            fecha = str(fetched_fecha)
        prev_val = _safe_number(_extract_field(payload, prev_keys))
        if prev_val not in (None, 0) and val is not None and variation is None:
            try:
                variation = ((val - prev_val) / prev_val) * 100.0
            except Exception:
                variation = None
        direct_var = _safe_number(_extract_field(payload, variation_keys))
        if direct_var is not None:
            variation = direct_var

    async def _parse_argdatos() -> None:
        nonlocal val, fecha, variation
        data = await fetch_json_httpx(httpx_client, ARG_DATOS_RIESGO_URL) if httpx_client else None
        if not data:
            data = await fetch_json(session, ARG_DATOS_RIESGO_URL)
        series = None
        if isinstance(data, list):
            series = data
        elif isinstance(data, dict):
            # Some endpoints wrap the series
            for candidate_key in ("data", "serie", "results"):
                if isinstance(data.get(candidate_key), list):
                    series = data[candidate_key]
                    break
        if not series:
            return
        latest_item = None
        prev_item = None
        for item in reversed(series):
            cur_val = _safe_number(_extract_field(item, value_keys))
            if cur_val is None:
                continue
            if latest_item is None:
                latest_item = (cur_val, _extract_field(item, date_keys))
            elif prev_item is None:
                prev_item = (cur_val, _extract_field(item, date_keys))
                break
        if latest_item:
            val = latest_item[0]
            if latest_item[1]:
                fecha = str(latest_item[1])
        if variation is None and latest_item and prev_item and prev_item[0] not in (None, 0):
            try:
                variation = ((latest_item[0] - prev_item[0]) / prev_item[0]) * 100.0
            except Exception:
                variation = None

    html = await fetch_text(session, DOLARITO_RIESGO_HTML, headers=REQ_HEADERS)
    build_id: Optional[str] = None
    if html:
        m_build = re.search(r"<!--([A-Za-z0-9]{10,})-->", html)
        if m_build:
            build_id = m_build.group(1)

    if build_id:
        json_url = f"https://www.dolarito.ar/_next/data/{build_id}/indices/riesgo-pais.json"
        payload = await fetch_json(session, json_url, headers=REQ_HEADERS)
        if isinstance(payload, dict):
            _parse_json_payload(payload)

    if val is None or variation is None:
        await _parse_argdatos()

    if (val is None or variation is None) and html:
        stripped = re.sub(r"<[^>]+>", " ", html)
        stripped = " ".join(stripped.split())
        m_val = re.search(r"Riesgo\s+pa[ií]s[^\d]{0,10}(\d{3,5})", stripped, flags=re.I)
        if m_val:
            try:
                val = float(m_val.group(1))
            except Exception:
                val = None
        if variation is None:
            m_var = re.search(r"([+\-]?\d+[\.,]\d+)%", stripped)
            if m_var:
                try:
                    variation = float(m_var.group(1).replace(",", "."))
                except Exception:
                    variation = variation

    if variation is None and isinstance(RIESGO_CACHE, dict):
        cached_var = _safe_number(RIESGO_CACHE.get("variation"))
        if cached_var is not None:
            variation = cached_var
        elif val is not None:
            prev_cached_val = _safe_number(RIESGO_CACHE.get("val"))
            if prev_cached_val not in (None, 0):
                try:
                    variation = ((val - prev_cached_val) / prev_cached_val) * 100.0
                except Exception:
                    variation = None

    if val is None:
        cache_val = RIESGO_CACHE.get("val") if isinstance(RIESGO_CACHE, dict) else None
        if cache_val is None:
            return None
        try:
            cache_val_int = int(cache_val)
        except Exception:
            return None
        res = (
            cache_val_int,
            RIESGO_CACHE.get("fecha") if isinstance(RIESGO_CACHE, dict) else None,
            RIESGO_CACHE.get("variation") if isinstance(RIESGO_CACHE, dict) else None,
            True,
        )
        SHORT_CACHE.set(cache_key, res)
        return res

    try:
        rounded = int(round(val))
    except Exception:
        return None

    _save_riesgo_cache(rounded, fecha, variation)
    res = (rounded, fecha, variation, from_cache)
    SHORT_CACHE.set(cache_key, res)
    return res


def _variation_arrow(var: float) -> str:
    if var < 0:
        return "🔴"
    if var > 0:
        return "🟢"
    return "➡️"


def _circle_indicator(val: float, *, up_icon: str, down_icon: str) -> str:
    if val > 0:
        return up_icon
    if val < 0:
        return down_icon
    return "🟡"


def _format_riesgo_variation(var: Optional[float]) -> str:
    if not isinstance(var, (int, float)):
        return ""
    arrow = _circle_indicator(var, up_icon="🔴", down_icon="🟢")
    sign = "" if var < 0 else "+" if var > 0 else ""
    return f" {arrow} {sign}{var:.2f}%"

def _format_inflacion_variation(var: Optional[float]) -> str:
    if not isinstance(var, (int, float)):
        return " —"
    arrow = _circle_indicator(var, up_icon="🔴", down_icon="🟢")
    sign = "+" if var > 0 else ""
    return f" {arrow} {sign}{var:.1f}%"

def _format_reservas_variation(prev_val: Optional[float], cur_val: Optional[float]) -> str:
    if not isinstance(prev_val, (int, float)) or not isinstance(cur_val, (int, float)):
        return ""
    if prev_val == 0:
        return ""
    try:
        var = ((cur_val - prev_val) / prev_val) * 100.0
    except Exception:
        return ""
    arrow = _circle_indicator(var, up_icon="🟢", down_icon="🔴")
    sign = "+" if var > 0 else ""
    return f" {arrow} {sign}{var:.2f}%"

async def get_inflacion_mensual(
    session: ClientSession, httpx_client: Optional[httpx.AsyncClient] = None
) -> Optional[Tuple[float, Optional[str], Optional[float]]]:
    def _parse_period(raw: Optional[str]) -> Optional[datetime]:
        if not raw:
            return None
        for fmt in ("%Y-%m-%d", "%Y-%m", "%Y%m", "%Y/%m/%d"):
            try:
                return datetime.strptime(raw[: len(fmt)], fmt)
            except Exception:
                continue
        return None

    def _calc_variation(cur: Optional[float], prev: Optional[float]) -> Optional[float]:
        if not isinstance(cur, (int, float)) or not isinstance(prev, (int, float)):
            return None
        if prev == 0:
            return None
        try:
            return ((cur - prev) / prev) * 100.0
        except Exception:
            return None

    candidates: List[str] = []
    for suf in ("/inflacion", "/inflacion/mensual", "/inflacion/mensual/ultimo"):
        for base in ARG_DATOS_BASES:
            candidates.append(base + suf)
    candidates.append("https://api.argentinadatos.com/v1/finanzas/indices/inflacion")

    series: List[Dict[str, Any]] = []
    latest_val: Optional[float] = None
    latest_period: Optional[str] = None
    prev_val: Optional[float] = None

    for url in candidates:
        j: Any = await fetch_json_httpx(httpx_client, url, follow_redirects=True) if httpx_client else None
        if not j:
            j = await fetch_json(session, url)
        if not j:
            continue
        if isinstance(j, dict) and "serie" in j and isinstance(j.get("serie"), list):
            j = j.get("serie") or []
        if isinstance(j, list):
            series = [item for item in j if isinstance(item, dict)]
        elif isinstance(j, dict):
            series = [j]
        else:
            continue
        if series:
            break

    if not series:
        return None

    parsed_entries: List[Tuple[Optional[datetime], str, float]] = []
    for item in series:
        raw_val = item.get("valor")
        if raw_val in (None, ""):
            continue
        try:
            value = float(raw_val)
        except Exception:
            continue
        per = item.get("fecha") or item.get("periodo")
        parsed_entries.append((_parse_period(per), per or "", value))

    if not parsed_entries:
        return None

    parsed_entries.sort(key=lambda t: (t[0] or datetime.min))
    _, latest_period, latest_val = parsed_entries[-1]
    prev_val = parsed_entries[-2][2] if len(parsed_entries) > 1 else None

    variation = _calc_variation(latest_val, prev_val)
    return (latest_val, latest_period or None, variation)


def _save_reservas_cache(val: float, fecha: Optional[str], prev_val: Optional[float]) -> None:
    global RESERVAS_CACHE
    new_cache = {
        "val": val,
        "prev_val": prev_val,
        "fecha": fecha,
        "updated_at": time(),
    }
    if RESERVAS_CACHE != new_cache:
        RESERVAS_CACHE = new_cache
        try:
            asyncio.create_task(save_state())
        except Exception:
            pass

async def get_reservas_lamacro(session: ClientSession) -> Optional[Tuple[float, Optional[str]]]:
    html = await fetch_text(session, LAMACRO_RESERVAS_URL)
    if not html:
        log.warning("Reservas lamacro: HTML vacío o inaccesible (%s)", LAMACRO_RESERVAS_URL)
        return None

    def _parse_reserva_val(raw: str) -> Optional[float]:
        cleaned = raw.strip().replace("\xa0", " ")
        cleaned = cleaned.replace(".", "").replace(",", ".")
        cleaned = re.sub(r"[^0-9\.-]", "", cleaned)
        try:
            return float(cleaned)
        except Exception:
            return None

    val: Optional[float] = None
    fecha: Optional[str] = None

    meta_match = re.search(
        r"Valor actual:\s*([0-9\.,]+).*?Última actualización:\s*([0-3]\d/[0-1]\d/\d{4})",
        html,
        flags=re.S,
    )
    if meta_match:
        val = _parse_reserva_val(meta_match.group(1))
        fecha = meta_match.group(2)

    if val is None:
        m_val = re.search(r"initialValue\\\":\s*([0-9\.,]+)", html) or re.search(r"initialValue\":\s*([0-9\.,]+)", html)
        if m_val:
            val = _parse_reserva_val(m_val.group(1))

    if val is None:
        m_val = re.search(r"(?:Último dato|Valor actual)\s*:\s*([0-9\.,]+)", html)
        if m_val:
            val = _parse_reserva_val(m_val.group(1))

    if fecha is None:
        m_date = re.search(r"Última actualización:?\s*([0-3]\d/[0-1]\d/\d{4})", html)
        if m_date:
            fecha = m_date.group(1)
    if fecha is None:
        m_date = re.search(r"([0-3]\d/[0-1]\d/\d{4})", html)
        if m_date:
            fecha = m_date.group(1)

    if val is None:
        log.warning("Reservas lamacro: no pude extraer valor desde HTML actual")
        return None

    return (val, fecha)


async def get_reservas_con_variacion(
    session: ClientSession,
) -> Optional[Tuple[float, Optional[str], Optional[float], bool]]:
    global RESERVAS_CACHE
    from_cache = False
    cache_key = "macro:reservas"
    cached_short = SHORT_CACHE.get(cache_key)
    if isinstance(cached_short, (list, tuple)) and len(cached_short) >= 3:
        return tuple(cached_short)  # type: ignore[return-value]
    res = await get_reservas_lamacro(session)
    if not res:
        cache_val = RESERVAS_CACHE.get("val") if isinstance(RESERVAS_CACHE, dict) else None
        cache_fecha = RESERVAS_CACHE.get("fecha") if isinstance(RESERVAS_CACHE, dict) else None
        cache_prev = RESERVAS_CACHE.get("prev_val") if isinstance(RESERVAS_CACHE, dict) else None
        if cache_val is None:
            log.warning("Reservas: sin dato fresco ni caché válida")
            return None
        log.warning("Reservas: usando dato en caché por fallo de scraping")
        try:
            cur_val = float(cache_val)
            prev_val = float(cache_prev) if cache_prev is not None else None
        except Exception:
            return None
        fecha = cache_fecha
        from_cache = True
    else:
        val, fecha = res
        try:
            cur_val = float(val)
        except Exception:
            return None
        cached_val = RESERVAS_CACHE.get("val") if isinstance(RESERVAS_CACHE, dict) else None
        cached_prev = RESERVAS_CACHE.get("prev_val") if isinstance(RESERVAS_CACHE, dict) else None
        prev_val: Optional[float] = None
        cached_val_f: Optional[float] = None
        cached_prev_f: Optional[float] = None
        if isinstance(cached_val, (int, float)):
            try:
                cached_val_f = float(cached_val)
            except Exception:
                cached_val_f = None
        if isinstance(cached_prev, (int, float)):
            try:
                cached_prev_f = float(cached_prev)
            except Exception:
                cached_prev_f = None

        if cached_val_f is not None and cached_val_f != cur_val:
            prev_val = cached_val_f
        elif cached_prev_f is not None:
            prev_val = cached_prev_f
        _save_reservas_cache(cur_val, fecha, prev_val)
    res_tuple = (cur_val, fecha, prev_val, from_cache)
    SHORT_CACHE.set(cache_key, res_tuple)
    return res_tuple

# ============================ RAVA ============================

async def _fetch_rava_profile(session: ClientSession, symbol: str) -> Optional[Dict[str, Any]]:
    url = RAVA_PERFIL_URL.format(symbol=quote(symbol))
    html: Optional[str] = None
    try:
        async with session.get(url, headers=REQ_HEADERS, timeout=ClientTimeout(total=12)) as resp:
            if resp.status == 200:
                html = await resp.text()
    except Exception as exc:
        log.info("rava_fetch_primary_failed url=%s err=%s", url, exc)

    if html is None:
        def _blocking_fetch() -> Optional[str]:
            try:
                req = urllib.request.Request(url, headers=REQ_HEADERS)
                with urllib.request.urlopen(req, timeout=12) as resp:  # type: ignore[arg-type]
                    if getattr(resp, "status", None) not in (None, 200):
                        return None
                    return resp.read().decode()
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:  # type: ignore[attr-defined]
                log.info("rava_fetch_fallback_failed url=%s err=%s", url, exc)
                return None

        html = await asyncio.to_thread(_blocking_fetch)
        if html is None:
            return None

    match = re.search(r':res="(\{.*?\})"', html, flags=re.S)
    if not match:
        return None
    try:
        data = json.loads(_html.unescape(match.group(1)))
    except Exception:
        return None
    return data


async def _screenermatic_bonds(session: ClientSession) -> Dict[str, Dict[str, Optional[float]]]:
    cache_key = "screenermatic:bonds"
    cached = SHORT_CACHE.get(cache_key)
    if isinstance(cached, dict):
        return cached  # type: ignore[return-value]

    html: Optional[str] = None
    try:
        async with session.get(SCREENERMATIC_BONDS_URL, headers=REQ_HEADERS, timeout=ClientTimeout(total=12)) as resp:
            if resp.status == 200:
                html = await resp.text()
    except Exception as exc:
        log.info("screenermatic_fetch_failed url=%s err=%s", SCREENERMATIC_BONDS_URL, exc)
        return {}

    if not html:
        return {}

    entries: Dict[str, Dict[str, Optional[float]]] = {}
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, flags=re.S)
    for row in rows:
        if "<td" not in row and "<th" not in row:
            continue
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, flags=re.S)
        if len(cells) < 12:
            continue
        clean = [re.sub("<[^<]+?>", "", c).strip() for c in cells]
        symbol = clean[0]
        if symbol not in BONOS_AR:
            continue
        last_px = None
        last_chg = None
        last_ts: Optional[int] = None
        try:
            last_px = float(clean[11].replace(".", "").replace(",", ".")) if clean[11] else None
        except Exception:
            last_px = None
        try:
            last_chg = float(clean[12].replace("%", "")) if len(clean) > 12 and clean[12] else None
        except Exception:
            last_chg = None
        try:
            if clean[9]:
                dt = datetime.strptime(clean[9], "%Y-%m-%d").replace(tzinfo=TZ)
                last_ts = int(dt.timestamp())
        except Exception:
            last_ts = None
        entries[symbol] = {"last_px": last_px, "last_chg": last_chg, "last_ts": last_ts}

    SHORT_CACHE.set(cache_key, entries, ttl=120)
    return entries

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

    window_1m = WINDOW_DAYS[1]
    window_3m = WINDOW_DAYS[3]
    window_6m = WINDOW_DAYS[6]

    base6 = closes[-window_6m] if len(closes) >= window_6m else None
    base3 = closes[-window_3m] if len(closes) >= window_3m else None
    base1 = closes[-window_1m] if len(closes) >= window_1m else None
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
    day = 24 * 3600
    t6 = last_ts - WINDOW_DAYS[6] * day
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
    slope_window = WINDOW_DAYS[1]
    if sma50[-1] is not None and len(closes) > (slope_window - 1) and sma50[-slope_window] is not None:
        try:
            slope50 = ((sma50[-1] / sma50[-slope_window]) - 1.0) * 100.0
        except Exception:
            slope50 = None
    trend_flag = None
    if sma200[-1] is not None:
        trend_flag = 1.0 if last > sma200[-1] else -1.0

    last_px: Optional[float] = None
    prev_px: Optional[float] = None
    last_chg: Optional[float] = None
    try:
        last_px = float(last)
    except Exception:
        last_px = None
    if prev is not None:
        try:
            prev_px = float(prev)
        except Exception:
            prev_px = None
        if prev_px not in (None, 0.0):
            try:
                last_chg = ((last_px / prev_px) - 1.0) * 100.0 if last_px is not None else None
            except Exception:
                last_chg = None

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
        "last_px": last_px,
        "prev_px": prev_px,
        "last_chg": last_chg,
    }

async def _rava_metrics(session: ClientSession, symbol: str) -> Dict[str, Optional[float]]:
    base = {"6m": None, "3m": None, "1m": None, "last_ts": None, "vol_ann": None,
            "dd6m": None, "hi52": None, "slope50": None, "trend_flag": None,
            "last_px": None, "prev_px": None, "last_chg": None}
    screenermatic = await _screenermatic_bonds(session)
    data = await _fetch_rava_profile(session, symbol)
    if not data:
        if symbol in screenermatic:
            base.update(screenermatic[symbol])
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

    if symbol in screenermatic:
        fallback = screenermatic[symbol]
        if base["last_px"] is None and fallback.get("last_px") is not None:
            base["last_px"] = fallback.get("last_px")
        if base["last_chg"] is None and fallback.get("last_chg") is not None:
            base["last_chg"] = fallback.get("last_chg")
        if base["last_ts"] is None and fallback.get("last_ts") is not None:
            base["last_ts"] = fallback.get("last_ts")

    base["currency"] = bono_moneda(symbol)
    return base

# ============================ YF MÉTRICAS ============================

async def _yf_chart_1y(session: ClientSession, symbol: str, interval: str) -> Optional[Dict[str, Any]]:
    for base in YF_URLS:
        params = {"range": "1y", "interval": interval, "events": "div,split"}
        base_url = base.format(symbol=symbol)
        suspended_until = http_service.is_suspended(urlparse(base_url).netloc or "yahoo")
        if suspended_until:
            log.info("skip_yahoo_source_suspended host=%s resume_at=%s", urlparse(base_url).netloc, suspended_until)
            continue
        j = await fetch_json(
            session,
            base_url,
            headers=YF_HEADERS,
            params=params,
            source=urlparse(base_url).netloc or "yahoo",
        )
        try:
            res = j.get("chart", {}).get("result", [])[0]
            return res
        except Exception:
            continue
    return None

def _metrics_from_chart(res: Dict[str, Any], symbol: Optional[str] = None) -> Optional[Dict[str, Optional[float]]]:
    try:
        ts = res["timestamp"]; closes_raw = res["indicators"]["adjclose"][0]["adjclose"]
        pairs = [(t,c) for t,c in zip(ts, closes_raw) if (t is not None and c is not None)]
        if len(pairs) < 30: return None
        ts = [p[0] for p in pairs]; closes = [p[1] for p in pairs]
        idx_last = len(closes)-1; last = closes[idx_last]; t_last = ts[idx_last]
        prev = closes[idx_last-1] if idx_last >= 1 else None
        last_chg = ((last/prev - 1.0)*100.0) if (prev is not None and prev > 0) else None
        if symbol and len(closes) < WINDOW_DAYS[6]:
            log.warning(
                "YF series corta para %s: %s ruedas (min %s)",
                symbol,
                len(closes),
                WINDOW_DAYS[6],
            )

        # Retornos close-to-close por ruedas desde el último cierre.
        window_1m = 21
        window_3m = 63
        window_6m = 126

        def _window_base(values: List[float], window: int) -> Optional[float]:
            if len(values) >= window:
                return values[-window]
            return None

        base1 = _window_base(closes, window_1m)
        base3 = _window_base(closes, window_3m)
        base6 = _window_base(closes, window_6m)
        ret1 = (last / base1 - 1.0) * 100.0 if base1 else None
        ret3 = (last / base3 - 1.0) * 100.0 if base3 else None
        ret6 = (last / base6 - 1.0) * 100.0 if base6 else None

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

        t6 = t_last - 180 * 24 * 3600
        idx_cut = next((i for i, t in enumerate(ts) if t >= t6), 0)
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

_INTERVAL_VALIDATED = False

def _validate_interval_independence() -> None:
    global _INTERVAL_VALIDATED
    if _INTERVAL_VALIDATED or os.getenv("YF_VALIDATE_INTERVAL") != "1":
        return
    _INTERVAL_VALIDATED = True
    start = datetime(2024, 1, 1, tzinfo=TZ)
    days = 240
    daily_ts = [int((start + timedelta(days=i)).timestamp()) for i in range(days)]
    weekly_ts = [daily_ts[i] for i in range(0, days, 7)]

    def _build_res(timestamps: List[int]) -> Dict[str, Any]:
        base = timestamps[0]
        closes = [100.0 + 0.1 * ((t - base) / 86400.0) for t in timestamps]
        return {"timestamp": timestamps, "indicators": {"adjclose": [{"adjclose": closes}]}}

    daily = _metrics_from_chart(_build_res(daily_ts), symbol="__validation_daily__")
    weekly = _metrics_from_chart(_build_res(weekly_ts), symbol="__validation_weekly__")
    if not daily or not weekly:
        log.warning("Validación intervalos: métricas no disponibles.")
        return
    for horizon in ("1m", "3m", "6m"):
        d_val = daily.get(horizon)
        w_val = weekly.get(horizon)
        if d_val is None or w_val is None:
            continue
        if abs(d_val - w_val) > 0.75:
            log.warning("Validación intervalos: %s diff=%.3f", horizon, abs(d_val - w_val))

async def _yf_metrics_1y(session: ClientSession, symbol: str) -> Dict[str, Optional[float]]:
    out = {"6m": None, "3m": None, "1m": None, "last_ts": None, "vol_ann": None, "dd6m": None, "hi52": None, "slope50": None,
           "trend_flag": None, "last_px": None, "prev_px": None, "last_chg": None}
    _validate_interval_independence()
    for interval in ("1d", "1wk"):
        res = await _yf_chart_1y(session, symbol, interval)
        if res:
            m = _metrics_from_chart(res, symbol=symbol)
            if m: out.update(m); break
    return out

def _combine_returns(usd_ret: Optional[float], fx_ret: Optional[float]) -> Optional[float]:
    if usd_ret is None and fx_ret is None:
        return None
    if usd_ret is None:
        return fx_ret
    if fx_ret is None:
        return usd_ret
    try:
        return ((1.0 + float(usd_ret) / 100.0) * (1.0 + float(fx_ret) / 100.0) - 1.0) * 100.0
    except Exception:
        return None

def _remove_fx_return(ars_ret: Optional[float], fx_ret: Optional[float]) -> Optional[float]:
    if ars_ret is None and fx_ret is None:
        return None
    if ars_ret is None:
        return None
    if fx_ret is None:
        return ars_ret
    try:
        return ((1.0 + float(ars_ret) / 100.0) / (1.0 + float(fx_ret) / 100.0) - 1.0) * 100.0
    except Exception:
        return None

async def _cedear_metrics(
    session: ClientSession,
    symbol: str,
    fx_metrics: Dict[str, Optional[float]],
) -> Dict[str, Optional[float]]:
    local = await _yf_metrics_1y(session, symbol)
    underlying = cedear_underlying_symbol(symbol)
    usd_metrics: Dict[str, Optional[float]] = {}
    if underlying:
        usd_metrics = await _yf_metrics_1y(session, underlying)

    for horizon in ("1m", "3m", "6m"):
        ars_val = local.get(horizon)
        fx_val = fx_metrics.get(horizon)
        usd_val = usd_metrics.get(horizon)
        if ars_val is None:
            ars_val = _combine_returns(usd_val, fx_val)
        if usd_val is None:
            usd_val = _remove_fx_return(ars_val, fx_val)
        local[horizon] = ars_val
        local[f"{horizon}_ars"] = ars_val
        local[f"{horizon}_usd"] = usd_val
        local[f"{horizon}_fx"] = fx_val

    ts_candidates = [v for v in [local.get("last_ts"), usd_metrics.get("last_ts")] if v]
    if ts_candidates:
        local["last_ts"] = max(ts_candidates)
    return local

async def metrics_for_symbols(session: ClientSession, symbols: List[str]) -> Tuple[Dict[str, Dict[str, Optional[float]]], Optional[int]]:
    out = {s: {"6m": None, "3m": None, "1m": None, "last_ts": None, "vol_ann": None, "dd6m": None, "hi52": None,
               "slope50": None, "trend_flag": None, "last_px": None, "prev_px": None, "last_chg": None} for s in symbols}
    sem = asyncio.Semaphore(4)
    fx_type = "ccl"
    fx_metrics = await _fx_metrics_series(session, fx_type)
    if not any(fx_metrics.get(k) for k in ("1m", "3m", "6m")):
        fx_type = "mep"
        fx_metrics = await _fx_metrics_series(session, fx_type)

    async def work(sym: str):
        async with sem:
            if sym.startswith("FCI-"):
                out[sym] = await _fci_metrics(session, sym)
            elif sym in BONOS_AR:
                out[sym] = await _rava_metrics(session, sym)
            elif sym.upper() in CEDEARS_SET:
                out[sym] = await _cedear_metrics(session, sym, fx_metrics)
            else:
                out[sym] = await _yf_metrics_1y(session, sym)
    await asyncio.gather(*(work(s) for s in symbols))
    last_ts = None
    for d in out.values():
        ts = d.get("last_ts")
        if ts: last_ts = ts if last_ts is None else max(last_ts, ts)
    return out, last_ts


async def metrics_for_symbols_cached(session: ClientSession, symbols: List[str], ttl: int = 60) -> Tuple[Dict[str, Dict[str, Optional[float]]], Optional[int]]:
    cache_key = f"rankings:metrics:{','.join(sorted(symbols))}"
    cached = SHORT_CACHE.get(cache_key)
    if isinstance(cached, (list, tuple)) and len(cached) == 2:
        data, ts = cached
        if isinstance(data, dict):
            return data, ts  # type: ignore[return-value]
    invalidate_rankings_cache()
    res = await metrics_for_symbols(session, symbols)
    SHORT_CACHE.set(cache_key, res, ttl=ttl)
    return res

# ============================ NOTICIAS ============================

from xml.etree import ElementTree as ET

NewsItem = Tuple[str, str, Optional[str]]
RawNewsEntry = Tuple[str, str, Optional[str], Optional[str]]
RSS_FEEDS = [
    # Fuentes nacionales directas para evitar links de Google News.
    "https://www.clarin.com/rss/economia/",
    "https://www.ambito.com/rss/pages/economia.xml",
    "https://www.cronista.com/rss/economiapolitica/",
    "https://www.iprofesional.com/rss/economia",
    "https://www.perfil.com/feed/economia",
]

# Fuentes extendidas (más sitios nacionales) que se consultan de manera
# incremental cuando faltan notas relevantes en las fuentes principales.
RSS_FEEDS_EXTENDED = [
    "https://www.clarin.com/rss/mundo/",
    "https://www.cronista.com/rss/finanzas-mercados/",
    "https://www.pagina12.com.ar/arc/outboundfeeds/rss/?outputType=xml",
    "https://www.telam.com.ar/rss2/economia.xml",
]
NATIONAL_NEWS_DOMAINS: Set[str] = {
    "ambito.com",
    "iprofesional.com",
    "infobae.com",
    "perfil.com",
    "baenegocios.com",
    "telam.com.ar",
    "cronista.com",
    "eleconomista.com.ar",
    "clarin.com",
    "lanacion.com.ar",
    "pagina12.com.ar",
}
KEYWORDS = ["inflación","ipc","índice de precios","devalu","dólar","ccl","mep","blue",
            "bcra","reservas","tasas","pases","fmi","deuda","riesgo país",
            "cepo","importaciones","exportaciones","merval","acciones","bonos","brecha",
            "subsidios","retenciones","tarifas","liquidez","recaudación","déficit"]
DOLLAR_NEWS_KEYWORDS = [
    "dolar",
    "dolares",
    "dolarizacion",
    "dolarizar",
    "usd",
    "blue",
    "mep",
    "ccl",
    "contado con liqui",
]
NEWS_TOPICS = [
    ("cambio", ["dolar", "blue", "mep", "ccl", "brecha", "cambio", "oficial", "mayorista", "contado con liqui"]),
    ("inflacion", ["inflacion", "ipc", "precios", "indice de precios", "inflacionaria"]),
    ("reservas", ["reserva", "bcra", "pases", "tasas", "leliq", "liquidez"]),
    ("deuda", ["bono", "bonos", "riesgo", "deuda", "fmi", "default", "canje"]),
    ("actividad", ["actividad", "pbi", "industria", "empleo", "salario", "consumo", "produccion", "pymes"]),
    ("fiscal", ["deficit", "superavit", "recaudacion", "impuesto", "tribut", "presupuesto"]),
    ("comercio", ["export", "import", "balanza", "aduana", "soja", "agro", "granos", "campo"]),
    ("energia", ["energia", "combustible", "nafta", "gas", "petroleo", "ypf", "electricidad"]),
]

NEWS_CATEGORY_KEYWORDS = [
    (
        "empresarial",
        [
            "empresa",
            "empresario",
            "empresas",
            "compania",
            "compañia",
            "corporacion",
            "corporación",
            "fusion",
            "adquisicion",
            "adquisición",
            "startup",
            "negocio",
            "sector",
            "inversion",
            "inversión",
            "pyme",
            "pymes",
            "proveedor",
            "proveedores",
            "cliente",
            "empleados",
            "facturacion",
            "facturación",
        ],
    ),
    (
        "financiero",
        [
            "bono",
            "bonos",
            "accion",
            "acciones",
            "merval",
            "wall street",
            "riesgo pais",
            "riesgo país",
            "mercado",
            "bolsa",
            "banco",
            "bancos",
            "fintech",
            "tarjeta",
            "tarjetas",
            "cedear",
            "cedears",
            "etf",
            "plazo fijo",
            "plazos fijos",
            "prestamo",
            "prestamos",
        ],
    ),
    (
        "economico",
        [
            "pbi",
            "actividad",
            "actividad economica",
            "actividad económica",
            "inflacion",
            "inflación",
            "consumo",
            "produccion",
            "producción",
            "superavit",
            "déficit",
            "deficit",
            "recaudacion",
            "industria",
            "crecimiento",
            "recuperacion",
            "recesion",
            "salario",
            "salarios",
            "empleo",
            "paritaria",
            "paritarias",
            "macro",
            "impuesto",
            "impuestos",
        ],
    ),
    (
        "social",
        [
            "protesta",
            "movilizacion",
            "movilización",
            "paro",
            "sindicato",
            "pobreza",
            "salud",
            "educacion",
            "educación",
            "seguridad social",
            "jubil",
            "conflicto",
            "despido",
            "despidos",
        ],
    ),
    (
        "politica",
        [
            "gobierno",
            "presidente",
            "ministro",
            "congreso",
            "senado",
            "diputados",
            "eleccion",
            "elección",
            "elecciones",
            "decreto",
            "ley",
            "casa rosada",
            "oficialismo",
            "oposicion",
            "oposición",
            "gobernador",
            "intendencia",
        ],
    ),
]
NEWS_CATEGORIES_ORDER = ["empresarial", "financiero", "economico", "social", "politica"]


def _normalize_topic_text(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text.lower())
    cleaned = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned)
    return " ".join(cleaned.split())


def _normalize_news_title(title: str) -> str:
    norm = _normalize_topic_text(title)
    return " ".join(re.sub(r"[^a-z0-9]+", " ", norm).split())


def _news_dedup_key(title: str, max_words: int = 9) -> str:
    normalized = _normalize_news_title(title)
    parts = normalized.split()
    if len(parts) > max_words:
        parts = parts[:max_words]
    return " ".join(parts)


def _dedup_news_items(items: List[NewsItem], limit: Optional[int] = None) -> List[NewsItem]:
    seen_titles: Set[str] = set()
    seen_stems: Set[str] = set()
    seen_links: Set[str] = set()
    deduped: List[NewsItem] = []
    for entry in items:
        title, link = entry[0], entry[1]
        img = entry[2] if len(entry) >= 3 else None
        norm_title = _normalize_news_title(title)
        stem_key = _news_dedup_key(title)
        clean_link = _canonical_news_link(link)
        if norm_title in seen_titles or stem_key in seen_stems or clean_link in seen_links:
            continue
        seen_titles.add(norm_title)
        seen_stems.add(stem_key)
        seen_links.add(clean_link)
        deduped.append((title, link, img))
        if limit is not None and len(deduped) >= limit:
            break
    return deduped


def _is_dollar_related(title: str, desc: Optional[str]) -> bool:
    parts = [_normalize_topic_text(title)]
    if desc:
        parts.append(_normalize_topic_text(desc))
    joined = " ".join(parts)
    return any(kw in joined for kw in DOLLAR_NEWS_KEYWORDS)


def _topic_for_title(title: str) -> str:
    norm = _normalize_topic_text(title)
    for topic, keywords in NEWS_TOPICS:
        if any(kw in norm for kw in keywords):
            return topic
    return "otros"


def _news_category_for(title: str, desc: Optional[str] = None) -> str:
    joined_parts = [
        _normalize_topic_text(title),
        _normalize_topic_text(desc or ""),
    ]
    joined = " ".join([part for part in joined_parts if part])
    for cat, keywords in NEWS_CATEGORY_KEYWORDS:
        if any(kw in joined for kw in keywords):
            return cat
    return "otros"

def domain_of(url: str) -> str:
    try: return urlparse(url).netloc.lower()
    except Exception: return ""

def _score_title(title: str) -> int:
    """Puntaje determinista que prioriza términos macro y de mercados locales."""
    t = title.lower(); score = 0
    # Peso fuerte para macro local
    for kw in KEYWORDS:
        if kw in t: score += 5
    # Dinámica de dólar e inflación
    for kw in ("dólar","dolar","inflación","ipc","brecha","ccl","mep"):
        if kw in t: score += 3
    # Medidas, leyes y shocks
    for kw in ("sube","baja","récord","acelera","cae","acuerdo","medida","ley","resolución","reperfil","anuncio","emergencia"):
        if kw in t: score += 1
    # Títulos más largos suelen ser más descriptivos
    score += min(len(t) // 12, 3)
    return score


def _is_economic_relevant(title: str, desc: Optional[str]) -> bool:
    text = " ".join(filter(None, [title, desc or ""]))
    normalized = _normalize_topic_text(text)
    if any(kw in normalized for kw in KEYWORDS):
        return True
    for _, keywords in NEWS_TOPICS:
        if any(kw in normalized for kw in keywords):
            return True
    return False


def _canonical_news_link(link: str) -> str:
    try:
        parsed = urlparse(link)
        cleaned = parsed._replace(query="", fragment="", params="")
        return cleaned.geturl()
    except Exception:
        return link

GOOGLE_NEWS_DOMAINS = {"news.google.com", "news.google.com.ar", "news.google.com.br"}


def _normalize_feed_link(link: str) -> Optional[str]:
    try:
        parsed = urlparse(link)
    except Exception:
        return link
    netloc = parsed.netloc.lower()
    if netloc in GOOGLE_NEWS_DOMAINS:
        qs = parsed.query or ""
        qparams = dict((k, v[0]) for k, v in parse_qs(qs).items() if v)
        target = qparams.get("url")
        if target and target.startswith("http"):
            target_netloc = domain_of(target)
            if target_netloc not in GOOGLE_NEWS_DOMAINS:
                return target
        return None
    return link

def _is_probably_article_url(link: str) -> bool:
    try:
        parsed = urlparse(link)
    except Exception:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    path = (parsed.path or "").strip()
    if not path or path == "/":
        return False
    parts = [p for p in path.strip("/").split("/") if p]
    if not parts:
        return False
    tail = parts[-1].lower()
    if tail in {"economia", "finanzas", "finanzas-mercados", "finanzas-y-mercados"}:
        return False
    if len(parts) == 1 and len(tail) < 10:
        return False
    has_signal = ("-" in tail) or any(ch.isdigit() for ch in tail) or len(tail) >= 10
    return has_signal or len(parts) >= 2

def _image_from_description(desc: str) -> Optional[str]:
    if not desc:
        return None
    m = re.search(r"<img[^>]+src=\"([^\"]+)\"", desc, flags=re.I)
    if m:
        url = _html.unescape(m.group(1).strip())
        return url if url.startswith("http") else None
    return None

def _extract_feed_image(element: Optional[ET.Element], desc: Optional[str]) -> Optional[str]:
    if element is None:
        return _image_from_description(desc or "")

    def _clean(url: Optional[str]) -> Optional[str]:
        if not url:
            return None
        url = _html.unescape(url.strip())
        return url if url.startswith("http") else None

    for xpath in (".//{*}content", ".//{*}thumbnail", "enclosure"):
        for child in element.findall(xpath):
            mime = (child.get("type") or "").lower()
            if xpath == "enclosure" and mime and not mime.startswith("image"):
                continue
            url = _clean(child.get("url") or child.get("href"))
            if url:
                return url

    return _image_from_description(desc or "")

def _parse_feed_entries(xml: str) -> List[RawNewsEntry]:
    out: List[RawNewsEntry] = []
    try:
        root = ET.fromstring(xml)
    except Exception:
        # Si el XML está mal formado, seguimos con el fallback regex
        root = None
    for item in root.findall(".//item") if root is not None else []:
        t_el = item.find("title"); l_el = item.find("link")
        d_el = item.find("description")
        t = (t_el.text or "").strip() if (t_el is not None and t_el.text) else None
        l = (l_el.text or "").strip() if (l_el is not None and l_el.text) else None
        l = _normalize_feed_link(l) if l else None
        desc = None
        if d_el is not None and d_el.text:
            desc = d_el.text.strip()
        if t and l and l.startswith("http") and _is_probably_article_url(l):
            out.append((t, l, desc, _extract_feed_image(item, desc)))
    for entry in root.findall(".//{*}entry") if root is not None else []:
        t_el = entry.find(".//{*}title")
        link_el = entry.find(".//{*}link[@rel='alternate']") or entry.find(".//{*}link")
        summary_el = entry.find(".//{*}summary")
        t = (t_el.text or "").strip() if (t_el is not None and t_el.text) else None
        l = link_el.get("href").strip() if (link_el is not None and link_el.get("href")) else None
        l = _normalize_feed_link(l) if l else None
        if (not l) and entry.find(".//{*}id") is not None:
            l = (entry.find(".//{*}id").text or "").strip()
        desc = None
        if summary_el is not None and summary_el.text:
            desc = summary_el.text.strip()
        if t and l and l.startswith("http") and _is_probably_article_url(l):
            out.append((t, l, desc, _extract_feed_image(entry, desc)))
    if not out:
        for m in re.finditer(r"<title>(.*?)</title>.*?<link>(https?://[^<]+)</link>", xml, flags=re.S|re.I):
            t = re.sub(r"<.*?>", "", m.group(1)).strip(); l = m.group(2).strip()
            if t and l and _is_probably_article_url(l):
                out.append((t, l, None, None))
    return out

_IMAGE_LOGO_PATTERNS = [
    "logo",
    "favicon",
    "sprite",
    "icon",
    "placeholder",
    "default",
    "brand",
    "header",
    "print",
    "google_news",
    "googlenews",
    "gn-logo",
    "gstatic",
    "googleusercontent",
]


def _clean_image_url(url: Optional[str], base_url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    url = _html.unescape(url.strip())
    if url.startswith("//"):
        url = f"https:{url}"
    if base_url:
        url = urljoin(base_url, url)
    if not url.startswith("http"):
        return None
    if url.lower().endswith(".svg"):
        return None
    if url.lower().startswith("data:"):
        return None
    return url


def _is_logo_like(url: str) -> bool:
    low = url.lower()
    if any(pattern in low for pattern in _IMAGE_LOGO_PATTERNS):
        return True
    if re.search(r"/(?:logo|logos|favicon|icons?)/", low):
        return True
    if re.search(r"(?:^|[\W_])logo(?:[\W_]|$)", low):
        return True
    return False


def _is_unwanted_image_host(url: str) -> bool:
    dom = domain_of(url)
    dom = dom[4:] if dom.startswith("www.") else dom
    if dom in GOOGLE_NEWS_DOMAINS:
        return True
    unwanted_hosts = {
        "gstatic.com",
        "googleusercontent.com",
        "google.com",
        "google.com.ar",
    }
    return dom in unwanted_hosts


def _extract_json_ld_images(html: str, base_url: Optional[str]) -> List[str]:
    images: List[str] = []
    for match in re.finditer(r"<script[^>]+type=['\"]application/ld\+json['\"][^>]*>(.*?)</script>", html, flags=re.I | re.S):
        try:
            data = json.loads(match.group(1))
        except Exception:
            continue

        def _collect_image(obj: Any) -> None:
            if isinstance(obj, str):
                cleaned = _clean_image_url(obj, base_url)
                if cleaned:
                    images.append(cleaned)
            elif isinstance(obj, list):
                for it in obj:
                    _collect_image(it)
            elif isinstance(obj, dict):
                if "image" in obj:
                    _collect_image(obj.get("image"))
                for key in ("thumbnailUrl", "thumbnail", "logo"):
                    if key in obj:
                        _collect_image(obj.get(key))

        _collect_image(data)
    return images


def _extract_page_image(html: str, base_url: Optional[str]) -> Optional[str]:
    if not html:
        return None

    patterns = [
        r"<meta[^>]+(?:property|name)=['\"]og:image(?::secure_url)?['\"][^>]+content=['\"]([^'\"]+)['\"]",
        r"<meta[^>]+name=['\"]twitter:image(?::src)?['\"][^>]+content=['\"]([^'\"]+)['\"]",
        r"<link[^>]+rel=['\"]image_src['\"][^>]+href=['\"]([^'\"]+)['\"]",
    ]

    candidates: List[Tuple[str, int]] = []

    def _score_image(url: str, width: Optional[int] = None, height: Optional[int] = None) -> int:
        score = 0
        if width and width >= 180:
            score += 2
        if height and height >= 180:
            score += 2
        if width and height and min(width, height) >= 300:
            score += 2
        if _is_unwanted_image_host(url):
            score -= 4
        if _is_logo_like(url):
            score -= 3
        return score

    for pat in patterns:
        for m in re.finditer(pat, html, flags=re.I):
            img = _clean_image_url(m.group(1), base_url)
            if img and not _is_logo_like(img) and not _is_unwanted_image_host(img):
                candidates.append((img, _score_image(img)))

    for json_img in _extract_json_ld_images(html, base_url):
        if json_img and not _is_logo_like(json_img) and not _is_unwanted_image_host(json_img):
            candidates.append((json_img, _score_image(json_img)))

    if candidates:
        candidates.sort(key=lambda it: it[1], reverse=True)
        return candidates[0][0]

    for m in re.finditer(r"<img[^>]+src=['\"]([^'\"]+)['\"][^>]*>", html, flags=re.I):
        img = _clean_image_url(m.group(1), base_url)
        if not img or _is_logo_like(img) or _is_unwanted_image_host(img):
            continue
        width_match = re.search(r"width=['\"]?(\d+)", m.group(0), flags=re.I)
        height_match = re.search(r"height=['\"]?(\d+)", m.group(0), flags=re.I)
        width = int(width_match.group(1)) if width_match else None
        height = int(height_match.group(1)) if height_match else None
        if width and width < 120:
            continue
        if height and height < 120:
            continue
        scored = _score_image(img, width, height)
        candidates.append((img, scored))

    if candidates:
        candidates.sort(key=lambda it: it[1], reverse=True)
        return candidates[0][0]
    return None

async def _fallback_image_from_url(session: ClientSession, url: str) -> Optional[str]:
    try:
        async with session.get(
            url,
            timeout=ClientTimeout(total=8),
            headers={**REQ_HEADERS, "Accept": "text/html,application/xhtml+xml"},
        ) as resp:
            if resp.status != 200:
                return None
            html = await resp.text()
            base_url = str(resp.url) if resp.url else url
    except Exception as exc:
        log.warning("fallback image %s: %s", url, exc)
        return None
    return _extract_page_image(html, base_url)

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

_PAYWALL_MARKERS: Dict[str, List[str]] = {
    "clarin.com": ["class=\"paywall", "paywall_content", "suscriptor", "suscriptores"],
}


def _paywall_normalized_domain(url: str) -> str:
    dom = domain_of(url)
    if dom.startswith("www."):
        dom = dom[4:]
    return dom


def _clarin_amp_url(url: str) -> str:
    if "clarin.com" not in domain_of(url):
        return url
    sep = "&" if "?" in url else "?"
    if "output=amp" in url:
        return url
    return f"{url}{sep}output=amp"


async def _clarin_is_paywalled(session: ClientSession, url: str) -> bool:
    amp_url = _clarin_amp_url(url)
    text = await fetch_text(session, amp_url, timeout=ClientTimeout(total=10))
    if not text:
        return False
    low = text.lower()
    return any(marker in low for marker in _PAYWALL_MARKERS["clarin.com"])


PAYWALL_CHECKERS: Dict[str, Callable[[ClientSession, str], Awaitable[bool]]] = {
    "clarin.com": _clarin_is_paywalled,
}


def _paywall_friendly_link(url: str) -> str:
    dom = _paywall_normalized_domain(url)
    if dom == "clarin.com":
        return _clarin_amp_url(url)
    return url


def _mentions_argentina(title: str, desc: Optional[str], domain: Optional[str] = None) -> bool:
    parts = [title or ""]
    if desc:
        parts.append(desc)
    text = " ".join(parts)
    normalized = unicodedata.normalize("NFKD", text)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    low = normalized.lower()
    keywords = [
        "argentina",
        "argentino",
        "argentinos",
        "argentinas",
        "pais",
        "país",
        "nacional",
        "nacion",
        "nación",
    ]
    has_keyword = any(k in low for k in keywords)

    if domain:
        norm_dom = domain[4:] if domain.startswith("www.") else domain
        if norm_dom in NATIONAL_NEWS_DOMAINS:
            return True
        return has_keyword

    return has_keyword


async def fetch_rss_entries(session: ClientSession, limit: int = 5) -> List[NewsItem]:
    global NEWS_HISTORY, NEWS_CACHE
    now_ts = time()
    _prune_news_history(now_ts)
    today = datetime.utcfromtimestamp(now_ts).date().isoformat()
    target_limit = max(limit, 5)
    fallback_generic: List[NewsItem] = [
        ("Mercados: sin novedades relevantes", "https://www.cronista.com/finanzas-mercados/", None),
        ("Actividad: esperando datos de inflación", "https://www.baenegocios.com/economia/", None),
        ("Consumo: expectativa por ventas minoristas", "https://www.perfil.com/economia", None),
        ("Créditos: panorama de tasas y costos", "https://www.infobae.com/economia/", None),
        ("Comercio exterior: dinámica de importaciones", "https://www.pagina12.com.ar/seccion/economia", None),
    ]

    async def _enrich_with_fallback_images(items: List[NewsItem]) -> List[NewsItem]:
        missing = [(idx, entry) for idx, entry in enumerate(items) if len(entry) < 3 or not entry[2]]
        if not missing:
            return items

        to_fetch = missing[:6]
        fetched = await asyncio.gather(*(_fallback_image_from_url(session, entry[1]) for _, entry in to_fetch))
        enriched: List[NewsItem] = list(items)
        changed = False
        used_images: Set[str] = {entry[2] for entry in items if len(entry) >= 3 and entry[2]}
        for (idx, entry), img in zip(to_fetch, fetched):
            if img and img not in used_images:
                enriched[idx] = (entry[0], entry[1], img)
                used_images.add(img)
                changed = True
        return enriched if changed else items

    if NEWS_CACHE.get("date") == today and NEWS_CACHE.get("items"):
        cached_items = _dedup_news_items(NEWS_CACHE.get("items", []), limit=target_limit)
        # Reutilizar solo si el cache tiene suficientes ítems y al menos una nota real
        cache_has_articles = any(_is_probably_article_url(item[1]) for item in cached_items)
        if cache_has_articles and len(cached_items) >= target_limit:
            if len(cached_items) != len(NEWS_CACHE.get("items", [])):
                NEWS_CACHE["items"] = cached_items
                await save_state()
            cached_items = await _enrich_with_fallback_images(cached_items)
            if cached_items != NEWS_CACHE.get("items"):
                NEWS_CACHE["items"] = cached_items
                await save_state()
            return cached_items[:limit]
        NEWS_CACHE.clear()
    history_stems: Set[str] = {stem for stem, _ in NEWS_HISTORY}
    raw_entries: List[RawNewsEntry] = []
    extended_used = False

    async def _collect_feed_entries(feed_urls: Iterable[str]) -> List[RawNewsEntry]:
        collected: List[RawNewsEntry] = []
        for url in feed_urls:
            try:
                xml = await fetch_text(
                    session,
                    url,
                    headers={"Accept": "application/rss+xml, application/atom+xml, */*"},
                )
            except RuntimeError as exc:
                log.error(
                    "RSS fetch error",
                    extra={"url": url, "event": "rss_fetch_error", "error": str(exc)},
                )
                continue
            except SourceSuspendedError as exc:
                log.error(
                    "RSS fetch error",
                    extra={"url": url, "event": "rss_fetch_error", "error": str(exc)},
                )
                continue
            except Exception as exc:
                log.error(
                    "RSS fetch error",
                    extra={"url": url, "event": "rss_fetch_error", "error": str(exc)},
                )
                continue
            if not xml:
                continue
            try:
                collected.extend(_parse_feed_entries(xml))
            except Exception as e:
                log.error(
                    "RSS parse error",
                    extra={"url": url, "event": "rss_parse_error", "error": str(e)},
                )
        return collected

    raw_entries.extend(await _collect_feed_entries(RSS_FEEDS))

    entries_meta: Dict[str, Tuple[str, Optional[str], Optional[str]]] = {}
    seen_titles: Set[str] = set()
    seen_stems: Set[str] = set()
    seen_links: Set[str] = set()

    def _ingest_entries(entries: Iterable[RawNewsEntry]) -> None:
        for title, link, desc, img in entries:
            if not link.startswith("http"):
                continue
            dom = domain_of(link)
            if dom in GOOGLE_NEWS_DOMAINS:
                continue
            if not _is_probably_article_url(link):
                continue
            if _is_dollar_related(title, desc):
                continue
            if not _is_economic_relevant(title, desc):
                continue
            norm_title = _normalize_news_title(title)
            stem_key = _news_dedup_key(title)
            clean_link = _canonical_news_link(link)
            if (
                norm_title in seen_titles
                or stem_key in seen_stems
                or clean_link in seen_links
                or stem_key in history_stems
            ):
                continue
            seen_titles.add(norm_title)
            seen_stems.add(stem_key)
            seen_links.add(clean_link)
            entries_meta[link] = (title, desc, img)

    _ingest_entries(raw_entries)

    if len(entries_meta) < target_limit:
        extended_entries = await _collect_feed_entries(RSS_FEEDS_EXTENDED)
        if extended_entries:
            extended_used = True
            raw_entries.extend(extended_entries)
            _ingest_entries(extended_entries)

    if not entries_meta:
        if NEWS_CACHE.get("items"):
            cached = await _enrich_with_fallback_images(NEWS_CACHE.get("items", []))
            if cached != NEWS_CACHE.get("items"):
                NEWS_CACHE["items"] = cached
                await save_state()
            return cached[:limit]
        if raw_entries:
            deduped_raw = _dedup_news_items(
                [
                    (entry[0], entry[1], entry[3] if len(entry) > 3 else None)
                    for entry in raw_entries
                ],
                limit=target_limit,
            )
            deduped_raw = await _enrich_with_fallback_images(deduped_raw)
            NEWS_CACHE = {"date": today, "items": deduped_raw[:target_limit]}
            await save_state()
            return NEWS_CACHE["items"][:limit]
        enriched_generic = await _enrich_with_fallback_images(fallback_generic[:target_limit])
        NEWS_CACHE = {"date": today, "items": enriched_generic}
        await save_state()
        return NEWS_CACHE["items"][:limit]

    def _build_scored_entries() -> List[Dict[str, Any]]:
        scored_local: List[Dict[str, Any]] = []
        for link, (title, desc, img) in entries_meta.items():
            dom = domain_of(link)
            norm_dom = dom[4:] if dom.startswith("www.") else dom
            mentions_ar = _mentions_argentina(title, desc, norm_dom)
            is_national = norm_dom in NATIONAL_NEWS_DOMAINS
            if not is_national and not mentions_ar:
                continue
            scored_local.append(
                {
                    "title": title,
                    "link": link,
                    "desc": desc,
                    "score": _score_title(title),
                    "domain": norm_dom,
                    "is_national": is_national,
                    "mentions": mentions_ar,
                    "category": _news_category_for(title, desc),
                    "image": img,
                }
            )
        scored_local.sort(
            key=lambda item: (
                1 if item["is_national"] else 0,
                1 if item["mentions"] else 0,
                item["score"],
                item["title"].lower(),
            ),
            reverse=True,
        )
        return scored_local

    scored = _build_scored_entries()

    picked: List[NewsItem] = []
    domain_counts: Dict[str, int] = {}
    paywall_cache: Dict[str, bool] = {}
    picked_titles: Set[str] = set()
    picked_stems: Set[str] = set()
    picked_links: Set[str] = set()
    filled_categories: Set[str] = set()

    def _available_categories(scored_entries: List[Dict[str, Any]]) -> Set[str]:
        return {
            entry["category"]
            for entry in scored_entries
            if entry.get("category") in NEWS_CATEGORIES_ORDER
        }

    def _missing_categories() -> List[str]:
        return [cat for cat in NEWS_CATEGORIES_ORDER if cat not in filled_categories]

    def _already_picked(title: str, link: str) -> bool:
        norm_title = _normalize_news_title(title)
        stem_key = _news_dedup_key(title)
        clean_link = _canonical_news_link(link)
        return (
            norm_title in picked_titles
            or stem_key in picked_stems
            or clean_link in picked_links
        )

    def _register_pick(title: str, link: str, domain: str) -> None:
        norm_title = _normalize_news_title(title)
        stem_key = _news_dedup_key(title)
        clean_link = _canonical_news_link(link)
        picked_titles.add(norm_title)
        picked_stems.add(stem_key)
        picked_links.add(clean_link)
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

    async def is_paywalled(link: str, domain: str) -> bool:
        ndom = _paywall_normalized_domain(domain)
        if ndom not in PAYWALL_CHECKERS:
            return False
        if link in paywall_cache:
            return paywall_cache[link]
        checker = PAYWALL_CHECKERS[ndom]
        try:
            result = await checker(session, link)
        except Exception as exc:
            log.debug("paywall check %s failed: %s", link, exc)
            result = False
        paywall_cache[link] = result
        return result

    async def pick_category(category: str, *, domain_cap: Optional[int], allow_paywall: bool) -> bool:
        for entry in scored:
            if entry["category"] != category:
                continue
            if len(picked) >= target_limit:
                break
            title = entry["title"]
            link = entry["link"]
            dom = entry["domain"]
            img = entry.get("image")
            cat = category or entry.get("category")
            if _already_picked(title, link):
                continue
            if domain_cap is not None and domain_counts.get(dom, 0) >= domain_cap:
                continue
            if not allow_paywall and await is_paywalled(link, dom):
                continue
            picked.append((title, _paywall_friendly_link(link), img))
            _register_pick(title, link, dom)
            NEWS_HISTORY.append((_news_dedup_key(title), now_ts))
            filled_categories.add(category)
            return True
        return False

    async def pick_remaining(*, domain_cap: Optional[int], allow_paywall: bool) -> None:
        for entry in scored:
            if len(picked) >= target_limit:
                break
            title = entry["title"]
            link = entry["link"]
            dom = entry["domain"]
            img = entry.get("image")
            if _already_picked(title, link):
                continue
            if domain_cap is not None and domain_counts.get(dom, 0) >= domain_cap:
                continue
            if not allow_paywall and await is_paywalled(link, dom):
                continue
            picked.append((title, _paywall_friendly_link(link), img))
            _register_pick(title, link, dom)
            NEWS_HISTORY.append((_news_dedup_key(title), now_ts))

    async def fill_categories(scored_entries: List[Dict[str, Any]]) -> bool:
        picked_any = False
        for domain_cap in [1, 2, None]:
            for allow_paywall in [False, True]:
                for category in NEWS_CATEGORIES_ORDER:
                    if len(picked) >= target_limit:
                        break
                    if category in filled_categories:
                        continue
                    if await pick_category(
                        category,
                        domain_cap=domain_cap,
                        allow_paywall=allow_paywall,
                    ):
                        picked_any = True
                if len(filled_categories) == len(NEWS_CATEGORIES_ORDER) or len(picked) >= target_limit:
                    break
            if len(filled_categories) == len(NEWS_CATEGORIES_ORDER) or len(picked) >= target_limit:
                break
        return picked_any

    await fill_categories(scored)

    while _missing_categories() and not extended_used:
        missing = _missing_categories()
        extended_entries = await _collect_feed_entries(RSS_FEEDS_EXTENDED)
        extended_used = True
        if extended_entries:
            log.info(
                "Noticias: extendiendo fuentes para cubrir %s (nuevos=%d)",
                missing,
                len(extended_entries),
            )
            raw_entries.extend(extended_entries)
            _ingest_entries(extended_entries)
            scored = _build_scored_entries()
            await fill_categories(scored)
        else:
            break

    missing_after_extension = _missing_categories()
    if missing_after_extension:
        log.info(
            "Noticias: categorías sin cubrir %s (candidatas=%s, picks=%d, total=%d)",
            missing_after_extension,
            sorted(_available_categories(scored)),
            len(picked),
            len(scored),
        )

    if len(picked) < target_limit:
        for domain_cap in [1, 2, None]:
            for allow_paywall in [False, True]:
                await pick_remaining(domain_cap=domain_cap, allow_paywall=allow_paywall)
                if len(picked) >= target_limit:
                    break
            if len(picked) >= target_limit:
                break
    if len(picked) < target_limit:
        fallback_scored: List[Dict[str, Any]] = []
        for title, link, desc, img in raw_entries:
            if not link.startswith("http"):
                continue
            if not _is_probably_article_url(link):
                continue
            if _already_picked(title, link):
                continue
            if _is_dollar_related(title, desc):
                continue
            if not _is_economic_relevant(title, desc):
                continue
            norm_dom = domain_of(link)
            norm_dom = norm_dom[4:] if norm_dom.startswith("www.") else norm_dom
            fallback_scored.append(
                {
                    "title": title,
                    "link": link,
                    "score": _score_title(title),
                    "domain": norm_dom,
                    "is_national": norm_dom in NATIONAL_NEWS_DOMAINS,
                    "mentions": _mentions_argentina(title, desc, norm_dom),
                    "image": img,
                }
            )

        fallback_scored.sort(
            key=lambda item: (
                1 if item["is_national"] else 0,
                1 if item["mentions"] else 0,
                item["score"],
                item["title"].lower(),
            ),
            reverse=True,
        )

        for entry in fallback_scored:
            if len(picked) >= target_limit:
                break
            title = entry["title"]
            link = entry["link"]
            dom = entry["domain"]
            img = entry.get("image")
            if await is_paywalled(link, dom):
                continue
            picked.append((title, _paywall_friendly_link(link), img))
            _register_pick(title, link, dom)
            NEWS_HISTORY.append((_news_dedup_key(title), now_ts))

    if len(picked) < target_limit:
        # Último intento: relajar límites de dominio y tema con las fuentes ya
        # procesadas, evitando paywalls cuando sea posible.
        for entry in scored:
            if len(picked) >= target_limit:
                break
            title = entry["title"]
            link = entry["link"]
            dom = entry["domain"]
            img = entry.get("image")
            if _already_picked(title, link):
                continue
            if await is_paywalled(link, dom):
                continue
            picked.append((title, _paywall_friendly_link(link), img))
            _register_pick(title, link, dom)
            NEWS_HISTORY.append((_news_dedup_key(title), now_ts))

    deduped_final = _dedup_news_items(picked, target_limit)
    if len(deduped_final) < target_limit:
        for title, link, img in fallback_generic:
            if len(deduped_final) >= target_limit:
                break
            existing_keys = {(t, l) for t, l, _ in deduped_final}
            if (title, link) not in existing_keys:
                deduped_final.append((title, link, img))

    deduped_final = await _enrich_with_fallback_images(deduped_final)
    NEWS_CACHE = {"date": today, "items": deduped_final[:target_limit]}
    await save_state()
    return NEWS_CACHE["items"][:limit]

def _short_title(text: str, limit: int = 32) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _short_link(link: str, max_len: int = 48) -> str:
    try:
        parsed = urlparse(link)
    except Exception:
        return link

    host = parsed.netloc or link
    host = host[4:] if host.startswith("www.") else host
    path = (parsed.path or "").strip()
    clean_path = re.sub(r"/+", "/", path)
    preview = f"{host}{clean_path}"
    if parsed.query:
        preview = f"{preview}?…"
    if len(preview) > max_len:
        preview = preview[: max_len - 1].rstrip("/") + "…"
    return preview or link


def _format_news_item(title: str, link: str) -> str:
    safe_title = _html.escape(title)
    safe_link = _html.escape(link, True)
    short_link = _html.escape(_short_link(link))
    return f"<b>{safe_title}</b>\n<a href=\"{safe_link}\">🔗 {short_link}</a>\n{_impact_lines(title)}"


def _build_news_layout(news: List[NewsItem]) -> Tuple[str, Optional[InlineKeyboardMarkup], List[Tuple[str, Optional[str]]]]:
    header = "<b>📰 Noticias</b>"
    if not news:
        return header, None, []

    body_lines: List[Tuple[str, Optional[str]]] = []
    for entry in news:
        title, link = entry[0], entry[1]
        img = entry[2] if len(entry) >= 3 else None
        fallback_img = entry[3] if len(entry) >= 4 else None
        body_lines.append((_format_news_item(title, link), img or fallback_img))
    return header, None, body_lines


def format_news_block(news: List[NewsItem]) -> Tuple[str, Optional[InlineKeyboardMarkup]]:
    header, markup, body_lines = _build_news_layout(news)
    body = "\n\n".join(text for text, _ in body_lines) if body_lines else "—"
    return f"{header}\n{body}", markup

# ============================ FORMATS & RANKINGS ============================

def _label_long(sym: str) -> str: return label_with_currency(sym)
def _label_short(sym: str) -> str:
    if sym.endswith(".BA"): return f"{NAME_ABBR.get(sym, sym)} ({sym[:-3]})"
    return label_with_currency(sym)

def format_dolar_panels(d: Dict[str, Dict[str, Any]]) -> Tuple[str, str]:
    fecha = None
    for row in d.values():
        f = row.get("fecha")
        if f:
            fecha = parse_iso_ddmmyyyy(f)

    header = "<b>💵 Dólares</b>" + (f" <i>Actualizado: {fecha}</i>" if fecha else "")
    order = [
        ("oficial", "Oficial"),
        ("mayorista", "Mayorista"),
        ("blue", "Blue"),
        ("mep", "MEP"),
        ("ccl", "CCL"),
        ("cripto", "Cripto"),
        ("tarjeta", "Tarjeta"),
    ]

    def _fmt_var(val: Optional[float]) -> str:
        if val is None:
            return f"{'—':>12}"
        arrow = _circle_indicator(val, up_icon="🔴", down_icon="🟢")
        num = f"{val:+.2f}%"
        return f"{arrow} {num:>8}"

    compra_lines = [
        header,
        "<b>📥 Compra</b>",
        "<pre>Tipo         Compra        Var. día</pre>",
    ]
    venta_lines = [
        header,
        "<b>📤 Venta</b>",
        "<pre>Tipo         Venta         Var. día</pre>",
    ]
    compra_rows: List[str] = []
    venta_rows: List[str] = []

    for k, label in order:
        row = d.get(k)
        if not row:
            continue
        # Los valores de venta y compra vienen invertidos en la fuente, por eso se muestran cruzados
        compra_val = row.get("venta")
        venta_val = row.get("compra")
        var_val = row.get("variation")

        compra = fmt_money_ars(compra_val) if compra_val is not None else "—"
        venta = fmt_money_ars(venta_val) if venta_val is not None else "—"
        var_txt = _fmt_var(var_val)

        compra_rows.append(f"<pre>{label:<12}{compra:>12} {var_txt}</pre>")
        venta_rows.append(f"<pre>{label:<12}{venta:>12} {var_txt}</pre>")

    compra_msg = "\n".join(compra_lines + compra_rows)
    venta_msg = "\n".join(venta_lines + venta_rows)

    return compra_msg, venta_msg


def format_dolar_message(d: Dict[str, Dict[str, Any]]) -> str:
    """Build a compact FX block combining purchase and sale tables."""

    compra_msg, venta_msg = format_dolar_panels(d)
    return "\n\n".join([compra_msg, venta_msg])

def _ret_label(base: str, currency_label: Optional[str]) -> str:
    if not currency_label:
        return base
    return f"{base} ({currency_label})"

def format_top3_table(
    title: str,
    fecha: Optional[str],
    rows_syms: List[str],
    retmap: Dict[str, Dict[str, Optional[float]]],
    *,
    key_suffix: str = "",
    currency_label: Optional[str] = None,
) -> str:
    head = f"<b>{title}</b>" + (f" <i>Últ. Dato: {fecha}</i>" if fecha else "")
    header = (
        f"<pre>Rank Empresa (Ticker)        {_ret_label('6M', currency_label):>10}"
        f"{_ret_label('3M', currency_label):>10}{_ret_label('1M', currency_label):>10}</pre>"
    )
    lines = [head, header]
    out = []
    for idx, sym in enumerate(rows_syms[:3], start=1):
        d = retmap.get(sym, {})
        def _val_with_fallback(key: str) -> Optional[float]:
            primary = d.get(f"{key}{key_suffix}")
            if primary is None and key_suffix == "_ars":
                return d.get(key)
            return primary

        p6_val = _val_with_fallback("6m")
        p3_val = _val_with_fallback("3m")
        p1_val = _val_with_fallback("1m")
        p6 = pct(p6_val, 2) if p6_val is not None else "—"
        p3 = pct(p3_val, 2) if p3_val is not None else "—"
        p1 = pct(p1_val, 2) if p1_val is not None else "—"
        label = pad(_label_short(sym), 28)
        c6 = center_text(p6, 10); c3 = center_text(p3, 10); c1 = center_text(p1, 10)
        l = f"{idx:<4} {label}{c6}{c3}{c1}"
        out.append(f"<pre>{l}</pre>")
    if not out: out.append("<pre>—</pre>")
    return "\n".join([lines[0], lines[1]] + out)

ProjectionRange = Tuple[Optional[float], Optional[float], Optional[float]]
PROJ_RANGE_SHRINK = 0.6
PROJ_RANGE_LIMITS = {WINDOW_DAYS[3]: (-50.0, 80.0), WINDOW_DAYS[6]: (-70.0, 120.0)}
PROJ_RANGE_PCTL = 0.674

def _format_projection_range(proj: ProjectionRange, nd: int = 1) -> str:
    center, low, high = proj
    if center is None:
        return "—"
    center_txt = pct(center, nd)
    if low is None or high is None:
        return center_txt
    return f"{center_txt} ({pct(low, nd)} a {pct(high, nd)})"

def _projection_bounds(proj: ProjectionRange) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    center, low, high = proj
    if center is None:
        return None, None, None
    if low is None or high is None:
        return center, center, center
    return center, low, high

def _projection_percentile(values: List[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    k = (len(values) - 1) * percentile / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return values[int(k)]
    return values[f] * (c - k) + values[c] * (k - f)


def _projection_recent_samples(m: Dict[str, Optional[float]], horizon_days: int) -> List[float]:
    horizon_months = max(1, round(horizon_days / WINDOW_DAYS[1]))
    samples: List[float] = []
    for months in WINDOW_MONTHS:
        key = f"{months}m"
        val = m.get(key)
        if val is None:
            continue
        try:
            base = 1.0 + float(val) / 100.0
            if base <= 0:
                continue
            scaled = (base ** (horizon_months / months) - 1.0) * 100.0
            samples.append(scaled)
        except Exception:
            continue
    return samples


def _projection_clip(value: float, floor: float, cap: float) -> float:
    return max(floor, min(cap, value))


def _projection_range_from_center(
    center: Optional[float],
    m: Dict[str, Optional[float]],
    horizon_days: int,
) -> ProjectionRange:
    if center is None:
        return None, None, None
    try:
        center_f = float(center)
    except Exception:
        return None, None, None

    low_model = high_model = None
    vol_ann = m.get("vol_ann")
    if vol_ann is not None:
        try:
            sigma_h = (float(vol_ann) / 100.0) * math.sqrt(horizon_days / 252.0)
            if center_f > -99.0:
                center_log = math.log1p(center_f / 100.0)
                low_model = (math.exp(center_log - PROJ_RANGE_PCTL * sigma_h) - 1.0) * 100.0
                high_model = (math.exp(center_log + PROJ_RANGE_PCTL * sigma_h) - 1.0) * 100.0
        except Exception:
            low_model = high_model = None

    samples = _projection_recent_samples(m, horizon_days)
    low_hist = _projection_percentile(samples, 25.0) if samples else None
    high_hist = _projection_percentile(samples, 75.0) if samples else None

    if low_model is not None and low_hist is not None and high_model is not None and high_hist is not None:
        low_raw = (low_model + low_hist) / 2.0
        high_raw = (high_model + high_hist) / 2.0
    elif low_model is not None and high_model is not None:
        low_raw = low_model
        high_raw = high_model
    elif low_hist is not None and high_hist is not None:
        low_raw = low_hist
        high_raw = high_hist
    else:
        return center_f, None, None

    floor, cap = PROJ_RANGE_LIMITS.get(horizon_days, (-80.0, 150.0))
    center_clipped = _projection_clip(center_f, floor, cap)
    low = center_clipped + (low_raw - center_clipped) * PROJ_RANGE_SHRINK
    high = center_clipped + (high_raw - center_clipped) * PROJ_RANGE_SHRINK
    low = _projection_clip(low, floor, cap)
    high = _projection_clip(high, floor, cap)
    if low > high:
        low, high = high, low
    return center_clipped, low, high


def _projection_range(
    m: Dict[str, Optional[float]],
    horizon: int,
    mu_override: Optional[float] = None,
) -> ProjectionRange:
    mu = _expected_daily_return(m) if mu_override is None else mu_override
    mu_h = mu * horizon
    center = (math.exp(mu_h) - 1.0) * 100.0
    return _projection_range_from_center(center, m, horizon)

def format_proj_dual(title: str, fecha: Optional[str], rows: List[Tuple[str, ProjectionRange, ProjectionRange]]) -> str:
    head = f"<b>{title}</b>" + (f" <i>Últ. Dato: {fecha}</i>" if fecha else "")
    sub = "<i>Proy. 3M (corto) y Proy. 6M (medio) — rango estimado P25–P75</i>"
    lines = [head, sub, "<pre>Rank Empresa (Ticker)     Proy. 3M (rango est.)    Proy. 6M (rango est.)</pre>"]
    out = []
    if not rows: out.append("<pre>—</pre>")
    else:
        for idx, (sym, p3v, p6v) in enumerate(rows[:5], start=1):
            p3 = _format_projection_range(p3v, 1)
            p6 = _format_projection_range(p6v, 1)
            label = pad(_label_short(sym), 28)
            c3 = center_text(p3, 26); c6 = center_text(p6, 26)
            l = f"{idx:<4} {label}{c3}{c6}"
            out.append(f"<pre>{l}</pre>")
    return "\n".join(lines + out)

def _nz(x: Optional[float], fb: float) -> float: return float(x) if x is not None else fb

def _expected_daily_return(m: Dict[str, Optional[float]]) -> float:
    components: List[Tuple[float, float]] = []
    weights = {1: 0.5, 3: 0.3, 6: 0.2}
    for months in WINDOW_MONTHS:
        key = f"{months}m"
        weight = weights.get(months, 0.0)
        if weight <= 0:
            continue
        val = m.get(key)
        if val is None:
            continue
        try:
            lr = math.log1p(float(val) / 100.0) / WINDOW_DAYS[months]
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

def calibrate_projection(raw: float, horizon: str) -> float:
    coeffs = PROJ_CALIBRATION.get(horizon)
    if not coeffs:
        return raw
    try:
        return coeffs["a"] + coeffs["b"] * raw
    except Exception:
        return raw


def projection_3m_raw(m: Dict[str, Optional[float]]) -> float:
    mu = _expected_daily_return(m)
    return (math.exp(mu * WINDOW_DAYS[3]) - 1.0) * 100.0


def projection_6m_raw(m: Dict[str, Optional[float]]) -> float:
    mu = _expected_daily_return(m)
    vol_ann = m.get("vol_ann")
    if vol_ann is not None:
        try:
            penalty = (float(vol_ann) / 100.0) ** 2 * 0.0005
            mu -= penalty
        except Exception:
            pass
    return (math.exp(mu * WINDOW_DAYS[6]) - 1.0) * 100.0


def projection_3m(m: Dict[str, Optional[float]]) -> ProjectionRange:
    raw = projection_3m_raw(m)
    center = calibrate_projection(raw, "3m")
    return _projection_range_from_center(center, m, WINDOW_DAYS[3])


def projection_6m(m: Dict[str, Optional[float]]) -> ProjectionRange:
    raw = projection_6m_raw(m)
    center = calibrate_projection(raw, "6m")
    return _projection_range_from_center(center, m, WINDOW_DAYS[6])


def _format_projection_date(ts: float) -> str:
    try:
        return datetime.fromtimestamp(ts, TZ).strftime("%Y-%m-%d")
    except Exception:
        return ""


def _trading_days_between(start_date: date, end_date: date) -> int:
    if end_date <= start_date:
        return 0
    days = 0
    cur = start_date + timedelta(days=1)
    while cur <= end_date:
        if cur.weekday() < 5:
            days += 1
        cur += timedelta(days=1)
    return days


def _build_projection_batch_id(horizon: int, created_at: float) -> str:
    stamp = int(created_at)
    date_str = _format_projection_date(created_at)
    return f"{date_str}-{stamp}-{horizon}"


def _record_projection_batch_from_rank(
    rows: List[Tuple[str, ProjectionRange, ProjectionRange]],
    metrics_by_symbol: Dict[str, Dict[str, Optional[float]]],
) -> bool:
    if not rows:
        return False
    created_at = time()
    created_date = _format_projection_date(created_at)
    base_prices: Dict[str, float] = {}
    projections_3m: Dict[str, float] = {}
    projections_6m: Dict[str, float] = {}
    for sym, p3, p6 in rows:
        c3, _, _ = _projection_bounds(p3)
        c6, _, _ = _projection_bounds(p6)
        if c3 is None or c6 is None:
            continue
        base_px = metric_last_price(metrics_by_symbol.get(sym, {}))
        if base_px is None:
            continue
        base_prices[sym] = base_px
        projections_3m[sym] = float(c3)
        projections_6m[sym] = float(c6)
    if not base_prices:
        return False
    for horizon, projections in ((WINDOW_DAYS[3], projections_3m), (WINDOW_DAYS[6], projections_6m)):
        if not projections:
            continue
        batch_id = _build_projection_batch_id(horizon, created_at)
        PROJECTION_BATCHES.append(
            {
                "batch_id": batch_id,
                "created_at": created_at,
                "created_date": created_date,
                "horizon": horizon,
                "symbols": sorted(projections.keys()),
                "predictions": dict(projections),
                "base_prices": {sym: base_prices[sym] for sym in projections},
                "evaluated": False,
            }
        )
        for sym, projection in projections.items():
            PROJECTION_RECORDS.append(
                {
                    "symbol": sym,
                    "horizon": horizon,
                    "base_price": base_prices[sym],
                    "projection": projection,
                    "created_at": created_at,
                    "created_date": created_date,
                    "batch_id": batch_id,
                    "evaluated": False,
                }
            )
    return True


def _rank_values(values: Dict[str, float]) -> Dict[str, float]:
    items = sorted(values.items(), key=lambda x: x[1], reverse=True)
    ranks: Dict[str, float] = {}
    i = 0
    while i < len(items):
        j = i
        val = items[i][1]
        while j < len(items) and items[j][1] == val:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[items[k][0]] = avg_rank
        i = j
    return ranks


def _spearman_correlation(
    predicted: Dict[str, float], actual: Dict[str, float]
) -> Optional[float]:
    common = [sym for sym in predicted.keys() if sym in actual]
    n = len(common)
    if n < 2:
        return None
    pred_vals = {sym: float(predicted[sym]) for sym in common}
    act_vals = {sym: float(actual[sym]) for sym in common}
    rank_pred = _rank_values(pred_vals)
    rank_act = _rank_values(act_vals)
    diff_sq = sum((rank_pred[s] - rank_act[s]) ** 2 for s in common)
    return 1.0 - (6.0 * diff_sq) / (n * (n * n - 1))


def _update_projection_record(
    batch_id: str,
    symbol: str,
    *,
    evaluated_at: float,
    actual_price: float,
    actual_return: float,
    error_abs: float,
    direction_hit: bool,
) -> None:
    for record in PROJECTION_RECORDS:
        if record.get("batch_id") != batch_id or record.get("symbol") != symbol:
            continue
        record.update(
            {
                "evaluated": True,
                "evaluated_at": evaluated_at,
                "actual_price": actual_price,
                "actual_return": actual_return,
                "error_abs": error_abs,
                "direction_hit": direction_hit,
            }
        )


async def _evaluate_projection_batches() -> int:
    if not PROJECTION_BATCHES:
        return 0
    today = datetime.now(TZ).date()
    matured: List[Dict[str, Any]] = []
    for batch in PROJECTION_BATCHES:
        if batch.get("evaluated"):
            continue
        created_date_str = batch.get("created_date")
        created_at = batch.get("created_at")
        if isinstance(created_date_str, str) and created_date_str:
            try:
                created_date = datetime.strptime(created_date_str, "%Y-%m-%d").date()
            except Exception:
                created_date = None
        else:
            created_date = None
        if created_date is None and created_at is not None:
            try:
                created_date = datetime.fromtimestamp(float(created_at), TZ).date()
            except Exception:
                created_date = None
        if created_date is None:
            continue
        try:
            horizon = int(batch.get("horizon"))
        except Exception:
            continue
        if _trading_days_between(created_date, today) >= horizon:
            matured.append(batch)
    if not matured:
        return 0

    symbols = sorted(
        {sym for batch in matured for sym in batch.get("symbols", []) if isinstance(sym, str)}
    )
    if not symbols:
        return 0

    async with ClientSession() as session:
        metrics_by_symbol, _ = await metrics_for_symbols(session, symbols)

    now_ts = time()
    for batch in matured:
        predictions = batch.get("predictions", {})
        base_prices = batch.get("base_prices", {})
        actual_returns: Dict[str, float] = {}
        errors: List[float] = []
        hit_count = 0
        count = 0
        for sym, proj in predictions.items():
            base_px = base_prices.get(sym)
            if base_px is None or base_px == 0:
                continue
            current_px = metric_last_price(metrics_by_symbol.get(sym, {}))
            if current_px is None:
                continue
            actual_return = (float(current_px) / float(base_px) - 1.0) * 100.0
            actual_returns[sym] = actual_return
            try:
                proj_val = float(proj)
            except Exception:
                continue
            error_abs = abs(actual_return - proj_val)
            errors.append(error_abs)
            direction_hit = (actual_return >= 0) == (proj_val >= 0)
            if direction_hit:
                hit_count += 1
            count += 1
            _update_projection_record(
                batch.get("batch_id", ""),
                sym,
                evaluated_at=now_ts,
                actual_price=float(current_px),
                actual_return=actual_return,
                error_abs=error_abs,
                direction_hit=direction_hit,
            )

        mae = (sum(errors) / count) if count else None
        hit_rate = (hit_count / count) if count else None
        spearman = _spearman_correlation(
            {k: float(v) for k, v in predictions.items() if k in actual_returns},
            actual_returns,
        )
        batch.update(
            {
                "evaluated": True,
                "evaluated_at": now_ts,
                "actual_returns": actual_returns,
                "mae": mae,
                "hit_rate": hit_rate,
                "hit_count": hit_count,
                "count": count,
                "spearman": spearman,
            }
        )
    return len(matured)


def _summarize_projection_performance(batches: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    total_count = 0
    total_abs_error = 0.0
    total_hits = 0
    spearman_vals: List[float] = []
    for batch in batches:
        count = batch.get("count")
        mae = batch.get("mae")
        hit_count = batch.get("hit_count")
        if isinstance(count, int) and count > 0 and isinstance(mae, (int, float)):
            total_count += count
            total_abs_error += float(mae) * count
        if isinstance(hit_count, int) and hit_count >= 0 and isinstance(count, int) and count > 0:
            total_hits += hit_count
        spearman = batch.get("spearman")
        if isinstance(spearman, (int, float)):
            spearman_vals.append(float(spearman))
    mae = (total_abs_error / total_count) if total_count else None
    hit_rate = (total_hits / total_count) if total_count else None
    spearman_avg = (sum(spearman_vals) / len(spearman_vals)) if spearman_vals else None
    return {
        "count": total_count,
        "mae": mae,
        "hit_rate": hit_rate,
        "spearman": spearman_avg,
    }

RET_SERIES_BASE: List[Tuple[str, str]] = [
    ("6m", "Rend. 6M"),
    ("3m", "Rend. 3M"),
    ("1m", "Rend. 1M"),
]

def _ret_series_order(currency_label: Optional[str] = None, key_suffix: str = "") -> List[Tuple[str, str]]:
    series = []
    for key, label in RET_SERIES_BASE:
        key_out = f"{key}{key_suffix}"
        if currency_label:
            label_out = f"{label} ({currency_label})"
        else:
            label_out = label
        series.append((key_out, label_out))
    return series


async def _rank_top3(
    update: Update,
    symbols: List[str],
    title: str,
    throttle_key: Optional[str] = "rankings:top3",
):
    chat_id = update.effective_chat.id if update.effective_chat else None
    user_id = update.effective_user.id if update.effective_user else None
    if throttle_key and is_throttled(throttle_key, chat_id, user_id, ttl=35):
        await update.effective_message.reply_text("⏳ Consultá de nuevo en unos segundos.")
        return
    async with ClientSession() as session:
        mets, last_ts = await metrics_for_symbols_cached(session, symbols)
        fecha = datetime.fromtimestamp(last_ts, TZ).strftime("%d/%m/%Y") if last_ts else None
        is_cedear = all(sym.upper() in CEDEARS_SET for sym in symbols)
        if is_cedear:
            pairs_ars = sorted(
                [(sym, m.get("6m_ars")) for sym, m in mets.items() if m.get("6m_ars") is not None],
                key=lambda x: x[1],
                reverse=True,
            )
            if not pairs_ars:
                pairs_ars = sorted(
                    [(sym, m.get("6m")) for sym, m in mets.items() if m.get("6m") is not None],
                    key=lambda x: x[1],
                    reverse=True,
                )
            top_syms_ars = [sym for sym, _ in pairs_ars[: RANK_TOP_LIMIT]]
            msg_parts = [
                format_top3_table(title, fecha, top_syms_ars, mets, key_suffix="_ars", currency_label="ARS")
            ]
            pairs_usd = sorted(
                [(sym, m.get("6m_usd")) for sym, m in mets.items() if m.get("6m_usd") is not None],
                key=lambda x: x[1],
                reverse=True,
            )
            if pairs_usd:
                top_syms_usd = [sym for sym, _ in pairs_usd[: RANK_TOP_LIMIT]]
                msg_parts.append(
                    format_top3_table(f"{title} (USD)", fecha, top_syms_usd, mets, key_suffix="_usd", currency_label="USD")
                )
            msg = "\n\n".join(msg_parts)
            pairs = pairs_ars
        else:
            pairs = sorted([(sym, m["6m"]) for sym, m in mets.items() if m.get("6m") is not None], key=lambda x: x[1], reverse=True)
            top_syms = [sym for sym, _ in pairs[: RANK_TOP_LIMIT]]
            msg = format_top3_table(title, fecha, top_syms, mets)
        await update.effective_message.reply_text(
            msg,
            parse_mode=ParseMode.HTML,
            link_preview_options=build_preview_options(),
        )

        if HAS_MPL and pairs:
            chart_rows: List[Tuple[str, List[Optional[float]]]] = []
            series_order = _ret_series_order("ARS", "_ars") if is_cedear else _ret_series_order()
            for sym, _ in pairs[: RANK_TOP_LIMIT]:
                metrics = mets.get(sym, {})
                values: List[Optional[float]] = []
                for key, _ in series_order:
                    raw_val = metrics.get(key)
                    if raw_val is None and is_cedear and key.endswith("_ars"):
                        raw_val = metrics.get(key.replace("_ars", ""))
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
                title=f"{title} — Rendimientos 6/3/1M{' (ARS)' if is_cedear else ''}",
                subtitle=subtitle,
                series_labels=[label for _, label in series_order],
            )
            if img:
                await update.effective_message.reply_photo(photo=img)


async def _rank_proj5(
    update: Update,
    symbols: List[str],
    title: str,
    throttle_key: Optional[str] = "rankings:top5",
):
    chat_id = update.effective_chat.id if update.effective_chat else None
    user_id = update.effective_user.id if update.effective_user else None
    if throttle_key and is_throttled(throttle_key, chat_id, user_id, ttl=35):
        await update.effective_message.reply_text("⏳ Consultá de nuevo en unos segundos.")
        return
    async with ClientSession() as session:
        mets, last_ts = await metrics_for_symbols_cached(session, symbols)
        fecha = datetime.fromtimestamp(last_ts, TZ).strftime("%d/%m/%Y") if last_ts else None
        rows: List[Tuple[str, ProjectionRange, ProjectionRange, Optional[float]]] = []
        history_added = False
        for sym, m in mets.items():
            if m.get("6m") is None:
                continue
            raw3 = projection_3m_raw(m)
            raw6 = projection_6m_raw(m)
            register_projection_history(sym, "3m", raw3, m)
            register_projection_history(sym, "6m", raw6, m)
            history_added = True
            p3 = projection_3m(m)
            p6 = projection_6m(m)
            c6, _, _ = _projection_bounds(p6)
            rows.append((sym, p3, p6, c6))
        rows.sort(key=lambda x: x[3] if x[3] is not None else float("-inf"), reverse=True)
        top_rows = [(sym, p3, p6) for sym, p3, p6, _ in rows[: RANK_PROJ_LIMIT]]
        msg = format_proj_dual(title, fecha, top_rows)
        if history_added:
            asyncio.create_task(save_state())
        await update.effective_message.reply_text(
            msg,
            parse_mode=ParseMode.HTML,
            link_preview_options=build_preview_options(),
        )

        if top_rows:
            recorded = _record_projection_batch_from_rank(top_rows, mets)
            if recorded:
                await save_state()

        if HAS_MPL and top_rows:
            chart_rows = []
            label_rows: List[List[Optional[str]]] = []
            for sym, p3, p6 in top_rows:
                c3, _, _ = _projection_bounds(p3)
                c6, _, _ = _projection_bounds(p6)
                chart_rows.append((_label_short(sym), [c3, c6]))
                label_rows.append([
                    _format_projection_range(p3, 1),
                    _format_projection_range(p6, 1),
                ])
            subtitle = f"Datos al {fecha}" if fecha else None
            img = _bar_image_from_rank(
                chart_rows,
                title=f"{title} — Proyecciones",
                subtitle=subtitle,
                series_labels=["Proy. 3M", "Proy. 6M"],
                value_labels=label_rows,
            )
            if img:
                await update.effective_message.reply_photo(photo=img)

# ============================ COMANDOS / MENÚS ============================

def set_menu_counter(context: ContextTypes.DEFAULT_TYPE, name: str, n: int):
    context.user_data.setdefault("menu_counts", {})[name] = n
async def dec_and_maybe_show(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    name: str,
    show_func,
):
    cnt = context.user_data.get("menu_counts", {}).get(name, 0)
    cnt = max(0, cnt - 1)
    context.user_data["menu_counts"][name] = cnt
    if cnt > 0:
        return await show_func(update, context)

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
            InlineKeyboardButton("💵 Tipos de cambio", callback_data="ECO:DOLAR"),
            InlineKeyboardButton("🏦 Reservas", callback_data="ECO:RESERVAS"),
        ],
        [
            InlineKeyboardButton("📰 Noticias", callback_data="ECO:NOTICIAS"),
            InlineKeyboardButton("📈 Acciones Top 3", callback_data="ACC:TOP3"),
        ],
        [
            InlineKeyboardButton("🏁 Acciones Proyección", callback_data="ACC:TOP5"),
            InlineKeyboardButton("🌎 Cedears Top 3", callback_data="CED:TOP3"),
        ],
        [
            InlineKeyboardButton("🌐 Cedears Proyección", callback_data="CED:TOP5"),
            InlineKeyboardButton("🔔 Mis alertas", callback_data="AL:LIST:0"),
        ],
        [
            InlineKeyboardButton("🧾 Resumen diario", callback_data="ST:SUBS"),
            InlineKeyboardButton("💼 Portafolio", callback_data="PF:MENU"),
        ],
    ]

    await update.effective_message.reply_text(
        intro,
        parse_mode=ParseMode.HTML,
        reply_markup=InlineKeyboardMarkup(kb_rows),
        link_preview_options=build_preview_options(),
    )

async def cmd_dolar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    async with ClientSession() as session:
        data = await get_dolares(session)
    if not data or not _has_fx_data(data):
        await update.effective_message.reply_text(
            "No pude obtener cotizaciones ahora.",
            parse_mode=ParseMode.HTML,
            link_preview_options=build_preview_options(),
        )
        return

    compra_msg, venta_msg = format_dolar_panels(data)
    for msg in (compra_msg, venta_msg):
        await update.effective_message.reply_text(
            msg,
            parse_mode=ParseMode.HTML,
            link_preview_options=build_preview_options(),
        )

def _extract_json_object_with_bands(text: str) -> Optional[Dict[str, Any]]:
    idx = text.find("\"bands\"")
    if idx == -1:
        return None

    start = text.rfind("{", 0, idx)
    if start == -1:
        return None

    depth = 0
    end = None
    for pos in range(start, len(text)):
        ch = text[pos]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = pos + 1
                break

    if end is None:
        return None

    try:
        return json.loads(text[start:end])
    except Exception:
        return None


async def _parse_dolarito_bandas_html(html: str) -> Optional[Dict[str, Any]]:
    search_spaces: List[str] = [html]

    for match in re.finditer(r"bands\\\":\{", html):
        start = max(0, match.start() - 120)
        end = min(len(html), match.end() + 600)
        snippet = html[start:end]
        try:
            decoded = snippet.encode("utf-8").decode("unicode_escape")
            search_spaces.append(decoded)
        except Exception:
            continue

    for text in search_spaces:
        parsed = _extract_json_object_with_bands(text)
        if not isinstance(parsed, dict):
            continue

        bands = parsed.get("bands")
        if not isinstance(bands, dict):
            continue

        data: Dict[str, Any] = {
            "banda_superior": bands.get("upper") or bands.get("upperBand"),
            "banda_inferior": bands.get("lower") or bands.get("lowerBand"),
            "variacion_diaria": (bands.get("dolarMayorista") or {}).get("variation"),
            "variacion_superior": bands.get("upperVariation"),
            "variacion_inferior": bands.get("lowerVariation"),
            "variacion_mensual_superior": bands.get("upperMonthlyVariation") or bands.get("upper_monthly_variation"),
            "variacion_mensual_inferior": bands.get("lowerMonthlyVariation") or bands.get("lower_monthly_variation"),
            "fecha": None,
            "fuente": "Dolarito.ar",
        }

        ts_candidates = [bands.get("timestamp"), parsed.get("timestamp")]
        for ts in ts_candidates:
            try:
                if ts is None:
                    continue
                dt = datetime.fromtimestamp(float(ts) / 1000.0, tz=TZ)
                data["fecha"] = dt.strftime("%Y-%m-%d %H:%M:%S")
                break
            except Exception:
                continue

        return data

    log.warning("No se pudo parsear bandas de Dolarito: sin coincidencias")
    return None


async def _fetch_dolarito_bandas_json(session: ClientSession) -> Optional[Dict[str, Any]]:
    try:
        async with session.get(
            DOLARITO_BANDAS_JSON, headers=REQ_HEADERS, timeout=ClientTimeout(total=10)
        ) as resp:
            if resp.status != 200:
                return None
            raw = await resp.json()
    except Exception:
        return None

    if not isinstance(raw, dict):
        return None

    bands = raw.get("bands") if isinstance(raw.get("bands"), dict) else raw
    data: Dict[str, Any] = {
        "banda_superior": bands.get("upper") or bands.get("upperBand"),
        "banda_inferior": bands.get("lower") or bands.get("lowerBand"),
        "variacion_mensual_superior": bands.get("upperMonthlyVariation")
        or bands.get("upper_monthly_variation"),
        "variacion_mensual_inferior": bands.get("lowerMonthlyVariation")
        or bands.get("lower_monthly_variation"),
        "variacion_diaria": (bands.get("dolarMayorista") or {}).get("variation"),
        "variacion_superior": bands.get("upperVariation"),
        "variacion_inferior": bands.get("lowerVariation"),
        "fecha": None,
        "fuente": "Dolarito.ar (JSON)",
    }

    ts_candidates = [bands.get("timestamp"), raw.get("timestamp")]
    for ts in ts_candidates:
        try:
            if ts is None:
                continue
            dt = datetime.fromtimestamp(float(ts) / 1000.0, tz=TZ)
            data["fecha"] = dt.strftime("%Y-%m-%d %H:%M:%S")
            break
        except Exception:
            continue

    return data

async def _fetch_dolarito_bandas(session: ClientSession) -> Optional[Dict[str, Any]]:
    html = await fetch_text(
        session,
        DOLARITO_BANDAS_HTML,
        headers=REQ_HEADERS,
    )
    if not html:
        return None

    return await _parse_dolarito_bandas_html(html)

async def get_bandas_cambiarias(session: ClientSession) -> Optional[Dict[str, Any]]:
    data = await _fetch_dolarito_bandas(session)

    if not data:
        data = await _fetch_dolarito_bandas_json(session)
    elif not (
        data.get("variacion_mensual_superior") and data.get("variacion_mensual_inferior")
    ):
        json_data = await _fetch_dolarito_bandas_json(session)
        if json_data:
            for key, val in json_data.items():
                if data.get(key) is None and val is not None:
                    data[key] = val
            if json_data.get("variacion_mensual_superior") or json_data.get("variacion_mensual_inferior"):
                data["fuente"] = json_data.get("fuente", data.get("fuente"))
    if data:
        return data

    try:
        async with session.get(BANDAS_CAMBIARIAS_URL, headers=REQ_HEADERS, timeout=ClientTimeout(total=10)) as resp:
            if resp.status != 200:
                log.warning("Bandas cambiarias HTTP %s", resp.status)
                return None
            data = await resp.json()
            if not isinstance(data, dict):
                return None
            variations = data.get("variaciones") or data.get("variations")
            if isinstance(variations, dict):
                data.setdefault("variacion_superior", variations.get("upper") or variations.get("superior"))
                data.setdefault("variacion_inferior", variations.get("lower") or variations.get("inferior"))
            data["fuente"] = "DolarAPI (respaldo)"
            return data
    except Exception as exc:
        log.warning("Error obteniendo bandas cambiarias: %s", exc)
        return None

def _fmt_band_val(row: Dict[str, Any], keys: List[str]) -> Optional[float]:
    for k in keys:
        val = row.get(k)
        if isinstance(val, (int, float)):
            return float(val)
        try:
            return float(val)
        except Exception:
            continue
    return None

def _fmt_band_pct(row: Dict[str, Any], keys: List[str]) -> Optional[float]:
    for k in keys:
        val = row.get(k)
        try:
            return float(val)
        except Exception:
            continue
    return None

def format_bandas_cambiarias(data: Dict[str, Any]) -> str:
    fecha_raw = data.get("fecha") or data.get("date")
    fecha = parse_iso_ddmmyyyy(str(fecha_raw)) if fecha_raw else None

    banda_sup = _fmt_band_val(data, ["banda_superior", "upper", "upperBand", "bandaSuperior"])
    banda_inf = _fmt_band_val(data, ["banda_inferior", "lower", "lowerBand", "bandaInferior"])
    daily_sup = _fmt_band_pct(
        data,
        [
            "variacion_superior",
            "upperVariation",
            "upper_variation",
            "upperDailyVariation",
            "variation_upper",
            "upperVar",
        ],
    )
    daily_inf = _fmt_band_pct(
        data,
        [
            "variacion_inferior",
            "lowerVariation",
            "lower_variation",
            "lowerDailyVariation",
            "variation_lower",
            "lowerVar",
        ],
    )
    daily_generic = _fmt_band_pct(
        data,
        [
            "variacion_diaria",
            "variacion",
            "daily_change",
            "dailyChange",
        ],
    )
    monthly_sup = _fmt_band_pct(
        data,
        [
            "variacion_mensual_superior",
            "upperMonthlyVariation",
            "upper_monthly_variation",
        ],
    )
    monthly_inf = _fmt_band_pct(
        data,
        [
            "variacion_mensual_inferior",
            "lowerMonthlyVariation",
            "lower_monthly_variation",
        ],
    )

    pct_sup = daily_sup if daily_sup is not None else daily_generic
    pct_inf = daily_inf if daily_inf is not None else daily_generic
    has_daily_data = (pct_sup is not None) or (pct_inf is not None)

    if pct_sup is None:
        pct_sup = monthly_sup
    if pct_inf is None:
        pct_inf = monthly_inf

    def _normalize_upper_variation(val: Optional[float]) -> Optional[float]:
        if val is None:
            return None

        reference_candidates: List[Optional[float]] = [
            daily_generic,
            daily_sup,
            monthly_sup,
            monthly_inf,
            pct_inf,
        ]
        reference = next((r for r in reference_candidates if r is not None), None)

        if reference is not None and val * reference < 0:
            return -val
        return val

    pct_sup = _normalize_upper_variation(pct_sup)

    sup_txt = fmt_money_ars(banda_sup) if banda_sup is not None else "—"
    inf_txt = fmt_money_ars(banda_inf) if banda_inf is not None else "—"
    var_label = "Variación diaria" if has_daily_data else "Variación"

    def _fmt_var(val: Optional[float], is_upper: bool) -> str:
        if val is None:
            return "—"
        icon = "🟢" if is_upper else "🔴"
        return f"{icon} {pct(val, 2)}"

    pct_sup_txt = _fmt_var(pct_sup, True)
    pct_inf_txt = _fmt_var(pct_inf, False)

    header = "<b>📊 Bandas cambiarias" + (" (solo diaria)" if not has_daily_data else "") + "</b>"
    header += f" <i>Actualizado: {fecha}</i>" if fecha else ""

    col1 = ["🟢 Banda superior", "🔴 Banda inferior"]
    col2 = [sup_txt, inf_txt]
    col3 = [pct_sup_txt, pct_inf_txt]

    col1_w = max(len(_html.unescape(t)) for t in col1)
    col2_w = max(len(_html.unescape(t)) for t in col2)

    rows = []
    for c1, c2, c3 in zip(col1, col2, col3):
        rows.append(
            f"{c1.ljust(col1_w)} | {c2.rjust(col2_w)} | {c3}"
        )

    table = "\n".join(rows)

    lines = [
        header,
        f"<pre>Nombre           | Importe | {var_label}\n" + table + "</pre>",
    ]
    return "\n".join(lines)

async def cmd_bandas_cambiarias(update: Update, context: ContextTypes.DEFAULT_TYPE):
    async with ClientSession() as session:
        data = await get_bandas_cambiarias(session)
    if not data:
        await update.effective_message.reply_text(
            "No pude obtener bandas cambiarias ahora.",
            parse_mode=ParseMode.HTML,
            link_preview_options=build_preview_options(),
        )
        return

    msg = format_bandas_cambiarias(data)
    await update.effective_message.reply_text(
        msg,
        parse_mode=ParseMode.HTML,
        link_preview_options=build_preview_options(),
    )

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
        [InlineKeyboardButton("Top 3 Cedears (Rendimiento ARS/USD)", callback_data="CED:TOP3")],
        [InlineKeyboardButton("Top 5 Cedears (Proyección ARS)", callback_data="CED:TOP5")],
    ])
    await update.effective_message.reply_text("🌎 Menú cedears", reply_markup=kb_menu)

async def acc_ced_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    data = q.data
    if data == "ACC:TOP3":
        await _rank_top3(
            update,
            ACCIONES_BA,
            "📈 Top 3 Acciones (Rendimiento)",
            throttle_key="rankings:acciones:top3",
        )
        await dec_and_maybe_show(update, context, "acciones", cmd_acciones_menu)
    elif data == "ACC:TOP5":
        await _rank_proj5(
            update,
            ACCIONES_BA,
            "🏁 Top 5 Acciones (Proyección)",
            throttle_key="rankings:acciones:top5",
        )
        await dec_and_maybe_show(update, context, "acciones", cmd_acciones_menu)
    elif data == "CED:TOP3":
        await _rank_top3(
            update,
            CEDEARS_BA,
            "🌎 Top 3 Cedears (Rendimiento)",
            throttle_key=None,
        )
        await dec_and_maybe_show(update, context, "cedears", cmd_cedears_menu)
    elif data == "CED:TOP5":
        await _rank_proj5(
            update,
            CEDEARS_BA,
            "🏁 Top 5 Cedears (Proyección ARS)",
            throttle_key=None,
        )
        await dec_and_maybe_show(update, context, "cedears", cmd_cedears_menu)
    else:
        await q.answer("Selección inválida.", show_alert=True)

# ---------- Macro ----------

async def cmd_reservas(update: Update, context: ContextTypes.DEFAULT_TYPE):
    async with ClientSession() as session:
        res = await get_reservas_con_variacion(session)
    if not res:
        txt = "No pude obtener reservas ahora."
    else:
        val, fecha, prev_val, from_cache = res
        var_txt = _format_reservas_variation(prev_val, val)
        cache_hint = " <i>(dato en caché)</i>" if from_cache else ""
        txt = (f"<b>🏦 Reservas BCRA</b>{f' <i>Últ. Act.: {fecha}</i>' if fecha else ''}{cache_hint}\n"
               f"<b>{fmt_number(val,0)} MUS$</b>{var_txt}")
    await update.effective_message.reply_text(txt, parse_mode=ParseMode.HTML)

async def cmd_inflacion(update: Update, context: ContextTypes.DEFAULT_TYPE):
    httpx_client = get_httpx_client(context.application.bot_data) if context and context.application else None
    async with ClientSession() as session:
        tup = await get_inflacion_mensual(session, httpx_client=httpx_client)
    if tup is None:
        txt = "No pude obtener inflación ahora."
    else:
        val, fecha, variation = (tup + (None, None, None))[:3]
        val_str = str(round(val,1)).replace(".", ",")
        change_txt = _format_inflacion_variation(variation)
        txt = f"<b>📉 Inflación Mensual</b>{f' <i>{fecha}</i>' if fecha else ''}\n<b>{val_str}%</b>{change_txt}"
    await update.effective_message.reply_text(txt, parse_mode=ParseMode.HTML)

async def cmd_riesgo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    httpx_client = get_httpx_client(context.application.bot_data) if context and context.application else None
    async with ClientSession() as session:
        tup = await get_riesgo_pais(session, httpx_client=httpx_client)
    if tup is None:
        txt = "No pude obtener riesgo país ahora."
    else:
        rp, f, var, from_cache = tup[0], tup[1], tup[2], bool(tup[3]) if len(tup) > 3 else False
        rp_num = rp if isinstance(rp, (int, float)) else None
        var_num = var if isinstance(var, (int, float)) else None
        if rp_num is None:
            txt = "No pude obtener riesgo país ahora."
        else:
            if var_num is None and isinstance(RIESGO_CACHE, dict):
                cached_var = RIESGO_CACHE.get("variation")
                var_num = cached_var if isinstance(cached_var, (int, float)) else None
            f_str = parse_iso_ddmmyyyy(f)
            change_txt = _format_riesgo_variation(var_num) if isinstance(var_num, (int, float)) else " —"
            freshness = ""
            if from_cache:
                cache_ts = RIESGO_CACHE.get("updated_at") if isinstance(RIESGO_CACHE, dict) else None
                cache_dt = None
                try:
                    if cache_ts:
                        cache_dt = datetime.fromtimestamp(float(cache_ts), tz=TZ)
                except Exception:
                    cache_dt = None
                cache_str = cache_dt.strftime("%d/%m %H:%M") if cache_dt else "guardado"
                freshness = f" <i>(dato de respaldo, {cache_str})</i>"
            txt = (
                f"<b>📈 Riesgo País</b>{f' <i>{f_str}</i>' if f_str else ''}{freshness}\n"
                f"<b>{int(rp_num)} pb</b>{change_txt}"
            )
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
        link_preview_options=build_preview_options(),
    )
    for text, img in items:
        if img:
            await update.effective_message.reply_photo(
                img,
                caption=text,
                parse_mode=ParseMode.HTML,
            )
        else:
            await update.effective_message.reply_text(
                text,
                parse_mode=ParseMode.HTML,
                link_preview_options=build_preview_options(disable_by_default=False, prefer_small_media=True),
            )

async def cmd_menu_economia(update: Update, context: ContextTypes.DEFAULT_TYPE):
    set_menu_counter(context, "economia", 6)
    kb_menu = InlineKeyboardMarkup([
        [InlineKeyboardButton("Tipos de Cambio", callback_data="ECO:DOLAR")],
        [InlineKeyboardButton("Bandas cambiarias", callback_data="ECO:BANDAS")],
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
    if data == "ECO:BANDAS":    await cmd_bandas_cambiarias(update, context)
    if data == "ECO:RESERVAS":  await cmd_reservas(update, context)
    if data == "ECO:INFLACION": await cmd_inflacion(update, context)
    if data == "ECO:RIESGO":    await cmd_riesgo(update, context)
    if data == "ECO:NOTICIAS":  await cmd_noticias(update, context)
    await dec_and_maybe_show(update, context, "economia", cmd_menu_economia)

# ============================ ALERTAS ============================

AL_KIND, AL_FX_TYPE, AL_FX_SIDE, AL_OP, AL_MODE, AL_VALUE, AL_METRIC_TYPE, AL_TICKER, AL_CRYPTO = range(9)
ALERTS_SILENT_UNTIL: Dict[int, float] = {}
ALERTS_PAUSED: Set[int] = set()
ALERT_PRICE_TOLERANCE_PCT = 0.002  # 0.2% de margen alrededor del umbral
ALERT_PRICE_TOLERANCE_ABS = 0.0005  # margen mínimo absoluto para precios muy bajos
ALERT_REARM_COOLDOWN_SECS = 15 * 60


def _float_equals(a: Any, b: Any, abs_tol: float = 1e-6) -> bool:
    try:
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=abs_tol)
    except Exception:
        return False


def _matches_with_tolerance(cur: Any, target: Any, op: str) -> bool:
    """Evalúa si se cumplió la condición de alerta con un margen flexible."""

    try:
        cur_f = float(cur)
        target_f = float(target)
    except Exception:
        return False

    margin = max(abs(target_f) * ALERT_PRICE_TOLERANCE_PCT, ALERT_PRICE_TOLERANCE_ABS)
    if op in (">", "<"):
        return target_f - margin <= cur_f <= target_f + margin
    return False


def _alert_rearm_ready(cur: Any, target: Any, op: str) -> bool:
    try:
        cur_f = float(cur)
        target_f = float(target)
    except Exception:
        return False
    margin = max(abs(target_f) * ALERT_PRICE_TOLERANCE_PCT, ALERT_PRICE_TOLERANCE_ABS)
    if op == ">":
        return cur_f <= target_f - margin
    if op == "<":
        return cur_f >= target_f + margin
    return False


def _alert_can_trigger(rule: Dict[str, Any], cur: Any, now_ts: float) -> Tuple[bool, bool]:
    state_changed = False
    if cur is None:
        return False, state_changed
    op = rule.get("op")
    target = rule.get("value")
    if op not in {">", "<"} or target is None:
        return False, state_changed
    armed = rule.get("armed")
    if armed is None:
        rule["armed"] = True
        armed = True
        state_changed = True
    if not armed:
        last_ts = rule.get("last_trigger_ts")
        if last_ts is None:
            rule["armed"] = True
            return False, True
        try:
            last_ts_f = float(last_ts)
        except Exception:
            rule["armed"] = True
            return False, True
        if now_ts - last_ts_f < ALERT_REARM_COOLDOWN_SECS:
            return False, state_changed
        if _alert_rearm_ready(cur, target, op):
            rule["armed"] = True
            state_changed = True
        return False, state_changed
    last_ts = rule.get("last_trigger_ts")
    if last_ts is not None:
        try:
            last_ts_f = float(last_ts)
        except Exception:
            last_ts_f = None
        if last_ts_f is not None and now_ts - last_ts_f < ALERT_REARM_COOLDOWN_SECS:
            return False, state_changed
    return _matches_with_tolerance(cur, target, op), state_changed


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


def _has_duplicate_alert(
    rules: List[Dict[str, Any]],
    candidate: Dict[str, Any],
    *,
    skip_idx: Optional[int] = None,
) -> bool:
    for idx, rule in enumerate(rules):
        if skip_idx is not None and idx == skip_idx:
            continue
        if _alerts_match(rule, candidate):
            return True
    return False

def kb(rows: List[List[Tuple[str,str]]]) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton(text, callback_data=data) for text, data in r] for r in rows])


def build_preview_options(
    *, disable_by_default: bool = True, prefer_small_media: bool = False
) -> Optional[LinkPreviewOptions]:
    prefer_small = prefer_small_media or LINK_PREVIEWS_PREFER_SMALL
    if LINK_PREVIEWS_ENABLED:
        return LinkPreviewOptions(is_disabled=False, prefer_small_media=prefer_small)
    if disable_by_default:
        return LinkPreviewOptions(is_disabled=True)
    if prefer_small:
        return LinkPreviewOptions(prefer_small_media=True)
    return None

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
        [("Listar", "AL:LIST:0"), ("Agregar", "AL:ADD")],
        [("Modificar", "AL:EDIT"), ("Borrar", "AL:CLEAR")],
        [("Pausar", "AL:PAUSE"), ("Reanudar", "AL:RESUME")],
    ])


async def cmd_alertas_menu(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    prefix: Optional[str] = None,
    *,
    edit: bool = False,
) -> None:
    context.user_data.pop("al_edit_idx", None)
    text = "🔔 Menú Alertas" if not prefix else f"{prefix}\n\n🔔 Menú Alertas"
    markup = kb_alertas_menu()
    if edit and update.callback_query:
        await update.callback_query.edit_message_text(text, reply_markup=markup)
    else:
        await update.effective_message.reply_text(text, reply_markup=markup)

async def alertas_menu_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    data = q.data
    if data == "AL:EDIT":
        await cmd_alertas_edit(update, context, edit=True)
    elif data == "AL:CLEAR":
        await cmd_alertas_clear(update, context, edit=True)
    elif data == "AL:PAUSE":
        await cmd_alertas_pause(update, context, edit=True)
    elif data == "AL:RESUME":
        await cmd_alertas_resume(update, context, edit=True)
    else:
        await q.answer("Opción de menú inválida.", show_alert=True)

async def cmd_alertas_list(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    start_idx: int = 0,
    *,
    edit: bool = False,
) -> None:
    chat_id = update.effective_chat.id
    rules = ALERTS.get(chat_id, []) or []
    now = datetime.now(TZ)

    if not rules:
        txt = "No tenés alertas configuradas.\nUsá /alertas_menu → Agregar."
        markup = kb([[("Volver", "AL:MENU")]])
    else:
        page_size = ALERTS_PAGE_SIZE
        last_start = max(0, ((len(rules) - 1) // page_size) * page_size)
        start = max(0, min(start_idx, last_start))
        end = min(len(rules), start + page_size)

        lines = [
            "<b>🔔 Alertas Configuradas</b>",
            f"<i>Mostrando {start + 1}-{end} de {len(rules)}</i>",
        ]
        for i, r in enumerate(rules[start:end], start=start + 1):
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

        buttons: List[List[Tuple[str, str]]] = []
        if len(rules) > page_size:
            row: List[Tuple[str, str]] = []
            prev_idx = max(0, start - page_size)
            if start > 0:
                row.append(("⬅️ Anteriores", f"AL:LIST:{prev_idx}"))
            if end < len(rules):
                row.append(("➡️ Siguientes", f"AL:LIST:{end}"))
            if row:
                buttons.append(row)
        buttons.append([("Volver", "AL:MENU")])
        markup = kb(buttons)

    if edit and update.callback_query:
        await update.callback_query.edit_message_text(
            txt, parse_mode=ParseMode.HTML, reply_markup=markup
        )
    else:
        await update.effective_message.reply_text(
            txt, parse_mode=ParseMode.HTML, reply_markup=markup
        )


async def alertas_list_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    data = q.data or ""
    parts = data.split(":", 2)
    start_idx = 0
    if len(parts) >= 3:
        try:
            start_idx = max(0, int(parts[2]))
        except ValueError:
            await q.answer("Página inválida.", show_alert=True)
            return
    await cmd_alertas_list(update, context, start_idx=start_idx, edit=True)


async def alertas_menu_back_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    await cmd_alertas_menu(update, context, edit=True)

async def cmd_alertas_edit(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    edit: bool = False,
):
    chat_id = update.effective_chat.id
    rules = ALERTS.get(chat_id, [])
    if not rules:
        await cmd_alertas_menu(
            update,
            context,
            prefix="No tenés alertas guardadas.",
            edit=edit or bool(update.callback_query),
        )
        return
    buttons: List[List[Tuple[str, str]]] = []
    for i, r in enumerate(rules, 1):
        label = f"{i}. {_alert_rule_label(r)}"
        buttons.append([(label[:64], f"AL:EDIT:{i-1}")])
    buttons.append([("Volver", "AL:MENU"), ("Cancelar", "AL:EDIT:CANCEL")])
    kb_markup = InlineKeyboardMarkup(
        [[InlineKeyboardButton(text, callback_data=data) for text, data in row] for row in buttons]
    )
    if edit and update.callback_query:
        await update.callback_query.edit_message_text(
            "Elegí qué alerta modificar:", reply_markup=kb_markup
        )
    else:
        await update.effective_message.reply_text("Elegí qué alerta modificar:", reply_markup=kb_markup)


async def alertas_edit_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    await cmd_alertas_menu(update, context, prefix="Operación cancelada.", edit=True)


def _infer_segment_for_symbol(sym: Optional[str]) -> Optional[str]:
    if not sym:
        return None
    if sym in ACCIONES_BA:
        return "acciones"
    if sym in CEDEARS_BA:
        return "cedears"
    return None


async def alertas_edit_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    parts = q.data.split(":", 2)
    if len(parts) < 3:
        await q.edit_message_text("No pude identificar la alerta.")
        await cmd_alertas_menu(update, context, edit=True)
        return ConversationHandler.END
    try:
        idx = int(parts[2])
    except ValueError:
        await q.edit_message_text("Número de alerta inválido.")
        await cmd_alertas_menu(update, context, edit=True)
        return ConversationHandler.END
    chat_id = q.message.chat_id
    rules = ALERTS.get(chat_id, [])
    if not (0 <= idx < len(rules)):
        await q.edit_message_text("Esa alerta ya no existe.")
        await cmd_alertas_menu(update, context, edit=True)
        return ConversationHandler.END
    rule = copy.deepcopy(rules[idx])
    context.user_data["al_edit_idx"] = idx
    kind = rule.get("kind")
    al: Dict[str, Any] = {"kind": kind}
    context.user_data["al"] = al

    if kind == "fx":
        al.update({
            "type": rule.get("type"),
            "side": rule.get("side"),
            "op": rule.get("op"),
            "mode": rule.get("mode"),
        })
        type_raw = (al.get("type") or "").lower()
        label = FX_TYPE_LABELS.get(type_raw, type_raw.upper() or "?")
        side_raw = al.get("side") or ""
        side_label = SIDE_LABELS.get(side_raw, side_raw or "?")
        op_val = al.get("op")
        op_text = "↑ Sube" if op_val == ">" else "↓ Baja" if op_val == "<" else "?"
        current = fmt_money_ars(rule.get("value"))
        kb_mode = kb([
            [("Ingresar Importe", "MODE:absolute"), ("Ingresar % vs valor actual", "MODE:percent")],
            [("Volver", "BACK:OP"), ("Cancelar", "CANCEL")],
        ])
        msg = (
            f"Tipo: {label} | Lado: {side_label} | Condición: {op_text}\n"
            f"Objetivo actual: {current}\n¿Cómo querés definir el nuevo umbral?"
        )
        await q.edit_message_text(msg, reply_markup=kb_mode)
        return AL_MODE

    if kind == "metric":
        al.update({
            "type": rule.get("type"),
            "op": rule.get("op"),
            "mode": rule.get("mode"),
        })
        type_raw = al.get("type") or ""
        label = METRIC_TYPE_LABELS.get(type_raw, type_raw.capitalize() or "?")
        op_val = al.get("op")
        op_text = "↑ Sube" if op_val == ">" else "↓ Baja" if op_val == "<" else "?"
        if type_raw == "riesgo":
            current = f"{(rule.get('value') or 0):.0f} pb"
        elif type_raw == "reservas":
            current = f"{fmt_number(rule.get('value'), 0)} MUS$"
        else:
            try:
                current = f"{str(round(float(rule.get('value') or 0), 1)).replace('.', ',')}%"
            except Exception:
                current = f"{rule.get('value')}%"
        kb_mode = kb([
            [("Ingresar Importe", "MODE:absolute"), ("Ingresar % vs valor actual", "MODE:percent")],
            [("Volver", "BACK:OP"), ("Cancelar", "CANCEL")],
        ])
        msg = (
            f"Métrica: {label} | Condición: {op_text}\n"
            f"Objetivo actual: {current}\n¿Cómo querés definir el nuevo umbral?"
        )
        await q.edit_message_text(msg, reply_markup=kb_mode)
        return AL_MODE

    if kind == "crypto":
        al.update({
            "symbol": (rule.get("symbol") or "").upper(),
            "crypto_base": (rule.get("base") or rule.get("crypto_base") or "").upper(),
            "crypto_quote": (rule.get("quote") or rule.get("crypto_quote") or "").upper(),
            "op": rule.get("op"),
            "mode": rule.get("mode"),
        })
        label = crypto_display_name(al.get("symbol"), al.get("crypto_base"), al.get("crypto_quote"))
        op_val = al.get("op")
        op_text = "↑ Sube" if op_val == ">" else "↓ Baja" if op_val == "<" else "?"
        current = fmt_crypto_price(rule.get("value"), al.get("crypto_quote"))
        kb_mode = kb([
            [("Ingresar Importe", "MODE:absolute"), ("Ingresar % vs valor actual", "MODE:percent")],
            [("Volver", "BACK:OP"), ("Cancelar", "CANCEL")],
        ])
        msg = (
            f"Cripto: {label} | Condición: {op_text}\n"
            f"Objetivo actual: {current}\n¿Cómo querés definir el nuevo umbral?"
        )
        await q.edit_message_text(msg, reply_markup=kb_mode)
        return AL_MODE

    if kind == "ticker":
        sym = (rule.get("symbol") or "").upper()
        segment = rule.get("segment") or _infer_segment_for_symbol(sym) or "acciones"
        al.update({
            "symbol": sym,
            "op": rule.get("op"),
            "segment": segment,
        })
        current = fmt_money_ars(rule.get("value"))
        current_op = rule.get("op")
        op_label = "↑ Sube" if current_op == ">" else "↓ Baja" if current_op == "<" else "?"
        kb_op = kb([
            [("↑ Sube", "OP:>"), ("↓ Baja", "OP:<")],
            [
                (
                    "Volver",
                    "BACK:" + ("TICKERS_ACC" if segment == "acciones" else "TICKERS_CEDEARS"),
                ),
                ("Cancelar", "CANCEL"),
            ],
        ])
        msg = (
            f"Ticker: {_label_long(sym)}\nCondición actual: {op_label}\n"
            f"Objetivo actual: {current}\nElegí la nueva condición:"
        )
        await q.edit_message_text(msg, reply_markup=kb_op)
        return AL_OP

    await q.edit_message_text("Esta alerta no se puede modificar desde el bot.")
    await cmd_alertas_menu(update, context, edit=True)
    return ConversationHandler.END

async def cmd_alertas_clear(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    edit: bool = False,
):
    chat_id = update.effective_chat.id
    rules = ALERTS.get(chat_id, [])
    if not rules:
        await cmd_alertas_menu(
            update,
            context,
            prefix="No tenés alertas guardadas.",
            edit=edit or bool(update.callback_query),
        )
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
    markup = InlineKeyboardMarkup(
        [[InlineKeyboardButton(t, callback_data=d) for t,d in row] for row in buttons]
    )
    if edit and update.callback_query:
        await update.callback_query.edit_message_text("Elegí qué alerta borrar:", reply_markup=markup)
    else:
        await update.effective_message.reply_text("Elegí qué alerta borrar:", reply_markup=markup)

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
        cnt = len(rules); ALERTS[chat_id] = []; invalidate_alerts_cache(); await save_state()
        await q.edit_message_text(f"Se eliminaron {cnt} alertas."); return
    try: idx = int(data)
    except Exception:
        await q.edit_message_text("Acción inválida."); return
    if 0 <= idx < len(rules):
        rules.pop(idx); invalidate_alerts_cache(); await save_state(); await q.edit_message_text("Alerta eliminada.")
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
        await cmd_alertas_menu(
            update,
            context,
            prefix=msg,
            edit=edit or bool(update.callback_query),
        )
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
                    invalidate_alerts_cache()
                    await save_state()
                await cmd_alertas_pause(update, context, prefix="🔔 Alertas reanudadas.", edit=True)
                return
            if op == "INF":
                ALERTS_PAUSED.add(chat_id); ALERTS_SILENT_UNTIL.pop(chat_id, None)
                invalidate_alerts_cache()
                await save_state()
                await cmd_alertas_pause(update, context, prefix="🔕 Alertas en pausa (indefinida).", edit=True)
                return
            try:
                hrs = int(op)
            except Exception:
                await q.answer("Acción inválida.", show_alert=True); return
            until = datetime.now(TZ) + timedelta(hours=hrs)
            ALERTS_SILENT_UNTIL[chat_id] = until.timestamp(); ALERTS_PAUSED.discard(chat_id)
            invalidate_alerts_cache()
            await save_state()
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
                invalidate_alerts_cache()
                await save_state()
            await cmd_alertas_pause(update, context, prefix=f"🔔 Alerta {idx+1} reanudada.", edit=True)
            return
        if op == "INF":
            rule["pause_indef"] = True
            rule.pop("pause_until", None)
            ALERTS_PAUSED.discard(chat_id); ALERTS_SILENT_UNTIL.pop(chat_id, None)
            invalidate_alerts_cache()
            await save_state()
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
        invalidate_alerts_cache()
        await save_state()
        await cmd_alertas_pause(
            update,
            context,
            prefix=f"🔕 Alerta {idx+1} en pausa por {hrs}h (hasta {until.strftime('%d/%m %H:%M')}).",
            edit=True,
        )
        return
    await q.answer("Acción inválida.", show_alert=True)

async def cmd_alertas_resume(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    edit: bool = False,
):
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
        invalidate_alerts_cache()
        await save_state()
    await cmd_alertas_menu(
        update,
        context,
        prefix="🔔 Alertas reanudadas.",
        edit=edit or bool(update.callback_query),
    )

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
    httpx_client = get_httpx_client(context.application.bot_data) if context and context.application else None
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
            rp = await get_riesgo_pais(session, httpx_client=httpx_client); infl = await get_inflacion_mensual(session, httpx_client=httpx_client); rv = await get_reservas_lamacro(session)
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
    edit_idx = context.user_data.get("al_edit_idx")
    if not isinstance(edit_idx, int):
        edit_idx = None
    is_edit = isinstance(edit_idx, int) and 0 <= edit_idx < len(rules)
    prev_rule = rules[edit_idx] if is_edit else None
    httpx_client = get_httpx_client(context.application.bot_data) if context and context.application else None
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
            candidate = {
                "kind": "fx",
                "type": al["type"],
                "side": al["side"],
                "op": al["op"],
                "value": float(thr),
                "mode": al.get("mode"),
                "armed": True,
                "last_trigger_ts": None,
                "last_trigger_price": None,
            }
            if _has_duplicate_alert(rules, candidate, skip_idx=edit_idx if is_edit else None):
                await update.message.reply_text("Ya tenés una alerta igual configurada. Probá con otro valor.")
                return AL_VALUE
            if is_edit:
                if prev_rule:
                    for key in ("pause_until", "pause_indef"):
                        if key in prev_rule:
                            candidate[key] = prev_rule[key]
                rules[edit_idx] = candidate
            else:
                rules.append(candidate)
            _record_alert_usage(chat_id, al)
            invalidate_alerts_cache()
            await save_state()
            msg = "Listo. Alerta actualizada ✅" if is_edit else "Listo. Alerta agregada ✅"
            await update.message.reply_text(msg)
            await cmd_alertas_menu(update, context)
            return ConversationHandler.END

        if al.get("kind") == "metric":
            rp = await get_riesgo_pais(session, httpx_client=httpx_client); infl = await get_inflacion_mensual(session, httpx_client=httpx_client); rv = await get_reservas_lamacro(session)
            curmap = {"riesgo": float(rp[0]) if rp else None, "inflacion": float(infl[0]) if infl else None, "reservas": rv[0] if rv else None}
            cur = curmap.get(al["type"])
            if cur is None:
                await update.message.reply_text("No pude leer el valor actual."); return ConversationHandler.END
            thr = cur*(1 + (val/100.0)) if al.get("mode")=="percent" and al["op"] == ">" else \
                  cur*(1 - (val/100.0)) if al.get("mode")=="percent" else val
            if al.get("mode") == "absolute":
                if (al["op"] == ">" and thr <= cur) or (al["op"] == "<" and thr >= cur):
                    await update.message.reply_text("El objetivo debe ser válido respecto al valor actual."); return AL_VALUE
            candidate = {
                "kind": "metric",
                "type": al["type"],
                "op": al["op"],
                "value": float(thr),
                "mode": al.get("mode"),
                "armed": True,
                "last_trigger_ts": None,
                "last_trigger_price": None,
            }
            if _has_duplicate_alert(rules, candidate, skip_idx=edit_idx if is_edit else None):
                await update.message.reply_text("Ya tenés una alerta igual configurada. Probá con otro valor.")
                return AL_VALUE
            if is_edit:
                if prev_rule:
                    for key in ("pause_until", "pause_indef"):
                        if key in prev_rule:
                            candidate[key] = prev_rule[key]
                rules[edit_idx] = candidate
            else:
                rules.append(candidate)
            _record_alert_usage(chat_id, al)
            invalidate_alerts_cache()
            await save_state()
            msg = "Listo. Alerta actualizada ✅" if is_edit else "Listo. Alerta agregada ✅"
            await update.message.reply_text(msg)
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
                "armed": True,
                "last_trigger_ts": None,
                "last_trigger_price": None,
            }
            if _has_duplicate_alert(rules, candidate, skip_idx=edit_idx if is_edit else None):
                await update.message.reply_text("Ya tenés una alerta igual configurada. Probá con otro valor.")
                return AL_VALUE
            if is_edit:
                if prev_rule:
                    for key in ("pause_until", "pause_indef"):
                        if key in prev_rule:
                            candidate[key] = prev_rule[key]
                rules[edit_idx] = candidate
            else:
                rules.append(candidate)
            _record_alert_usage(chat_id, al)
            invalidate_alerts_cache()
            await save_state()
            target_s = fmt_crypto_price(thr, quote)
            direction = "sube a" if op == ">" else "baja a"
            prefix = "Listo. Alerta actualizada ✅" if is_edit else "Listo. Alerta agregada ✅"
            await update.message.reply_text(
                f"{prefix}\nSe disparará si el precio {direction} {target_s}."
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
        candidate = {
            "kind": "ticker",
            "symbol": sym,
            "op": op,
            "value": float(thr),
            "mode": "absolute",
            "segment": al.get("segment"),
            "armed": True,
            "last_trigger_ts": None,
            "last_trigger_price": None,
        }
        if _has_duplicate_alert(rules, candidate, skip_idx=edit_idx if is_edit else None):
            await update.message.reply_text("Ya tenés una alerta igual configurada. Probá con otro valor.")
            return AL_VALUE
        if is_edit:
            if prev_rule:
                for key in ("pause_until", "pause_indef"):
                    if key in prev_rule:
                        candidate[key] = prev_rule[key]
            rules[edit_idx] = candidate
        else:
            rules.append(candidate)
        _record_alert_usage(chat_id, al)
        invalidate_alerts_cache()
        await save_state()
        msg = "Listo. Alerta actualizada ✅" if is_edit else "Listo. Alerta agregada ✅"
        await update.message.reply_text(msg)
        await cmd_alertas_menu(update, context)
        return ConversationHandler.END

# ============================ LOOP ALERTAS ============================

async def alerts_loop(app: Application):
    try:
        await asyncio.sleep(5)
        timeout = ClientTimeout(total=12)
        httpx_client = get_httpx_client(app.bot_data)
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
                        rp = await get_riesgo_pais(session, httpx_client=httpx_client)
                        infl = await get_inflacion_mensual(session, httpx_client=httpx_client)
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
                                    ok, changed = _alert_can_trigger(r, cur, now_ts)
                                    if changed:
                                        state_dirty = True
                                    if ok:
                                        r["last_trigger_ts"] = now_ts
                                        try:
                                            r["last_trigger_price"] = float(cur)
                                        except Exception:
                                            r["last_trigger_price"] = None
                                        r["armed"] = False
                                        state_dirty = True
                                        trig.append(("fx", r["type"], r["side"], r["op"], r["value"], cur))
                                elif r.get("kind") == "metric":
                                    cur = vals.get(r["type"])
                                    if cur is None: continue
                                    ok, changed = _alert_can_trigger(r, cur, now_ts)
                                    if changed:
                                        state_dirty = True
                                    if ok:
                                        r["last_trigger_ts"] = now_ts
                                        try:
                                            r["last_trigger_price"] = float(cur)
                                        except Exception:
                                            r["last_trigger_price"] = None
                                        r["armed"] = False
                                        state_dirty = True
                                        trig.append(("metric", r["type"], r["op"], r["value"], cur))
                                elif r.get("kind") == "ticker":
                                    sym = r["symbol"]; m = metmap.get(sym, {}); cur = m.get("last_px")
                                    if cur is None: continue
                                    ok, changed = _alert_can_trigger(r, cur, now_ts)
                                    if changed:
                                        state_dirty = True
                                    if ok:
                                        r["last_trigger_ts"] = now_ts
                                        try:
                                            r["last_trigger_price"] = float(cur)
                                        except Exception:
                                            r["last_trigger_price"] = None
                                        r["armed"] = False
                                        state_dirty = True
                                        trig.append(("ticker_px", sym, r["op"], r["value"], cur))
                                elif r.get("kind") == "crypto":
                                    sym = (r.get("symbol") or "").upper()
                                    cur = crypto_prices.get(sym)
                                    if cur is None: continue
                                    ok, changed = _alert_can_trigger(r, cur, now_ts)
                                    if changed:
                                        state_dirty = True
                                    if ok:
                                        r["last_trigger_ts"] = now_ts
                                        try:
                                            r["last_trigger_price"] = float(cur)
                                        except Exception:
                                            r["last_trigger_price"] = None
                                        r["armed"] = False
                                        state_dirty = True
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
                        await save_state()
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


async def _job_projection_performance(context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        updated = await _evaluate_projection_batches()
        if updated:
            await save_state()
    except Exception as exc:
        log.warning("performance job error: %s", exc)

def _job_name_daily(chat_id: int) -> str: return f"daily_{chat_id}"

def _schedule_daily_for_chat(app: Application, chat_id: int, hhmm: str):
    for j in app.job_queue.get_jobs_by_name(_job_name_daily(chat_id)): j.schedule_removal()
    h, m = [int(x) for x in hhmm.split(":")]
    app.job_queue.run_daily(_job_send_daily, time=dtime(hour=h, minute=m, tzinfo=TZ), chat_id=chat_id, name=_job_name_daily(chat_id))

def _schedule_all_subs(app: Application):
    for chat_id, conf in SUBS.items():
        hhmm = conf.get("daily")
        if hhmm: _schedule_daily_for_chat(app, chat_id, hhmm)


async def _job_recalibrate_projections(context: ContextTypes.DEFAULT_TYPE) -> None:
    await recalibrate_projection_coeffs()


def _schedule_projection_calibration(app: Application) -> None:
    job_name = "proj_calibration_weekly"
    for j in app.job_queue.get_jobs_by_name(job_name):
        j.schedule_removal()
    app.job_queue.run_repeating(
        _job_recalibrate_projections,
        interval=7 * 24 * 3600,
        first=3600,
        name=job_name,
    )

def kb_times_full() -> InlineKeyboardMarkup:
    rows, row = [], []
    rows.append([("📃 Ver resumen ahora", "SUBS:NOW")])
    for h in range(24):
        label = f"{h:02d}:00"; row.append((label, f"SUBS:T:{label}"))
        if len(row) == 4: rows.append(row); row = []
    if row: rows.append(row)
    rows.append([("Desuscribirme","SUBS:OFF"),("Cerrar","SUBS:CLOSE")])
    return kb(rows)

async def cmd_subs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    cur = SUBS.get(chat_id, {}).get("daily")
    txt = (
        "<b>📬 Suscripción</b>\n"
        f"Resumen Diario: {'ON ('+cur+')' if cur else 'OFF'}\n"
        "Podés verlo ahora o elegir un horario (hora AR):"
    )
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
            SUBS[chat_id]["daily"] = None; await save_state()
            for j in context.application.job_queue.get_jobs_by_name(_job_name_daily(chat_id)): j.schedule_removal()
        await q.edit_message_text("Suscripción cancelada."); return ConversationHandler.END
    if data == "SUBS:NOW":
        await cmd_resumen_diario(update, context)
        return SUBS_SET_TIME
    if data.startswith("SUBS:T:"):
        hhmm = data.split(":",2)[2]
        SUBS.setdefault(chat_id, {})["daily"] = hhmm; await save_state()
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

def kb_pf_main(chat_id: Optional[int] = None) -> InlineKeyboardMarkup:
    pf = pf_get(chat_id) if chat_id is not None else None
    has_instruments = bool(pf and pf.get("items"))

    rows = [
        [InlineKeyboardButton("Ayuda", callback_data="PF:HELP")],
        [InlineKeyboardButton("Fijar base", callback_data="PF:SETBASE"), InlineKeyboardButton("Fijar monto", callback_data="PF:SETMONTO")],
        [InlineKeyboardButton("Agregar instrumento", callback_data="PF:ADD")],
    ]

    if has_instruments:
        rows.extend([
            [InlineKeyboardButton("Ver composición", callback_data="PF:LIST"), InlineKeyboardButton("Editar instrumento", callback_data="PF:EDIT")],
            # El botón «Proyección» muestra las gráficas de proyección vs. rendimiento
            # (barras vs. línea) calculadas en pf_show_projection_below.
            [InlineKeyboardButton("Rendimiento", callback_data="PF:RET"), InlineKeyboardButton("Proyección", callback_data="PF:PROJ")],
            [InlineKeyboardButton("Exportar", callback_data="PF:EXPORT")],
            [InlineKeyboardButton("Eliminar portafolio", callback_data="PF:CLEAR")],
        ])

    return InlineKeyboardMarkup(rows)

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
    lines.append(f"💱 Base: {base} / {tc}")
    lines.append(f"🎯 Monto objetivo: {f_money(monto)}")
    lines.append(f"💸 Valor invertido: {f_money(total_invertido)}")
    lines.append(f"🪙 Restante: {f_money(restante)}")
    if pf.get("items"):
        lines.append(f"📊 Valor actual estimado: {f_money(total_actual)}")
    lines.append(f"🧾 Instrumentos cargados: {len(pf.get('items', []))}")
    if tc_val is not None:
        tc_line = f"Tipo de cambio ref. ({tc}): {fmt_money_ars(tc_val)} por USD"
        if tc_ts:
            dt = datetime.fromtimestamp(tc_ts, TZ)
            tc_line += f" (al {dt.strftime('%d/%m/%Y %H:%M')})"
        lines.append(tc_line)
    return "\n".join(lines)

async def pf_refresh_menu(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    *,
    force_new: bool = False,
):
    text = await pf_main_menu_text(chat_id)
    kb_main = kb_pf_main(chat_id)
    msg_id = (
        context.user_data.get("pf_menu_msg_id")
        if isinstance(context.user_data, dict)
        else None
    )
    if msg_id and not force_new:
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=msg_id,
                text=text,
                parse_mode=ParseMode.HTML,
                reply_markup=kb_main,
            )
        except Exception:
            msg_id = None
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
        reply_markup=kb_pf_main(chat_id),
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
        await save_state()
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
        base_conf = pf.get("base", {})
        base_currency = (base_conf.get("moneda") or "ARS").upper()
        f_money = fmt_money_ars if base_currency == "ARS" else fmt_money_usd

        lines = ["<b>Elegí instrumento a editar</b>"]
        buttons = []
        for i, it in enumerate(pf["items"], 1):
            sym = it.get("simbolo") or ""
            label = _label_long(sym) if sym else (it.get("tipo", "").upper() or "Instrumento")
            qty = it.get("cantidad")
            qty_str = None
            if qty is not None:
                try:
                    qty_val = float(qty)
                    if requires_integer_units(sym):
                        qty_str = f"cant: {int(qty_val)}"
                    else:
                        qty_str = f"cant: {qty_val:.4f}".rstrip("0").rstrip(".")
                except (TypeError, ValueError):
                    qty_str = None
            importe = it.get("importe")
            importe_str = None
            if importe is not None:
                try:
                    importe_str = f"importe: {f_money(float(importe))}"
                except (TypeError, ValueError):
                    importe_str = None
            extra = ", ".join(v for v in [qty_str, importe_str] if v)
            line_detail = f"{i}. {label}"
            if extra:
                line_detail += f" — {extra}"
            lines.append(line_detail)
            buttons.append([InlineKeyboardButton(f"{i}. {label}", callback_data=f"PF:EDIT:{i-1}")])
        buttons.append([InlineKeyboardButton("Volver", callback_data="PF:BACK")])
        await _send_below_menu(
            context,
            chat_id,
            text="\n".join(lines),
            reply_markup=InlineKeyboardMarkup(buttons),
        )
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
            await _send_below_menu(context, chat_id, text="Instrumento eliminado.")
            await pf_refresh_menu(context, chat_id, force_new=True)
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
            await save_state()
            await pf_refresh_menu(context, chat_id, force_new=True)
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
            await save_state()
            await pf_refresh_menu(context, chat_id, force_new=True)
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
            await save_state()
            await pf_refresh_menu(context, chat_id, force_new=True)
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
        pf["monto"] = float(v); await save_state()
        usado = await _pf_total_usado(chat_id)
        pf_base = pf["base"]["moneda"].upper()
        f_money = fmt_money_ars if pf_base=="ARS" else fmt_money_usd
        await update.message.reply_text(f"Monto fijado: {f_money(v)} · Restante: {_restante_str(usado)}")
        await pf_refresh_menu(context, chat_id, force_new=True)
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
            mets, _ = await metrics_for_symbols(session, [yfsym])
            price_native = metric_last_price(mets.get(yfsym, {})) if mets else None
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
        await pf_refresh_menu(context, chat_id, force_new=True)
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

        await save_state()
        usado = await _pf_total_usado(chat_id)
        f_money = fmt_money_ars if pf_base=="ARS" else fmt_money_usd
        await update.message.reply_text("Actualizado ✅ · Restante: " + f_money(max(0.0, pf["monto"]-usado)))
        await pf_refresh_menu(context, chat_id, force_new=True)
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
        await save_state()

    enriched: List[Dict[str, Any]] = []
    total_invertido = 0.0
    total_actual = 0.0
    for it in items:
        sym = it.get("simbolo", "")
        tipo = it.get("tipo", "")
        qty = float(it["cantidad"]) if it.get("cantidad") is not None else None
        invertido = float(it.get("importe") or 0.0)
        met_raw = mets.get(sym, {}) if sym in mets else {}
        met = dict(met_raw) if met_raw else {}
        price_native = metric_last_price(met) if met else None
        if price_native is None:
            met = {}
        else:
            met["last_px"] = price_native
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
    value_labels: Optional[List[List[Optional[str]]]] = None,
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

    if value_labels and len(value_labels) != len(clean_rows):
        value_labels = None

    for idx in range(n_series):
        heights = [vals[idx] if vals[idx] is not None else 0.0 for _, vals in clean_rows]
        present_flags = [vals[idx] is not None for _, vals in clean_rows]
        label_texts = []
        if value_labels:
            for row_idx in range(len(clean_rows)):
                row_labels = value_labels[row_idx]
                label_texts.append(row_labels[idx] if row_labels and idx < len(row_labels) else None)
        else:
            label_texts = [None] * len(clean_rows)
        offsets = [x + (idx - (n_series - 1) / 2) * width for x in x_positions]
        bars = ax.bar(
            offsets,
            heights,
            width=width,
            color=palette[idx % len(palette)],
            label=series_labels[idx] if series_labels and idx < len(series_labels) else None,
        )

        for bar, val, present, label_text in zip(bars, heights, present_flags, label_texts):
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
            text_value = label_text or f"{val:.1f}%"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y + offset,
                text_value,
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


def _truncate_chart_label(label: str, max_len: int = 18) -> str:
    if max_len <= 0:
        return label
    label = label.strip()
    if len(label) <= max_len:
        return label
    return label[: max_len - 1].rstrip() + "…"


def _pie_image_from_items(
    pf: Dict[str, Any],
    snapshot: Optional[List[Dict[str, Any]]] = None,
    top_n: int = 8,
    label_max_len: int = 18,
) -> Optional[bytes]:
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
    if top_n > 0 and len(pair_details) > top_n:
        selected_details = list(pair_details[:top_n])
        otros_bucket = list(pair_details[top_n:])
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

    labels2 = [_truncate_chart_label(detail["label"], label_max_len) for detail in selected_details]
    vals2 = [detail["valor_actual"] for detail in selected_details]

    fig, (ax_pie, ax_info) = plt.subplots(
        1,
        2,
        figsize=(13, 6.5),
        dpi=200,
        gridspec_kw={"width_ratios": [3.6, 2.4]},
    )

    cmap = plt.get_cmap("tab20c")
    color_positions = np.linspace(0, 1, len(vals2)) if vals2 else []
    colors = [cmap(pos) for pos in color_positions]

    def autopct_fmt(pct: float) -> str:
        value = total * pct / 100.0
        return f"{pct_plain(pct, 1)}\n{f_money(value)}"

    wedges, _, autotexts = ax_pie.pie(
        vals2,
        labels=None,
        autopct=autopct_fmt,
        pctdistance=0.75,
        startangle=90,
        colors=colors,
        wedgeprops=dict(width=0.35, edgecolor="white"),
    )

    for text in autotexts:
        text.set_color("#1a1a1a")
        text.set_fontsize(8)

    legend_labels = [
        f"{label} ({pct_plain(val / total * 100.0, 1)})" if total else label for label, val in zip(labels2, vals2)
    ]
    ax_pie.legend(
        wedges,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        fontsize=8,
        frameon=False,
    )

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
    ax_info.set_xlim(0, 1.45)
    ax_info.set_ylim(0, 1)

    ax_info.text(
        0.02,
        0.95,
        "Detalle por instrumento",
        fontsize=12,
        fontweight="bold",
        va="center",
        ha="left",
    )
    headers = ["Instrumento", "%", "Actual", "Invertido", "Variación"]
    col_x = [0.08, 0.38, 0.72, 1.04, 1.32]
    header_align = ["left", "center", "right", "right", "right"]
    header_y = 0.85
    for x, header, align in zip(col_x, headers, header_align):
        ax_info.text(
            x,
            header_y,
            header,
            fontsize=10,
            fontweight="bold",
            va="center",
            ha=align,
        )

    n_rows = len(selected_details)
    row_spacing = 0.65 / max(1, n_rows)
    start_y = header_y - row_spacing

    for color, detail in zip(colors, selected_details):
        pct_value = detail["valor_actual"] / total * 100.0 if total else 0.0
        invertido = detail.get("invertido", 0.0)
        variacion = detail["valor_actual"] - invertido
        ax_info.scatter(0.03, start_y, color=color, s=70, marker="s")
        ax_info.text(
            0.08,
            start_y,
            _truncate_chart_label(detail["label"], label_max_len),
            fontsize=8.5,
            va="center",
            ha="left",
        )
        ax_info.text(0.38, start_y, pct_plain(pct_value, 1), fontsize=8.5, va="center", ha="center")
        ax_info.text(0.72, start_y, f_money(detail["valor_actual"]), fontsize=8.5, va="center", ha="right")
        ax_info.text(1.04, start_y, f_money(invertido), fontsize=8.5, va="center", ha="right")
        ax_info.text(1.32, start_y, f_money(variacion), fontsize=8.5, va="center", ha="right")
        start_y -= row_spacing

    ax_info.text(
        0.02,
        max(0.05, start_y),
        f"Total: {f_money(total)} {base_currency}",
        fontsize=9.5,
        fontweight="bold",
        va="center",
    )

    fig.subplots_adjust(wspace=0.25, left=0.04, right=0.98)

    fig.suptitle("Composición del Portafolio", fontsize=14, fontweight="bold")
    buf = io.BytesIO()
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _projection_bar_image(
    points: List[Tuple[str, Optional[float], Optional[str]]],
    formatter: Callable[[Optional[float], Optional[str]], str],
    title: str,
    subtitle: Optional[str] = None,
) -> Optional[bytes]:
    if not HAS_MPL:
        return None

    cleaned: List[Tuple[str, float, Optional[str]]] = []
    for label, value, value_label in points:
        if value is None:
            continue
        try:
            cleaned.append((label, float(value), value_label))
        except (TypeError, ValueError):
            continue

    if len(cleaned) < 2:
        return None

    labels = [label for label, _, _ in cleaned]
    values = [val for _, val, _ in cleaned]
    value_labels = [val_label for _, _, val_label in cleaned]
    max_val = max(values)
    label_wrap = 26

    fig, ax = plt.subplots(figsize=(6.6, 4.6), dpi=160)
    palette = ["#3478bc", "#34a853", "#fbbc04", "#a142f4", "#f26f5e"]
    bars = ax.bar(range(len(values)), values, color=palette[: len(values)])

    max_display = max_val if max_val else 1.0
    inner_margin = max_display * 0.04
    top_margin = max_display * 0.08
    for bar, val, value_label in zip(bars, values, value_labels):
        label = formatter(val, value_label)
        wrapped_label = textwrap.fill(label, width=label_wrap)
        text_x = bar.get_x() + bar.get_width() / 2
        height = bar.get_height()
        rotation = 90 if len(label) > 20 else 0

        if height <= 0:
            ax.text(
                text_x,
                height + top_margin,
                wrapped_label,
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
            wrapped_label,
            ha="center",
            va=va,
            fontsize=8,
            rotation=rotation,
            color=color,
        )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=0)
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


def _projection_by_instrument_image(
    points: List[Tuple[str, ProjectionRange, ProjectionRange]],
    title: str,
    subtitle: Optional[str] = None,
    formatter: Callable[[float], str] = lambda v: f"{v:+.1f}%",
) -> Optional[bytes]:
    if not HAS_MPL or np is None:
        return None

    cleaned: List[Tuple[str, ProjectionRange, ProjectionRange]] = []
    for label, val_3m, val_6m in points:
        if val_3m[0] is None and val_6m[0] is None:
            continue
        cleaned.append((label, val_3m, val_6m))

    if not cleaned:
        return None

    labels: List[str] = []
    values_3m: List[float] = []
    values_6m: List[float] = []
    missing_3m: List[bool] = []
    missing_6m: List[bool] = []
    labels_3m: List[str] = []
    labels_6m: List[str] = []
    for label, val_3m, val_6m in cleaned:
        center_3m, _, _ = _projection_bounds(val_3m)
        center_6m, _, _ = _projection_bounds(val_6m)
        labels.append(label)
        missing_3m.append(center_3m is None)
        missing_6m.append(center_6m is None)
        values_3m.append(center_3m if center_3m is not None else 0.0)
        values_6m.append(center_6m if center_6m is not None else 0.0)
        labels_3m.append(_format_projection_range(val_3m, 1))
        labels_6m.append(_format_projection_range(val_6m, 1))

    all_values = [abs(v) for v, miss in zip(values_3m, missing_3m) if not miss]
    all_values += [abs(v) for v, miss in zip(values_6m, missing_6m) if not miss]
    max_abs = max(all_values) if all_values else 0.0
    if max_abs <= 0:
        max_abs = 1.0

    x = np.arange(len(labels))
    width = 0.26
    gap = 0.12
    label_wrap = 12

    def _color(val: float, missing: bool, is_longer: bool) -> str:
        if missing:
            return "#c3c7cf" if not is_longer else "#a1a6ad"
        if val > 0:
            return "#8fd694" if not is_longer else "#2c7d3b"
        if val < 0:
            return "#f28b82" if not is_longer else "#c5221f"
        return "#c3c7cf" if not is_longer else "#a1a6ad"

    fig, ax = plt.subplots(figsize=(7.8, 4.8), dpi=160)
    bars_3m = ax.bar(
        x - (width / 2 + gap / 2),
        values_3m,
        width,
        color=[_color(v, miss, False) for v, miss in zip(values_3m, missing_3m)],
        edgecolor="#ffffff",
        linewidth=0.6,
        label="3M",
    )
    bars_6m = ax.bar(
        x + (width / 2 + gap / 2),
        values_6m,
        width,
        color=[_color(v, miss, True) for v, miss in zip(values_6m, missing_6m)],
        edgecolor="#ffffff",
        linewidth=0.6,
        label="6M",
    )

    outer_offset = max_abs * 0.09
    inner_offset = max_abs * 0.07
    label_offsets_3m = [outer_offset * 0.35 if idx % 2 == 0 else -outer_offset * 0.35 for idx in range(len(labels))]
    label_offsets_6m = [-outer_offset * 0.35 if idx % 2 == 0 else outer_offset * 0.35 for idx in range(len(labels))]

    def _annotate_bar(
        bar,
        val: float,
        missing: bool,
        text_override: str,
        offset_jitter: float,
    ) -> None:
        height = bar.get_height()
        text = "N/D" if missing else text_override
        if missing:
            y = height + outer_offset + offset_jitter
            va = "bottom"
            color = "#1a1a1a"
        elif val > 0:
            if height > max_abs * 0.18:
                y = height - inner_offset + offset_jitter
                va = "top"
                color = "#ffffff"
            else:
                y = height + outer_offset + offset_jitter
                va = "bottom"
                color = "#1a1a1a"
        elif val < 0:
            if abs(height) > max_abs * 0.18:
                y = height + inner_offset + offset_jitter
                va = "bottom"
                color = "#ffffff"
            else:
                y = height - outer_offset + offset_jitter
                va = "top"
                color = "#1a1a1a"
        else:
            y = height + outer_offset + offset_jitter
            va = "bottom"
            color = "#1a1a1a"

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            text,
            ha="center",
            va=va,
            fontsize=8,
            color=color,
        )

    for idx, (bar, val, miss, text) in enumerate(zip(bars_3m, values_3m, missing_3m, labels_3m)):
        _annotate_bar(bar, val, miss, text, label_offsets_3m[idx])
    for idx, (bar, val, miss, text) in enumerate(zip(bars_6m, values_6m, missing_6m, labels_6m)):
        _annotate_bar(bar, val, miss, text, label_offsets_6m[idx])

    ax.set_xticks(x)
    ax.set_xticklabels([textwrap.fill(lbl, width=label_wrap) for lbl in labels], rotation=0, ha="center")
    ax.axhline(0, color="#4a4a4a", linewidth=0.8)
    ax.set_ylabel("Variación %")
    ax.set_title(title + (f"\n{subtitle}" if subtitle else ""))
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend()
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax.set_ylim(-max_abs * 1.25, max_abs * 1.25)

    ax.margins(x=0.08)

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
    top_n: int = 8,
    label_max_len: int = 16,
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

    cleaned.sort(key=lambda item: abs(item[1]), reverse=True)

    if top_n > 0 and len(cleaned) > top_n:
        selected = cleaned[:top_n]
        remainder = cleaned[top_n:]
        remainder_vals = [val for _, val in remainder]
        remainder_avg = sum(remainder_vals) / len(remainder_vals) if remainder_vals else 0.0
        selected.append(("Otros", remainder_avg))
        cleaned = selected

    labels = [label for label, _ in cleaned]
    values = [val for _, val in cleaned]
    max_abs = max(abs(val) for val in values) if values else 0.0
    if max_abs <= 0:
        max_abs = 1.0

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
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

    display_labels = [_truncate_chart_label(label, label_max_len) for label in labels]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(display_labels, rotation=20, ha="right", fontsize=8)
    ax.margins(x=0.05)
    ax.axhline(0, color="#4a4a4a", linewidth=0.8)
    ax.set_ylabel("Variación %")
    ax.set_title(title + (f"\n{subtitle}" if subtitle else ""))
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ylim = max_abs * 1.25
    ax.set_ylim(-ylim, ylim)

    legend_labels = [
        f"{label} ({(formatter(val) if formatter else f'{val:+.1f}%')})" for label, val in zip(display_labels, values)
    ]
    ax.legend(
        bars,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        fontsize=7,
        frameon=False,
    )

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.read()

async def pf_send_composition(context: ContextTypes.DEFAULT_TYPE, chat_id: int, top_n: int = 8):
    pf = pf_get(chat_id)
    pf_base = pf["base"]["moneda"].upper()
    f_money = fmt_money_ars if pf_base=="ARS" else fmt_money_usd
    if not pf["items"]:
        await _send_below_menu(context, chat_id, text="Tu portafolio está vacío. Usá «Agregar instrumento»."); return
    snapshot, last_ts, total_invertido, total_actual, tc_val, tc_ts = await pf_market_snapshot(pf)
    fecha = datetime.fromtimestamp(last_ts, TZ).strftime("%d/%m/%Y") if last_ts else None
    header = f"<b>🎨 Portafolio</b> — Base: {pf['base']['moneda'].upper()}/{pf['base']['tc'].upper()}"
    if fecha:
        header += f" <i>Datos al {fecha}</i>"
    lines = [header, f"🎯 Monto objetivo: {f_money(pf['monto'])}"]
    lines.append(f"💵 Valor invertido: {f_money(total_invertido)}")
    lines.append(f"🧮 Valor actual estimado: {f_money(total_actual)}")
    delta = total_actual - total_invertido
    if total_invertido > 0:
        lines.append(f"📊 Variación estimada: {f_money(delta)} ({pct(delta/total_invertido*100.0,2)})")
    restante = max(0.0, pf['monto'] - total_invertido)
    lines.append(f"🪙 Restante del objetivo: {f_money(restante)}")
    if tc_val is not None:
        tc_line = f"💱 Tipo de cambio ref. ({pf['base']['tc'].upper()}): {fmt_money_ars(tc_val)} por USD"
        if tc_ts:
            tc_line += f" (al {datetime.fromtimestamp(tc_ts, TZ).strftime('%d/%m/%Y %H:%M')})"
        lines.append(tc_line)
    lines.append("")
    for i, entry in enumerate(snapshot, 1):
        if i > 1:
            lines.append("")
        linea = f"{i}. {entry['label']}"
        linea += f" · 💰 Valor: {f_money(entry['valor_actual'])}"
        if entry['invertido'] > 0:
            r_ind = (entry['valor_actual']/entry['invertido']-1.0)*100.0
            linea += f" ({pct(r_ind,2)} vs {f_money(entry['invertido'])})"
        qty_txt = format_quantity(entry['symbol'], entry.get('cantidad'))
        if qty_txt:
            linea += f" · 📦 Cant: {qty_txt}"
        if entry.get('peso'):
            linea += f" · ⚖️ Peso: {pct_plain(entry['peso']*100.0,1)}"
        added_str = format_added_date(entry.get('added_ts'))
        if added_str:
            linea += f" · ⏳ Desde: {added_str}"
        lines.append(linea)
    if not HAS_MPL:
        lines.append("")
        lines.append("ℹ️ Instalá matplotlib para ver la composición en gráficos.")
    await _send_below_menu(context, chat_id, text="\n".join(lines))
    # torta
    img = _pie_image_from_items(pf, snapshot, top_n=top_n)
    if img:
        await _send_below_menu(context, chat_id, photo_bytes=img)
    await pf_refresh_menu(context, chat_id, force_new=True)

# --- Rendimiento (debajo del menú) ---

async def pf_show_return_below(context: ContextTypes.DEFAULT_TYPE, chat_id: int, top_n: int = 8):
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
        tc_line = f"💱 Tipo de cambio ref. ({pf['base']['tc'].upper()}): {fmt_money_ars(tc_val)} por USD"
        if tc_ts:
            tc_line += f" (al {datetime.fromtimestamp(tc_ts, TZ).strftime('%d/%m/%Y %H:%M')})"
        lines.append(tc_line)

    port_daily_vals = [entry['peso'] * entry['daily_change'] for entry in snapshot if entry.get('daily_change') is not None]
    daily_sum: Optional[float] = None
    if port_daily_vals:
        daily_sum = sum(port_daily_vals)
        lines.append(f"⚡ Variación diaria estimada: {pct(daily_sum,2)}")

    has_daily_data = any(entry.get('daily_change') is not None for entry in snapshot)
    return_points: List[Tuple[str, Optional[float]]] = []
    daily_points: List[Tuple[str, Optional[float]]] = []
    lines.append("")
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
            detail += f" · 📦 Cant: {qty_txt}"
        if entry.get('precio_base') is not None:
            detail += f" · 💵 Px: {f_money(entry['precio_base'])}"
        daily = entry.get('daily_change')
        if daily is not None:
            detail += f" · 🌅 Día: {pct(daily,2)}"
        if entry.get('peso'):
            detail += f" · ⚖️ Peso: {pct_plain(entry['peso']*100.0,1)}"
        added_str = format_added_date(entry.get('added_ts'))
        if added_str:
            detail += f" · ⏳ Desde: {added_str}"
        lines.append(detail)

        short_label = _label_short(entry['symbol']) if entry.get('symbol') else label
        if ret_pct is not None:
            return_points.append((short_label, ret_pct))
        if has_daily_data:
            daily_points.append((short_label, daily if daily is not None else None))

    delta_t = total_actual - total_invertido
    lines.append("")
    lines.append(f"💸 Invertido: {f_money(total_invertido)}")
    lines.append(f"🧮 Valor actual estimado: {f_money(total_actual)}")
    if total_invertido > 0:
        lines.append(f"📊 Variación total: {f_money(delta_t)} ({pct((delta_t/total_invertido)*100.0,2)})")
    else:
        lines.append(f"📊 Variación total: {f_money(delta_t)}")

    sin_datos = [entry['label'] for entry in snapshot if not entry.get('metrics')]
    if sin_datos:
        lines.append("")
        lines.append("Sin datos recientes para: " + ", ".join(sin_datos) + ". Se mantiene el valor cargado.")

    if not HAS_MPL:
        lines.append("")
        lines.append("ℹ️ Instalá matplotlib para ver el gráfico de rendimiento.")

    await _send_below_menu(context, chat_id, text="\n".join(lines))

    cleaned_daily = [pt for pt in daily_points if pt[1] is not None]
    if cleaned_daily:
        daily_img = _return_bar_image(
            cleaned_daily,
            "Variación diaria",
            "Cambios porcentuales del día",
            formatter=lambda v: f"{v:+.2f}%",
            top_n=top_n,
        )
        if daily_img:
            await _send_below_menu(context, chat_id, photo_bytes=daily_img)

    if return_points:
        img = _return_bar_image(
            return_points,
            "Rendimiento por instrumento",
            "Variación acumulada vs. invertido",
            formatter=lambda v: f"{v:+.1f}%",
            top_n=top_n,
        )
        if img:
            await _send_below_menu(context, chat_id, photo_bytes=img)
    await pf_refresh_menu(context, chat_id, force_new=True)

# --- Proyección (debajo del menú) ---

async def pf_show_projection_below(context: ContextTypes.DEFAULT_TYPE, chat_id: int):
    """Renderiza y envía las comparativas de proyección vs. rendimiento."""
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
    w3_low = 0.0
    w3_high = 0.0
    w6_low = 0.0
    w6_high = 0.0
    detail: List[str] = []
    per_instrument_points: List[Tuple[str, ProjectionRange, ProjectionRange]] = []
    history_added = False
    for entry in snapshot:
        metrics = entry.get("metrics") or {}
        weight = float(entry.get("peso") or 0.0)
        if not metrics:
            continue

        raw3 = projection_3m_raw(metrics)
        raw6 = projection_6m_raw(metrics)
        symbol = entry.get("symbol")
        if symbol:
            register_projection_history(symbol, "3m", raw3, metrics)
            register_projection_history(symbol, "6m", raw6, metrics)
            history_added = True
        p3 = projection_3m(metrics)
        p6 = projection_6m(metrics)
        c3, l3, h3 = _projection_bounds(p3)
        c6, l6, h6 = _projection_bounds(p6)
        w3 += weight * (c3 or 0.0)
        w6 += weight * (c6 or 0.0)
        w3_low += weight * (l3 or c3 or 0.0)
        w3_high += weight * (h3 or c3 or 0.0)
        w6_low += weight * (l6 or c6 or 0.0)
        w6_high += weight * (h6 or c6 or 0.0)

        short_label = _label_short(entry["symbol"]) if entry.get("symbol") else entry["label"]
        per_instrument_points.append((short_label, p3, p6))

        if detail:
            detail.append("")
        extras = [f"⚖️ Peso {pct_plain(weight * 100.0, 1)}"]
        added_str = format_added_date(entry.get("added_ts"))
        if added_str:
            extras.append(f"⏳ Desde {added_str}")

        invertido = float(entry.get("invertido") or 0.0)
        valor_actual = float(entry.get("valor_actual") or 0.0)
        delta = valor_actual - invertido
        actual_pct = (delta / invertido) * 100.0 if invertido > 0 else None
        actual_txt = pct(actual_pct, 2)
        proj_txt = f"🔭 3M {_format_projection_range(p3, 2)} | 6M {_format_projection_range(p6, 2)}"
        delta_txt = f"📈 Δ {f_money(delta)}"

        detail.append(
            "• "
            + short_label
            + f" → Rend. actual 📊 {actual_txt} ({delta_txt}) | Proyección {proj_txt} ("
            + " · ".join(extras)
            + ")"
        )

    forecast3 = total_actual * (1.0 + w3/100.0)
    forecast6 = total_actual * (1.0 + w6/100.0)
    forecast3_low = total_actual * (1.0 + w3_low/100.0)
    forecast3_high = total_actual * (1.0 + w3_high/100.0)
    forecast6_low = total_actual * (1.0 + w6_low/100.0)
    forecast6_high = total_actual * (1.0 + w6_high/100.0)
    total_pct = ((total_actual / total_invertido) - 1.0) * 100.0 if total_invertido > 0 else math.nan

    header = "<b>🔮 Proyección del Portafolio</b>"
    if fecha:
        header += f" <i>Datos al {fecha}</i>"
    lines = [header, f"🧮 Valor actual estimado: {f_money(total_actual)}"]
    lines.append(
        "✨ Proyección 3M (rango estimado): "
        + f"{pct(w3,2)} ({pct(w3_low,2)} a {pct(w3_high,2)}) → "
        + f"{f_money(forecast3)} ({f_money(forecast3_low)} a {f_money(forecast3_high)})"
    )
    lines.append(
        "🌟 Proyección 6M (rango estimado): "
        + f"{pct(w6,2)} ({pct(w6_low,2)} a {pct(w6_high,2)}) → "
        + f"{f_money(forecast6)} ({f_money(forecast6_low)} a {f_money(forecast6_high)})"
    )
    if tc_val is not None:
        tc_line = f"💱 Tipo de cambio ref. ({pf['base']['tc'].upper()}): {fmt_money_ars(tc_val)} por USD"
        if tc_ts:
            tc_line += f" (al {datetime.fromtimestamp(tc_ts, TZ).strftime('%d/%m/%Y %H:%M')})"
        lines.append(tc_line)

    if detail:
        lines.append("")
        lines.extend(detail)

    sin_datos = [entry['label'] for entry in snapshot if not entry.get('metrics')]
    if sin_datos:
        lines.append("")
        lines.append(
            "Sin datos de mercado para: "
            + ", ".join(sin_datos)
            + ". Se asumió proyección 0% para esos instrumentos."
        )

    if not HAS_MPL:
        lines.append("")
        lines.append("ℹ️ Instalá matplotlib para ver la proyección en gráficos.")

    await _send_below_menu(context, chat_id, text="\n".join(lines))
    if history_added:
        asyncio.create_task(save_state())

    img = _projection_bar_image(
        [
            ("Actual", total_actual, f_money(total_actual)),
            ("3M", forecast3, f"{f_money(forecast3)} ({f_money(forecast3_low)} a {f_money(forecast3_high)})"),
            ("6M", forecast6, f"{f_money(forecast6)} ({f_money(forecast6_low)} a {f_money(forecast6_high)})"),
        ],
        lambda value, label: label or f_money(value),
        "Proyección del portafolio",
        "Valores estimados",
    )
    if img:
        await _send_below_menu(context, chat_id, photo_bytes=img)

    per_instrument_img = _projection_by_instrument_image(
        per_instrument_points,
        "Proyección por instrumento",
        "Variación porcentual esperada",
    )
    if per_instrument_img:
        await _send_below_menu(context, chat_id, photo_bytes=per_instrument_img)
    await pf_refresh_menu(context, chat_id, force_new=True)

# ============================ RESUMEN DIARIO ============================

async def cmd_resumen_diario(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id if update.effective_chat else None
    user_id = update.effective_user.id if update.effective_user else None
    if is_throttled("resumen", chat_id, user_id, ttl=45):
        await update.effective_message.reply_text("⏳ Esperá unos segundos antes de pedir otro resumen.")
        return
    httpx_client = get_httpx_client(context.application.bot_data) if context and context.application else None
    async with ClientSession() as session:
        fx = await get_dolares(session)
        bandas = await get_bandas_cambiarias(session)
        rp = await get_riesgo_pais(session, httpx_client=httpx_client)
        infl = await get_inflacion_mensual(session, httpx_client=httpx_client)
        rv = await get_reservas_lamacro(session)
        news = await fetch_rss_entries(session, limit=3)

    partes = []
    if fx:
        partes.append(format_dolar_message(fx))
    if bandas:
        partes.append(format_bandas_cambiarias(bandas))
    if rp:
        change_txt = _format_riesgo_variation(rp[2])
        partes.append(
            f"<b>📈 Riesgo País</b> {rp[0]} pb{change_txt}"
            + (f" <i>({parse_iso_ddmmyyyy(rp[1])})</i>" if rp[1] else "")
        )
    if infl:
        partes.append(f"<b>📉 Inflación Mensual</b> {str(round(infl[0],1)).replace('.',',')}%" + (f" <i>({infl[1]})</i>" if infl[1] else ""))
    if rv:
        partes.append(f"<b>🏦 Reservas</b> {fmt_number(rv[0],0)} MUS$" + (f" <i>({rv[1]})</i>" if rv[1] else ""))
    if news:
        partes.append(format_news_block(news)[0])

    txt = "\n\n".join(partes) if partes else "Sin datos para el resumen ahora."
    await update.effective_message.reply_text(
        txt,
        parse_mode=ParseMode.HTML,
        link_preview_options=build_preview_options(),
    )


async def cmd_performance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    evaluated = [b for b in PROJECTION_BATCHES if b.get("evaluated")]
    if not evaluated:
        await update.effective_message.reply_text("Sin métricas de performance todavía.")
        return
    batches_3m = [b for b in evaluated if b.get("horizon") == WINDOW_DAYS[3]]
    batches_6m = [b for b in evaluated if b.get("horizon") == WINDOW_DAYS[6]]
    summary_3m = _summarize_projection_performance(batches_3m)
    summary_6m = _summarize_projection_performance(batches_6m)
    lines = ["<b>📊 Performance de Proyecciones</b>"]
    for label, summary in (
        (f"3M ({WINDOW_DAYS[3]} ruedas)", summary_3m),
        (f"6M ({WINDOW_DAYS[6]} ruedas)", summary_6m),
    ):
        if summary.get("count"):
            lines.append(
                f"• {label}: MAE {pct_plain(summary.get('mae'), 2)}"
                f" | Hit {pct_plain((summary.get('hit_rate') or 0) * 100.0, 1)}"
                f" | Spearman {fmt_number(summary.get('spearman'), 2)}"
                f" | N={summary.get('count')}"
            )
        else:
            lines.append(f"• {label}: sin datos evaluados.")

    latest = sorted(
        evaluated,
        key=lambda b: b.get("evaluated_at") or b.get("created_at") or 0,
        reverse=True,
    )[:5]
    if latest:
        lines.append("")
        lines.append("<b>Últimos batches</b>")
        for batch in latest:
            created_date = batch.get("created_date") or "s/d"
            horizon = batch.get("horizon")
            mae = pct_plain(batch.get("mae"), 2)
            hit_rate = pct_plain((batch.get("hit_rate") or 0) * 100.0, 1)
            spearman = fmt_number(batch.get("spearman"), 2)
            count = batch.get("count") or 0
            lines.append(
                f"• {created_date} · {horizon}r: MAE {mae} | Hit {hit_rate} | ρ {spearman} | N={count}"
            )

    await update.effective_message.reply_text(
        "\n".join(lines),
        parse_mode=ParseMode.HTML,
        link_preview_options=build_preview_options(),
    )

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

    async def _metrics(_: web.Request) -> web.Response:
        return web.json_response(metrics.snapshot())

    async def _performance(_: web.Request) -> web.Response:
        evaluated = [b for b in PROJECTION_BATCHES if b.get("evaluated")]
        batches_3m = [b for b in evaluated if b.get("horizon") == WINDOW_DAYS[3]]
        batches_6m = [b for b in evaluated if b.get("horizon") == WINDOW_DAYS[6]]
        return web.json_response(
            {
                "summary": {
                    "3m": _summarize_projection_performance(batches_3m),
                    "6m": _summarize_projection_performance(batches_6m),
                },
                "batches": evaluated[-50:],
                "generated_at": time(),
            }
        )

    router = inner_app.router

    for path, handler in (
        ("/", _health),
        ("/healthz", _health),
        ("/metrics", _metrics),
        ("/performance", _performance),
    ):
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
            router.add_get(path, handler)
        except (RuntimeError, ValueError) as exc:
            logging.debug("No se pudo registrar ruta %s: %s", path, exc)

BOT_COMMANDS = [
    BotCommand("economia","Menú de economía"),
    BotCommand("acciones","Menú acciones"),
    BotCommand("cedears","Menú cedears"),
    BotCommand("alertas_menu","Configurar alertas"),
    BotCommand("portafolio","Menú portafolio"),
    BotCommand("subs","Suscripción a resumen diario"),
    BotCommand("performance","Métricas de performance"),
]


async def _shutdown_httpx_client(app: Application) -> None:
    client = app.bot_data.get(HTTPX_CLIENT_KEY)
    if isinstance(client, httpx.AsyncClient):
        await client.aclose()


def build_application() -> Application:
    httpx_client = build_httpx_client()
    app = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .post_shutdown(_shutdown_httpx_client)
        .build()
    )
    app.bot_data[HTTPX_CLIENT_KEY] = httpx_client

    # Comandos
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("economia", cmd_menu_economia))
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
    app.add_handler(CallbackQueryHandler(alertas_list_cb, pattern="^AL:LIST(:[0-9]+)?$"))
    app.add_handler(CallbackQueryHandler(alertas_menu_back_cb, pattern="^AL:MENU$", block=False))
    app.add_handler(CallbackQueryHandler(alertas_menu_cb, pattern="^AL:(EDIT|CLEAR|PAUSE|RESUME)$"))
    app.add_handler(CallbackQueryHandler(alertas_edit_cancel, pattern="^AL:EDIT:CANCEL$"))
    app.add_handler(CallbackQueryHandler(alertas_clear_cb, pattern="^CLR:"))
    app.add_handler(CommandHandler("alertas_pause", instrument_command("alertas_pause", cmd_alertas_pause)))
    app.add_handler(CallbackQueryHandler(alerts_pause_cb, pattern="^AP:"))

    # Alertas - conversación Agregar
    alert_conv = ConversationHandler(
        entry_points=[
            CallbackQueryHandler(alertas_add_start, pattern="^AL:ADD$"),
            CallbackQueryHandler(alertas_edit_start, pattern="^AL:EDIT:[0-9]+$"),
        ],
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
            CommandHandler("subs", instrument_command("subs", cmd_subs)),
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
    app.add_handler(CommandHandler("portafolio", instrument_command("portafolio", cmd_portafolio)))
    app.add_handler(CallbackQueryHandler(pf_menu_cb, pattern="^PF:"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, pf_text_input))

    # Resumen diario on-demand
    app.add_handler(CommandHandler("resumen", instrument_command("resumen", cmd_resumen_diario)))
    app.add_handler(CommandHandler("performance", instrument_command("performance", cmd_performance)))

    app.add_error_handler(handle_error)

    return app

async def main():
    await load_state()
    application = build_application()
    _schedule_all_subs(application)
    _schedule_projection_calibration(application)

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
        from main import configure_logging

        configure_logging()
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
