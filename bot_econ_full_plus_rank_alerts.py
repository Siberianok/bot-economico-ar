#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, re, html, json, time, logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Tuple, Optional

import httpx
from telegram import Update, LinkPreviewOptions
from telegram.constants import ParseMode
from telegram.ext import (
    ApplicationBuilder, Application, CommandHandler,
    ContextTypes, Defaults
)

# ===================== CONFIG =====================

BOT_TOKEN = os.environ.get("BOT_TOKEN", "").strip()
if not BOT_TOKEN:
    print("ERROR: Falta BOT_TOKEN en variables de entorno.", file=sys.stderr)
    sys.exit(1)

BASE_URL = os.environ.get("RENDER_EXTERNAL_URL") or "https://bot-economico-ar.onrender.com"
WEBHOOK_PATH = os.environ.get("WEBHOOK_PATH", "/tgwebhook")
PORT = int(os.environ.get("PORT", "10000"))
LISTEN = os.environ.get("LISTEN", "0.0.0.0")

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
COMMON_HEADERS = {
    "User-Agent": UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "es-AR,es;q=0.9,en-US;q=0.8",
    "Connection": "keep-alive",
    "Referer": "https://finance.yahoo.com/",
}

TIMEOUT = 10.0
HTTP  = httpx.Client(timeout=TIMEOUT, headers=COMMON_HEADERS, follow_redirects=True)
AHTTP = httpx.AsyncClient(timeout=TIMEOUT, headers=COMMON_HEADERS, follow_redirects=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("bot")

# ===================== CACHE ======================

_cache: Dict[str, Tuple[datetime, Any]] = {}

def now_utc() -> datetime: return datetime.now(timezone.utc)

def cache_get(k:str):
    v=_cache.get(k); 
    if not v: return None
    exp, val = v
    if now_utc()>=exp: _cache.pop(k,None); return None
    return val

def cache_set(k:str, val:Any, ttl:int): _cache[k]=(now_utc()+timedelta(seconds=ttl), val)

def html_bold(s:str)->str: return f"<b>{html.escape(s)}</b>"
def html_code(s:str)->str: return f"<code>{html.escape(s)}</code>"

def fmt_num(x)->str:
    if x is None: return "N/D"
    try:
        return f"{float(x):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except: return str(x)

def pct(x: float) -> str:
    try:
        return f"{x:+.2f}%"
    except:
        return "N/D"

# ===================== DÓLARES ====================

def get_dolares() -> Dict[str, Any]:
    ck="dolares_v2"; c=cache_get(ck)
    if c: return c
    data={}
    base="https://dolarapi.com/v1/dolares/"
    for tipo in ["blue","oficial","mayorista","cripto"]:
        try:
            r=HTTP.get(base+tipo)
            if r.status_code==200:
                j=r.json()
                data[tipo]={
                    "compra": j.get("compra"),
                    "venta": j.get("venta"),
                    "fecha": j.get("fechaActualizacion") or j.get("fecha_actualizacion"),
                }
            else:
                log.warning(f"DolarAPI fallo {tipo}: {r.status_code}")
        except Exception as e:
            log.warning(f"DolarAPI error {tipo}: {e}")
    # MEP/CCL por CriptoYa (DolarAPI no los expone)
    try:
        r=HTTP.get("https://criptoya.com/api/dolar")
        if r.status_code==200:
            jj=r.json()
            if isinstance(jj, dict):
                if "mep" in jj: data["mep"]={"venta": jj["mep"]}
                if "ccl" in jj: data["ccl"]={"venta": jj["ccl"]}
        else:
            log.warning(f"CriptoYa fallo: {r.status_code}")
    except Exception as e:
        log.warning(f"CriptoYa error: {e}")
    cache_set(ck,data,120)
    return data

# ========== RIESGO / INFLACIÓN / RESERVAS ==========

def get_riesgo_pais()->Optional[Dict[str,Any]]:
    ck="riesgo_v2"; c=cache_get(ck)
    if c: return c
    for url in [
        "https://api.argentinadatos.com/v1/finanzas/indices/riesgo-pais/ultimo/",
        "https://api.argentinadatos.com/v1/finanzas/indices/riesgo-pais/ultimo"
    ]:
        try:
            r=HTTP.get(url)
            if r.status_code==200:
                j=r.json(); cache_set(ck,j,300); return j
        except Exception as e:
            log.warning(f"ArgentinaDatos riesgo error: {e}")
    return None

def get_inflacion()->Optional[Dict[str,Any]]:
    ck="inflacion_v2"; c=cache_get(ck)
    if c: return c
    for url in [
        "https://api.argentinadatos.com/v1/finanzas/indices/inflacion/",
        "https://api.argentinadatos.com/v1/finanzas/indices/inflacion"
    ]:
        try:
            r=HTTP.get(url)
            if r.status_code==200:
                arr=r.json()
                if isinstance(arr,list) and arr:
                    ultimo=arr[-1]; cache_set(ck,ultimo,6*3600); return ultimo
        except Exception as e:
            log.warning(f"ArgentinaDatos inflacion error: {e}")
    return None

def get_reservas()->Optional[Dict[str,Any]]:
    ck="reservas_v2"; c=cache_get(ck)
    if c: return c
    try:
        r=HTTP.get("https://www.lamacro.ar/variables")
        if r.status_code==200:
            s=r.text
            m = re.search(r"Reservas Internacionales.*?(\d{1,3}(?:\.\d{3})+)", s, re.I|re.S)
            f = re.search(r"Últ\. act:\s*([0-9]{2}/[0-9]{2}/[0-9]{4})", s, re.I|re.S)
            if m:
                val=float(m.group(1).replace(".",""))
                out={"valor_musd": val, "fecha": f.group(1) if f else None}
                cache_set(ck,out,6*3600); return out
            else:
                log.warning("No se pudo parsear Reservas en LaMacro.")
    except Exception as e:
        log.warning(f"Reservas error: {e}")
    return None

# ===================== NOTICIAS ===================

NEWS_SOURCES=[
    "https://www.iprofesional.com/rss/economia",
    "https://www.cronista.com/rss/economia/",
    "https://www.cronista.com/files/rss/news.xml",
    "https://www.clarin.com/rss/economia/",
    "https://www.lanacion.com.ar/arc/outboundfeeds/rss/?outputType=xml&section=economia",
]

def parse_rss(xml_text:str)->List[Tuple[str,str]]:
    items = re.findall(r"<item>.*?<title>(.*?)</title>.*?<link>(.*?)</link>.*?</item>", xml_text, re.S|re.I)
    out=[]
    for t,l in items:
        t=re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", t)
        l=re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", l)
        t=html.unescape(re.sub("<.*?>","",t)).strip()
        l=html.unescape(re.sub("<.*?>","",l)).strip()
        if t and l: out.append((t,l))
    return out

def get_noticias(n=5)->List[Tuple[str,str]]:
    ck="news_v3"; c=cache_get(ck)
    if c: return c[:n]
    acc=[]; seen=set()
    for url in NEWS_SOURCES:
        try:
            r=HTTP.get(url)
            if r.status_code==200:
                for t,u in parse_rss(r.text):
                    if t not in seen:
                        seen.add(t); acc.append((t,u))
                        if len(acc)>=30: break
            else:
                log.warning(f"RSS fallo {url}: {r.status_code}")
        except Exception as e:
            log.warning(f"RSS error {url}: {e}")
    acc=acc[:max(n,5)]
    cache_set(ck,acc,30*60)
    return acc[:n]

# ===================== QUOTES =====================

ACCIONES_BA=[
 "GGAL.BA","BMA.BA","YPFD.BA","PAMP.BA","CEPU.BA","TGSU2.BA","TGNO4.BA",
 "ALUA.BA","TXAR.BA","LOMA.BA","BYMA.BA","BBAR.BA","VALO.BA","MIRG.BA",
 "SUPV.BA","COME.BA","EDN.BA","TRAN.BA",
]
CEDEARS_BA=[
 "AAPL.BA","MSFT.BA","NVDA.BA","TSLA.BA","AMZN.BA","META.BA","GOOGL.BA",
 "KO.BA","JPM.BA","WMT.BA","DIS.BA","XOM.BA","PFE.BA",
]

def _yahoo_chart_once(symbol:str, rng="1d", interval="1d")->Optional[Dict[str,Any]]:
    """Usa v8/finance/chart por símbolo (evita 401 del endpoint quote)."""
    def _call(host)->httpx.Response:
        url=f"https://{host}/v8/finance/chart/{symbol}"
        return HTTP.get(url, params={"range":rng, "interval":interval, "lang":"es-AR","region":"AR"})
    try:
        r=_call("query1.finance.yahoo.com")
        if r.status_code==401 or r.status_code==403:
            r=_call("query2.finance.yahoo.com")
        if r.status_code==429:
            time.sleep(0.6)
            r=_call("query2.finance.yahoo.com")
        if r.status_code!=200:
            log.warning(f"chart {symbol} status {r.status_code}")
            return None
        j=r.json()
        res=j.get("chart",{}).get("result")
        if not res: return None
        res=res[0]
        meta=res.get("meta",{})
        price=meta.get("regularMarketPrice")
        ch=res.get("indicators",{}).get("quote",[])
        closes=(ch[0].get("close") if ch else None) or []
        # Variación % diaria (último vs penúltimo válido)
        last=None; prev=None
        for v in reversed(closes):
            if v is not None:
                if last is None: last=v
                elif prev is None: prev=v; break
        chg=None; chgp=None
        if last is not None and prev not in (None,0):
            chg=last-prev
            chgp=(chg/prev*100.0)
        if price is None and last is not None: price=last
        return {"price": price, "changePercent": chgp}
    except Exception as e:
        log.warning(f"chart {symbol} error: {e}")
        return None

def get_quotes_acciones()->Dict[str,Dict[str,Any]]:
    ck="q_acc_v4"; c=cache_get(ck)
    if c: return c
    out={}
    for sym in ACCIONES_BA:
        q=_yahoo_chart_once(sym)
        if q: out[sym]=q
        time.sleep(0.25)   # evitar 429
    cache_set(ck,out,15*60)
    return out

def get_quotes_cedears()->Dict[str,Dict[str,Any]]:
    ck="q_ced_v4"; c=cache_get(ck)
    if c: return c
    out={}
    for sym in CEDEARS_BA:
        q=_yahoo_chart_once(sym)
        if q: out[sym]=q
        time.sleep(0.25)
    cache_set(ck,out,15*60)
    return out

def ranking_from_quotes(quotes:Dict[str,Dict[str,Any]], topn=10)->Tuple[List[Tuple[str,float]],List[Tuple[str,float]]]:
    arr=[]
    for sym,q in quotes.items():
        p=q.get("changePercent")
        if isinstance(p,(int,float)): arr.append((sym,float(p)))
    arr.sort(key=lambda x:x[1], reverse=True)
    return arr[:topn], list(reversed(arr[-topn:]))

def weekly_change(symbols:List[str])->Dict[str,float]:
    """Variación % 5 días (semanal) vía chart 5d."""
    out={}
    for sym in symbols:
        try:
            r=HTTP.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}", params={"range":"5d","interval":"1d"})
            if r.status_code!=200:
                r=HTTP.get(f"https://query2.finance.yahoo.com/v8/finance/chart/{sym}", params={"range":"5d","interval":"1d"})
            if r.status_code==200:
                j=r.json()
                res=j.get("chart",{}).get("result")
                if not res: continue
                closes=res[0].get("indicators",{}).get("quote",[{}])[0].get("close",[])
                series=[v for v in closes if v is not None]
                if len(series)>=2 and series[0]!=0:
                    out[sym]=(series[-1]/series[0]-1.0)*100.0
        except Exception:
            pass
        time.sleep(0.25)
    return out

# ===================== COMMANDS ===================

async def cmd_start(update:Update, context:ContextTypes.DEFAULT_TYPE):
    txt = (
        html_bold("Observatorio Económico 🇦🇷")+"\n"
        "Comandos:\n"
        "• /resumen – panorama rápido (dólar, riesgo, inflación, reservas + 5 titulares con link)\n"
        "• /dolar – cotizaciones (blue, mep, ccl, etc.)\n"
        "• /reservas – reservas BCRA (MUSD)\n"
        "• /inflacion – último dato INDEC\n"
        "• /riesgo – riesgo país (EMBI AR)\n"
        "• /acciones – precios/variación BYMA\n"
        "• /cedears – precios/variación CEDEARs\n"
        "• /ranking – top/bottom del día\n"
        "• /semanal – variación semanal (5d)\n"
    )
    await update.message.reply_text(txt, disable_web_page_preview=True)

async def cmd_dolar(update:Update, context:ContextTypes.DEFAULT_TYPE):
    d=get_dolares()
    def v(name,k="venta"):
        x=d.get(name,{}).get(k)
        return "$ "+fmt_num(x) if x is not None else "N/D"
    txt=(
        html_bold("Dólares")+"\n"
        f"• Blue: {v('blue')}\n"
        f"• MEP: {v('mep')}\n"
        f"• CCL: {v('ccl')}\n"
        f"• Oficial: {v('oficial')}\n"
        f"• Mayorista: {v('mayorista')}\n"
        f"• Cripto: {v('cripto')}\n"
    )
    await update.message.reply_text(txt, disable_web_page_preview=True)

async def cmd_riesgo(update:Update, context:ContextTypes.DEFAULT_TYPE):
    r=get_riesgo_pais()
    if not r: 
        await update.message.reply_text("Riesgo País: dato no disponible."); return
    val=r.get("valor")
    fecha=r.get("fecha") or r.get("ultima_actualizacion") or ""
    txt=f"{html_bold('Riesgo País (EMBI AR)')}: {int(val)} pb · {fecha}" if val is not None else "Riesgo País: N/D"
    await update.message.reply_text(txt, disable_web_page_preview=True)

async def cmd_inflacion(update:Update, context:ContextTypes.DEFAULT_TYPE):
    inf=get_inflacion()
    if not inf:
        await update.message.reply_text("Inflación: dato no disponible."); return
    val=inf.get("valor"); fecha=inf.get("fecha","")
    txt=f"{html_bold('Inflación INDEC')}: {float(val):.2f}% · {fecha}" if val is not None else "Inflación: N/D"
    await update.message.reply_text(txt, disable_web_page_preview=True)

async def cmd_reservas(update:Update, context:ContextTypes.DEFAULT_TYPE):
    res=get_reservas()
    if not res:
        await update.message.reply_text("Reservas BCRA: dato no disponible."); return
    n=res.get("valor_musd"); fecha=res.get("fecha")
    val = f"{int(n):,}".replace(",",".") if isinstance(n,(int,float)) else "N/D"
    txt=f"{html_bold('Reservas Internacionales BCRA')}: {val} MUS$"
    if fecha: txt+=f" · {fecha}"
    await update.message.reply_text(txt, disable_web_page_preview=True)

def fmt_quotes_table(q:Dict[str,Dict[str,Any]])->str:
    if not q: return "Sin datos."
    lines=[html_code(f"{'TICKER':<10}{'PRECIO':>12}{'VAR%':>10}")]
    for sym in sorted(q.keys()):
        price=q[sym].get("price"); chp=q[sym].get("changePercent")
        ps = f"{price:,.2f}".replace(",",".") if isinstance(price,(int,float)) else "-"
        cs = f"{chp:+.2f}%" if isinstance(chp,(int,float)) else "-"
        lines.append(html_code(f"{sym:<10}{ps:>12}{cs:>10}"))
    return "\n".join(lines)

async def cmd_acciones(update:Update, context:ContextTypes.DEFAULT_TYPE):
    q=get_quotes_acciones()
    txt=html_bold("Acciones BYMA")+"\n"+fmt_quotes_table(q)
    await update.message.reply_text(txt, disable_web_page_preview=True)

async def cmd_cedears(update:Update, context:ContextTypes.DEFAULT_TYPE):
    q=get_quotes_cedears()
    txt=html_bold("CEDEARs")+"\n"+fmt_quotes_table(q)
    await update.message.reply_text(txt, disable_web_page_preview=True)

async def cmd_ranking(update:Update, context:ContextTypes.DEFAULT_TYPE):
    allq={}; allq.update(get_quotes_acciones()); allq.update(get_quotes_cedears())
    top,bot=ranking_from_quotes(allq, topn=10)
    def block(title, arr):
        lines=[html_bold(title)]
        for sym,pch in arr: lines.append(f"• {html_code(sym)}  {pct(pch)}")
        return "\n".join(lines)
    await update.message.reply_text(block("Top 10",top)+"\n\n"+block("Bottom 10",bot), disable_web_page_preview=True)

async def cmd_resumen(update:Update, context:ContextTypes.DEFAULT_TYPE):
    d=get_dolares()
    mep=d.get("mep",{}).get("venta"); ccl=d.get("ccl",{}).get("venta")
    blue=d.get("blue",{}).get("venta"); ofi=d.get("oficial",{}).get("venta"); may=d.get("mayorista",{}).get("venta")
    r=get_riesgo_pais(); inf=get_inflacion(); res=get_reservas()
    riesgo_txt = f"{int(r['valor'])} pb" if r and r.get("valor") is not None else "N/D"
    infl_txt   = f"{float(inf['valor']):.2f}% ({inf.get('fecha','')})" if inf and inf.get("valor") is not None else "N/D"
    res_txt    = f"{int(res['valor_musd']):,} MUS$".replace(",",".") if res and res.get("valor_musd") is not None else "N/D"

    linea=(
        f"{html_bold('USD')} blue: $ {fmt_num(blue)} | mep: $ {fmt_num(mep)} | ccl: $ {fmt_num(ccl)} | oficial: $ {fmt_num(ofi)} | mayorista: $ {fmt_num(may)}\n"
        f"{html_bold('Riesgo País')}: {riesgo_txt}  ·  {html_bold('Inflación')}: {infl_txt}  ·  {html_bold('Reservas')}: {res_txt}\n"
    )
    items=get_noticias(5)
    news = "\n".join([f"• <a href=\"{html.escape(u)}\">{html.escape(t)}</a>" for (t,u) in items]) if items else "Sin noticias por el momento."
    await update.message.reply_text(linea+"\n"+html_bold("Últimos titulares:")+"\n"+news, disable_web_page_preview=True)

async def cmd_semanal(update:Update, context:ContextTypes.DEFAULT_TYPE):
    # Para hacerlo ágil, tomamos subset representativo
    universe = ACCIONES_BA[:10] + CEDEARS_BA[:10]
    ch = weekly_change(universe)
    if not ch:
        await update.message.reply_text("Resumen semanal: datos no disponibles.")
        return
    top = sorted(ch.items(), key=lambda x:x[1], reverse=True)[:10]
    bot = sorted(ch.items(), key=lambda x:x[1])[:10]
    def block(title, arr):
        lines=[html_bold(title)]
        for sym,p in arr: lines.append(f"• {html_code(sym)}  {pct(p)}")
        return "\n".join(lines)
    await update.message.reply_text(block("Top 10 (5d)", top)+"\n\n"+block("Bottom 10 (5d)", bot), disable_web_page_preview=True)

# ===================== JOBS =======================

async def job_prefetch(context:ContextTypes.DEFAULT_TYPE):
    try:
        get_dolares(); get_riesgo_pais(); get_inflacion(); get_reservas()
        get_quotes_acciones(); get_quotes_cedears()
    except Exception as e:
        log.warning(f"Prefetch job error: {e}")

async def job_news(context:ContextTypes.DEFAULT_TYPE):
    try: get_noticias(5)
    except Exception as e: log.warning(f"News job error: {e}")

async def job_refresh_dolares(context:ContextTypes.DEFAULT_TYPE):
    try: get_dolares()
    except: pass

async def job_refresh_riesgo(context:ContextTypes.DEFAULT_TYPE):
    try: get_riesgo_pais()
    except: pass

async def job_keepalive(context:ContextTypes.DEFAULT_TYPE):
    try: await AHTTP.get(BASE_URL.rstrip("/")+"/ping")
    except: pass

# ===================== APP/WEBHOOK =================

def build_app()->Application:
    defaults = Defaults(parse_mode=ParseMode.HTML, link_preview_options=LinkPreviewOptions(is_disabled=True))
    app = ApplicationBuilder().token(BOT_TOKEN).defaults(defaults).build()

    app.add_handler(CommandHandler("start",   cmd_start))
    app.add_handler(CommandHandler("help",    cmd_start))
    app.add_handler(CommandHandler("dolar",   cmd_dolar))
    app.add_handler(CommandHandler("riesgo",  cmd_riesgo))
    app.add_handler(CommandHandler("inflacion", cmd_inflacion))
    app.add_handler(CommandHandler("reservas",  cmd_reservas))
    app.add_handler(CommandHandler("acciones",  cmd_acciones))
    app.add_handler(CommandHandler("cedears",   cmd_cedears))
    app.add_handler(CommandHandler("ranking",   cmd_ranking))
    app.add_handler(CommandHandler("resumen",   cmd_resumen))
    app.add_handler(CommandHandler("semanal",   cmd_semanal))

    jq=app.job_queue
    jq.run_repeating(job_prefetch,        interval=15*60, first=5)
    jq.run_repeating(job_news,            interval=10*60, first=10)
    jq.run_repeating(job_refresh_dolares, interval=2*60,  first=3)
    jq.run_repeating(job_refresh_riesgo,  interval=5*60,  first=7)
    jq.run_repeating(job_keepalive,       interval=14*60, first=20)
    return app

def main():
    app=build_app()
    webhook_url = BASE_URL.rstrip("/") + WEBHOOK_PATH
    log.info(f"Levantando webhook en {LISTEN}:{PORT} path={WEBHOOK_PATH}")
    log.info(f"Webhook URL = {webhook_url}")
    app.run_webhook(
        listen=LISTEN, port=PORT, url_path=WEBHOOK_PATH.lstrip("/"),
        webhook_url=webhook_url, drop_pending_updates=True,
    )

if __name__=="__main__":
    main()
