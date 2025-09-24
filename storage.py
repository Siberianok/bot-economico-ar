# storage.py
# -*- coding: utf-8 -*-
import os, json, time, base64, hashlib
from typing import Dict, List, Optional, Tuple
import redis.asyncio as redis

# ---------- Config ----------
REDIS_URL = os.getenv("REDIS_URL")
PREFIX    = os.getenv("REDIS_PREFIX", "bot-ar")   # opcional para namespacing

_r: Optional[redis.Redis] = None

def _k(*parts: str) -> str:
    return ":".join([PREFIX, *parts])

async def _client() -> redis.Redis:
    global _r
    if _r is None:
        if not REDIS_URL:
            raise RuntimeError("REDIS_URL no definido")
        _r = redis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)
    return _r

# ---------- Diagnóstico ----------
async def redis_ping() -> str:
    try:
        r = await _client()
        pong = await r.ping()
        return "Redis (ping OK)" if pong else "Redis (ping FAIL)"
    except Exception as e:
        return f"Redis (error: {e.__class__.__name__})"

# ---------- Alertas ----------
# Estructura:
#   HSET  alerts:{chat_id}  <alert_id>  <json>
#   SADD  alerts:chats  {chat_id}
#   Hash json incluye: {"created_at": int(ts), ...}
def _k_alerts(chat_id: int) -> str:
    return _k("state", "alerts", str(chat_id))

def _k_alerts_chats() -> str:
    return _k("state", "alerts_chats")

# Pausas de alertas:
#   String alerts:pause:{chat_id} -> "inf"  |  "until:<ts>"
def _k_alerts_pause(chat_id: int) -> str:
    return _k("state", "alerts_pause", str(chat_id))

def _make_id(payload: Dict) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    h = hashlib.sha1(raw + b"|" + str(time.time()).encode("ascii")).digest()[:9]
    return base64.urlsafe_b64encode(h).decode("ascii").rstrip("=")

async def alerts_add(chat_id: int, rule: Dict) -> str:
    r = await _client()
    if "created_at" not in rule:
        rule["created_at"] = int(time.time())
    aid = _make_id(rule)
    await r.hset(_k_alerts(chat_id), aid, json.dumps(rule, ensure_ascii=False))
    await r.sadd(_k_alerts_chats(), chat_id)
    return aid

async def alerts_list(chat_id: int) -> List[Dict]:
    r = await _client()
    items = await r.hgetall(_k_alerts(chat_id))
    out: List[Dict] = []
    for aid, raw in items.items():
        try:
            d = json.loads(raw)
            d["_id"] = aid
            out.append(d)
        except Exception:
            continue
    out.sort(key=lambda x: x.get("created_at", 0))  # orden estable por creación
    return out

async def alerts_del_all(chat_id: int) -> int:
    r = await _client()
    removed = await r.delete(_k_alerts(chat_id))
    await r.srem(_k_alerts_chats(), chat_id)
    return removed

async def alerts_del_by_index(chat_id: int, idx: int) -> bool:
    lst = await alerts_list(chat_id)
    if not (0 <= idx < len(lst)): return False
    aid = lst[idx]["_id"]
    r = await _client()
    await r.hdel(_k_alerts(chat_id), aid)
    return True

async def alert_chats_all() -> List[int]:
    r = await _client()
    ids = await r.smembers(_k_alerts_chats())
    return [int(x) for x in ids] if ids else []

# ---------- Portafolios ----------
# Guardamos cada portafolio como JSON individual para cada chat.

def _k_pf(chat_id: int) -> str:
    return _k("state", "pf", str(chat_id))

def _k_pf_chats() -> str:
    return _k("state", "pf_chats")

async def pf_set(chat_id: int, payload: Dict) -> None:
    r = await _client()
    await r.set(_k_pf(chat_id), json.dumps(payload, ensure_ascii=False))
    await r.sadd(_k_pf_chats(), chat_id)

async def pf_get(chat_id: int) -> Optional[Dict]:
    r = await _client()
    raw = await r.get(_k_pf(chat_id))
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None

async def pf_del(chat_id: int) -> None:
    r = await _client()
    await r.delete(_k_pf(chat_id))
    await r.srem(_k_pf_chats(), chat_id)

async def pf_chats_all() -> List[int]:
    r = await _client()
    ids = await r.smembers(_k_pf_chats())
    return [int(x) for x in ids] if ids else []

# Pausas
async def alerts_pause_indef(chat_id: int):
    r = await _client()
    await r.set(_k_alerts_pause(chat_id), "inf")

async def alerts_pause_hours(chat_id: int, hours: int):
    until = int(time.time() + hours*3600)
    r = await _client()
    await r.set(_k_alerts_pause(chat_id), f"until:{until}")

async def alerts_pause_until(chat_id: int, until_ts: int):
    r = await _client()
    await r.set(_k_alerts_pause(chat_id), f"until:{int(until_ts)}")

async def alerts_resume(chat_id: int):
    r = await _client()
    await r.delete(_k_alerts_pause(chat_id))

async def alerts_pause_status(chat_id: int) -> Dict:
    r = await _client()
    v = await r.get(_k_alerts_pause(chat_id))
    now = int(time.time())
    if not v:
        return {"paused": False, "indef": False, "until": None, "active": True}
    if v == "inf":
        return {"paused": True, "indef": True, "until": None, "active": False}
    if v.startswith("until:"):
        try:
            ts = int(v.split(":", 1)[1])
            if ts > now:
                return {"paused": True, "indef": False, "until": ts, "active": False}
            else:
                # expiró; limpiar
                await r.delete(_k_alerts_pause(chat_id))
                return {"paused": False, "indef": False, "until": None, "active": True}
        except Exception:
            await r.delete(_k_alerts_pause(chat_id))
            return {"paused": False, "indef": False, "until": None, "active": True}
    # valor inválido
    await r.delete(_k_alerts_pause(chat_id))
    return {"paused": False, "indef": False, "until": None, "active": True}

# ---------- Suscripciones (Resumen Diario) ----------
#   HSET  subs:{chat_id}  sub  {"hhmm":"13:00","tz":"America/Argentina/Buenos_Aires","paused":false}
#   SADD  subs:chats  {chat_id}
def _k_subs(chat_id: int) -> str:
    return _k("state", "subs", str(chat_id))

def _k_subs_chats() -> str:
    return _k("state", "subs_chats")

async def subs_set(chat_id: int, hhmm: str, tz: str = "America/Argentina/Buenos_Aires", paused: bool = False):
    r = await _client()
    payload = {"hhmm": hhmm, "tz": tz, "paused": bool(paused), "updated_at": int(time.time())}
    await r.hset(_k_subs(chat_id), "sub", json.dumps(payload, ensure_ascii=False))
    await r.sadd(_k_subs_chats(), chat_id)

async def subs_get(chat_id: int) -> Optional[Dict]:
    r = await _client()
    raw = await r.hget(_k_subs(chat_id), "sub")
    if not raw: return None
    try:
        return json.loads(raw)
    except Exception:
        return None

async def subs_del(chat_id: int):
    r = await _client()
    await r.delete(_k_subs(chat_id))

async def subs_chats_all() -> List[int]:
    r = await _client()
    ids = await r.smembers(_k_subs_chats())
    return [int(x) for x in ids] if ids else []

# ---------- Contadores útiles ----------
async def counts() -> Tuple[int, int]:
    return len(await alert_chats_all()), len(await subs_chats_all())
