"""Read-only RSS news service using the legacy feed catalog."""

from __future__ import annotations

import asyncio
import copy
import html
import re
import time
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlparse
from xml.etree import ElementTree as ET

from aiohttp import ClientSession, ClientTimeout

from backend.app.config import get_settings
from backend.app.schemas.envelope import envelope
from backend.app.services.legacy_adapter import legacy_adapter
from backend.app.utils.time import utc_now_iso


_TAG_RE = re.compile(r"<[^>]+>")
_IMAGE_RE = re.compile(r"<img[^>]+src=[\"']([^\"']+)", re.IGNORECASE)
_RELEVANCE_TERMS = (
    "argentina",
    "inflacion",
    "dolar",
    "mep",
    "ccl",
    "reservas",
    "bcra",
    "riesgo pais",
    "mercado",
    "acciones",
    "bonos",
    "deuda",
    "fmi",
    "actividad",
    "empleo",
    "salario",
    "consumo",
    "inversion",
    "comercio",
    "industria",
    "empresas",
    "energia",
    "petroleo",
    "export",
    "import",
    "credito",
    "banco",
    "tasas",
    "reforma",
    "recaudacion",
    "deficit",
    "superavit",
    "produccion",
    "vaca muerta",
    "rigi",
)
_SPORTS_TITLE_RE = re.compile(
    r"\b(?:vs\.?|torneo|futbol|copa libertadores|gol|seleccion|mundial de clubes)\b",
    re.IGNORECASE,
)
_NEWS_CACHE_TTL_SECONDS = 120.0
_NEWS_FETCH_LIMIT = 30
_news_cache: tuple[float, dict[str, Any]] | None = None


def _clean_text(value: str | None) -> str:
    return html.unescape(_TAG_RE.sub(" ", value or "")).strip()


def _published_iso(value: str | None) -> str | None:
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
        return parsed.isoformat()
    except (TypeError, ValueError, OverflowError):
        return value.strip() or None


def _image_from_item(item: ET.Element) -> str | None:
    media = item.find("{http://search.yahoo.com/mrss/}content")
    if media is not None and media.attrib.get("url"):
        return media.attrib["url"]
    thumbnail = item.find("{http://search.yahoo.com/mrss/}thumbnail")
    if thumbnail is not None and thumbnail.attrib.get("url"):
        return thumbnail.attrib["url"]
    for enclosure in item.findall("enclosure"):
        if enclosure.attrib.get("url") and enclosure.attrib.get("type", "").startswith("image/"):
            return enclosure.attrib["url"]
    html_blocks = [
        item.findtext("description"),
        item.findtext("{http://purl.org/rss/1.0/modules/content/}encoded"),
    ]
    for block in html_blocks:
        match = _IMAGE_RE.search(html.unescape(block or ""))
        if match and match.group(1).startswith(("http://", "https://")):
            return match.group(1)
    return None


def _normalized_title(title: str) -> str:
    value = title.lower()
    replacements = str.maketrans("áéíóúüñ", "aeiouun")
    return value.translate(replacements)


def _is_relevant(entry: dict[str, Any]) -> bool:
    title = _normalized_title(str(entry.get("title") or ""))
    if any(prefix in title for prefix in ("euro hoy en ", "dolar hoy en ")) and "argentina" not in title:
        return False
    if _SPORTS_TITLE_RE.search(title):
        return False
    return any(term in title for term in _RELEVANCE_TERMS)


def _published_sort_value(entry: dict[str, Any]) -> float:
    raw = entry.get("published_at")
    if not raw:
        return 0.0
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError, OverflowError):
        return 0.0


def parse_news_feed(xml_text: str, feed_url: str) -> list[dict[str, Any]]:
    """Parse real RSS/Atom entries. Invalid or incomplete entries are discarded."""
    root = ET.fromstring(xml_text)
    source = urlparse(feed_url).netloc.removeprefix("www.")
    items: list[dict[str, Any]] = []

    for item in root.findall(".//item"):
        title = _clean_text(item.findtext("title"))
        link = _clean_text(item.findtext("link"))
        if not title or not link.startswith(("http://", "https://")):
            continue
        items.append(
            {
                "title": title,
                "url": link,
                "source": source,
                "published_at": _published_iso(item.findtext("pubDate")),
                "image_url": _image_from_item(item),
            }
        )

    if items:
        return items

    for entry in root.findall(".//{*}entry"):
        title = _clean_text(entry.findtext("{*}title"))
        link_element = entry.find("{*}link")
        link = link_element.attrib.get("href", "") if link_element is not None else ""
        if not title or not link.startswith(("http://", "https://")):
            continue
        items.append(
            {
                "title": title,
                "url": link,
                "source": source,
                "published_at": _published_iso(
                    entry.findtext("{*}published") or entry.findtext("{*}updated")
                ),
                "image_url": None,
            }
        )
    return items


def _limited_payload(payload: dict[str, Any], limit: int) -> dict[str, Any]:
    result = copy.deepcopy(payload)
    data = result.get("data")
    if not isinstance(data, dict):
        return result
    items = data.get("items")
    if not isinstance(items, list):
        return result
    data["items"] = items[:limit]
    data["sources"] = sorted({item["source"] for item in data["items"]})
    return result


async def get_news(limit: int = 12) -> dict[str, Any]:
    global _news_cache
    settings = get_settings()
    if not settings.legacy_live_fetch_enabled:
        return envelope(
            status="not_available",
            timestamp=utc_now_iso(),
            source="news_service",
            freshness="unknown",
            data={"items": [], "sources": []},
            warnings=["Noticias live desactivadas por LEGACY_LIVE_FETCH_ENABLED"],
        )

    now = time.monotonic()
    if _news_cache and now - _news_cache[0] < _NEWS_CACHE_TTL_SECONDS:
        return _limited_payload(_news_cache[1], limit)

    catalog = legacy_adapter.list_news_feeds()
    if catalog.status != "ok" or not catalog.data:
        return envelope(
            status="not_available",
            timestamp=utc_now_iso(),
            source=catalog.source,
            freshness="unknown",
            data={"items": [], "sources": []},
            warnings=catalog.warnings,
        )

    timeout_seconds = min(settings.legacy_market_fetch_timeout_seconds, 5.0)
    timeout = ClientTimeout(total=timeout_seconds)
    warnings: list[str] = []

    async with ClientSession(
        timeout=timeout,
        headers={"User-Agent": "ObeconBot/2.0 RSS reader"},
    ) as session:
        async def fetch_feed(url: str) -> tuple[str, list[dict[str, Any]]]:
            try:
                async with session.get(url) as response:
                    if response.status >= 400:
                        warnings.append(f"{urlparse(url).netloc}: HTTP {response.status}")
                        return url, []
                    body = await response.text(errors="ignore")
                return url, parse_news_feed(body, url)
            except (TimeoutError, ET.ParseError) as exc:
                warnings.append(f"{urlparse(url).netloc}: {exc.__class__.__name__}")
                return url, []
            except Exception as exc:
                warnings.append(f"{urlparse(url).netloc}: {exc.__class__.__name__}")
                return url, []

        results = await asyncio.gather(*(fetch_feed(url) for url in catalog.data))

    deduped: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    for _, entries in results:
        for entry in entries:
            title_key = re.sub(r"\W+", " ", entry["title"].lower()).strip()
            if entry["url"] in seen_urls or title_key in seen_titles:
                continue
            seen_urls.add(entry["url"])
            seen_titles.add(title_key)
            deduped.append(entry)

    relevant = [entry for entry in deduped if _is_relevant(entry)]
    selection_pool = relevant
    by_source: dict[str, list[dict[str, Any]]] = {}
    for entry in selection_pool:
        by_source.setdefault(entry["source"], []).append(entry)
    for source_items in by_source.values():
        source_items.sort(key=_published_sort_value, reverse=True)

    selected: list[dict[str, Any]] = []
    while len(selected) < _NEWS_FETCH_LIMIT and any(by_source.values()):
        for source in sorted(by_source):
            if by_source[source] and len(selected) < _NEWS_FETCH_LIMIT:
                selected.append(by_source[source].pop(0))

    deduped = selected

    if not deduped:
        return envelope(
            status="not_available",
            timestamp=utc_now_iso(),
            source="rss_direct",
            freshness="unknown",
            data={"items": [], "sources": []},
            warnings=warnings or ["Ninguna fuente RSS devolvio noticias validas"],
        )

    sources = sorted({entry["source"] for entry in deduped})
    payload = envelope(
        status="partial" if warnings else "ok",
        timestamp=utc_now_iso(),
        source="rss_direct",
        freshness="current",
        data={"items": deduped, "sources": sources},
        warnings=warnings,
    )
    _news_cache = (time.monotonic(), payload)
    return _limited_payload(payload, limit)
