from backend.app.services.news_service import _is_relevant, parse_news_feed


def test_parse_news_feed_returns_only_real_linked_entries():
    xml = """
    <rss version="2.0">
      <channel>
        <item>
          <title>Mercado argentino: nueva rueda</title>
          <link>https://example.com/economia/mercado</link>
          <pubDate>Sat, 30 Aug 2026 10:00:00 GMT</pubDate>
          <enclosure url="https://example.com/image.jpg" type="image/jpeg" />
        </item>
        <item><title>Sin enlace</title></item>
      </channel>
    </rss>
    """

    items = parse_news_feed(xml, "https://example.com/feed.xml")

    assert items == [
        {
            "title": "Mercado argentino: nueva rueda",
            "url": "https://example.com/economia/mercado",
            "source": "example.com",
            "published_at": "2026-08-30T10:00:00+00:00",
            "image_url": "https://example.com/image.jpg",
        }
    ]


def test_parse_news_feed_rejects_non_http_links():
    xml = """
    <rss version="2.0"><channel>
      <item><title>Dato</title><link>javascript:alert(1)</link></item>
    </channel></rss>
    """

    assert parse_news_feed(xml, "https://example.com/feed.xml") == []


def test_news_relevance_rejects_sports_even_when_title_contains_banco():
    sports = {
        "title": "River vs. Banfield por el torneo con Ponzio en el banco",
    }
    economic = {
        "title": "El Banco Central ajustó las tasas del mercado",
    }

    assert _is_relevant(sports) is False
    assert _is_relevant(economic) is True
