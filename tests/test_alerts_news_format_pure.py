import bot_econ_full_plus_rank_alerts as bot


def test_alert_can_trigger_with_tolerance_and_cooldown():
    now = 2_000_000.0
    rule = {"op": ">", "value": 100.0, "armed": True}

    ok, changed = bot._alert_can_trigger(rule, 99.85, now)
    assert (ok, changed) == (True, False)

    rule["last_trigger_ts"] = now - 60
    ok, changed = bot._alert_can_trigger(rule, 100.0, now)
    assert (ok, changed) == (False, False)


def test_alert_rearm_flow_and_anti_duplicate_detection():
    now = 2_000_000.0
    rule = {"op": ">", "value": 100.0, "armed": False, "last_trigger_ts": now - 2000}

    ok, changed = bot._alert_can_trigger(rule, 99.0, now)
    assert (ok, changed) == (False, True)
    assert rule["armed"] is True

    rules = [
        {"kind": "ticker", "symbol": "GGAL.BA", "op": ">", "value": 100.0},
        {"kind": "ticker", "symbol": "BMA.BA", "op": "<", "value": 300.0},
    ]
    candidate = {"kind": "ticker", "symbol": "GGAL.BA", "op": ">", "value": 100.0}

    assert bot._has_duplicate_alert(rules, candidate) is True
    assert bot._has_duplicate_alert(rules, candidate, skip_idx=0) is False


def test_news_link_normalization_dedup_and_filter():
    wrapped = (
        "https://news.google.com/rss/articles/CBMiSWh0dHBzOi8vZXhhbXBsZS5jb20vbm90YS0xMjM"
        "?hl=es-419&gl=AR&ceid=AR:es-419&url=https://example.com/nota-123?utm_source=rss"
    )
    assert bot._normalize_feed_link(wrapped) == "https://example.com/nota-123?utm_source=rss"

    items = [
        ("Dólar sube con fuerza en Argentina", "https://example.com/nota-123?utm=1", None),
        ("Dolar sube con fuerza en argentina", "https://example.com/nota-123?utm=2", None),
        ("Mercados globales en rojo", "https://example.com/otra-nota-456", None),
    ]
    deduped = bot._dedup_news_items(items)
    assert len(deduped) == 2

    assert bot._is_economic_relevant("Sube el dólar blue", None) is True
    assert bot._is_economic_relevant("Festival de cine independiente", "Cartelera del fin de semana") is False


def test_format_outputs_for_key_blocks_are_stable():
    msg = bot.format_dolar_message(
        {
            "blue": {"compra": 1200.0, "venta": 1220.0, "variation": 1.1, "fecha": "2025-01-01"},
            "mep": {"compra": 1180.0, "venta": 1195.0, "variation": -0.4, "fecha": "2025-01-01"},
        }
    )
    assert "<b>💵 Dólares</b>" in msg
    assert "<b>📥 Compra</b>" in msg
    assert "<b>📤 Venta</b>" in msg
    assert "$ 1.220,00" in msg

    news_block, _ = bot.format_news_block([("Título de prueba", "https://example.com/a", None)])
    assert "<b>📰 Noticias</b>" in news_block
    assert "Título de prueba" in news_block
    assert "https://example.com/a" in news_block
