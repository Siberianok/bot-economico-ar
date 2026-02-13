import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
os.environ.setdefault("TELEGRAM_TOKEN", "test-token")

import bot_econ_full_plus_rank_alerts as bot


def test_catalog_has_versioned_entities_and_fci_schema():
    assert bot.CATALOG_PROVIDER.version
    fci = bot.CATALOG_PROVIDER.fci(only_active=False)
    assert fci
    sample = fci[0]
    for attr in ("id", "nombre", "administradora", "moneda", "tipo", "liquidez", "riesgo", "fuente"):
        assert getattr(sample, attr) is not None


def test_catalog_filters_by_moneda_and_tipo_for_add_menu():
    ars_money_market = bot._filter_catalog_entries("fci", "ARS", "money market")
    assert ars_money_market
    assert all(x.moneda == "ARS" for x in ars_money_market)
    assert all(x.tipo == "money market" for x in ars_money_market)

    usd_soberanos = bot._filter_catalog_entries("bono", "USD", "soberano")
    assert usd_soberanos
    assert all(x.moneda == "USD" for x in usd_soberanos)


def test_menu_keyboards_expose_filter_and_symbol_selection_callbacks():
    filters_kb = bot.kb_catalog_filters("fci", "ARS", "money market")
    callbacks = [b.callback_data for row in filters_kb.inline_keyboard for b in row]
    assert any(cb.startswith("PF:CAT:MON:fci:") for cb in callbacks)
    assert any(cb.startswith("PF:CAT:TIPO:fci:") for cb in callbacks)
    assert "PF:CAT:LIST:fci" in callbacks

    entries = bot._filter_catalog_entries("fci", "ARS", "money market")
    pick_kb = bot.kb_pick_catalog_entries(entries, "fci")
    pick_callbacks = [b.callback_data for row in pick_kb.inline_keyboard for b in row]
    assert any(cb.startswith("PF:PICK:") for cb in pick_callbacks)
