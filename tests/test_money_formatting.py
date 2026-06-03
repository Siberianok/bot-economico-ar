from backend.app.utils.money import format_money, format_number


def test_format_number_argentine_style():
    assert format_number(1234.56) == "1.234,56"


def test_format_money_ars_usd():
    assert format_money(1000, "ARS") == "$ 1.000,00"
    assert format_money(1000, "USD") == "US$ 1.000,00"

