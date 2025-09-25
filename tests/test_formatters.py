from bot_econ.formatters import fmt_money_ars, fmt_money_usd, fmt_number, fmt_pct


def test_fmt_number_locale():
    assert fmt_number(1234.5) == "1.234,50"


def test_fmt_money():
    assert fmt_money_ars(10) == "$ 10,00"
    assert fmt_money_usd(10) == "US$ 10,00"


def test_fmt_pct():
    assert fmt_pct(1.234) == "+1,23%"
    assert fmt_pct(None) == "—"
