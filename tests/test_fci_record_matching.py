import bot_econ_full_plus_rank_alerts as mod


def test_pick_fci_record_prioritizes_preferred_identifier_over_keyword():
    records = [
        {
            "fundId": "A1",
            "ticker": "MM-ALPHA",
            "fundName": "Fondo Money Market Alpha",
            "fundStrategy": "Liquidez ARS",
            "fundCurrency": "ARS",
        },
        {
            "fundId": "B2",
            "ticker": "MM-BETA",
            "fundName": "Fondo Money Market Beta",
            "fundStrategy": "Liquidez ARS",
            "fundCurrency": "ARS",
        },
    ]

    picked, confident, selected_id = mod._pick_fci_record(
        "FCI-MoneyMarket",
        records,
        preferred_identifier="B2",
    )

    assert confident is True
    assert picked == records[1]
    assert selected_id == "B2"


def test_pick_fci_record_weighted_scoring_with_threshold():
    records = [
        {
            "fundId": "MM-USD-1",
            "fundName": "Fondo Money Market Dólar",
            "fundStrategy": "Cash management en USD",
            "fundCurrency": "USD",
            "managerName": "Administradora XYZ",
        },
        {
            "fundId": "MM-ARS-1",
            "fundName": "Fondo Liquidez ARS",
            "fundStrategy": "Liquidez inmediata",
            "fundCurrency": "ARS",
            "managerName": "Administradora XYZ",
        },
    ]

    picked, confident, selected_id = mod._pick_fci_record("FCI-BonosUSD", records)

    assert confident is True
    assert picked == records[0]
    assert selected_id == "MM-USD-1"


def test_pick_fci_record_returns_none_when_ambiguous_and_low_confidence():
    records = [
        {
            "fundId": "MM-1",
            "fundName": "Fondo Money Market A",
            "fundStrategy": "Liquidez",
            "fundCurrency": "ARS",
        },
        {
            "fundId": "MM-2",
            "fundName": "Fondo Money Market B",
            "fundStrategy": "Liquidez",
            "fundCurrency": "ARS",
        },
    ]

    picked, confident, selected_id = mod._pick_fci_record("FCI-MoneyMarket", records)

    assert picked is None
    assert confident is False
    assert selected_id is None
