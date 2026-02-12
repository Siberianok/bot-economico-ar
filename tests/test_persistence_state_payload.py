from bot.persistence.state import CURRENT_STATE_VERSION, deserialize_state_payload


def test_deserialize_state_payload_cleans_dirty_payload_and_migrates(dirty_state_payload):
    payload = deserialize_state_payload(dirty_state_payload)

    assert payload["version"] == CURRENT_STATE_VERSION
    assert list(payload["alerts"].keys()) == [100]

    clean_rule = payload["alerts"][100][0]
    assert clean_rule["kind"] == "ticker"
    assert clean_rule["last_trigger_ts"] == 1711111111.0
    assert clean_rule["last_trigger_price"] == 999.9
    assert clean_rule["armed"] is False

    assert payload["subs"][100]["daily"] == "930"
    assert payload["pf"][100]["items"] == [{"simbolo": "GGAL.BA"}]

    assert payload["projection_records"] == [
        {
            "symbol": "GGAL.BA",
            "horizon": 63,
            "base_price": 100.0,
            "projection": 10.0,
            "created_at": 1700000000.0,
            "created_date": "2024-01-01",
            "batch_id": "batch-1",
        }
    ]

    assert payload["projection_batches"] == [
        {
            "batch_id": "batch-1",
            "horizon": 63,
            "created_at": 1700000000.0,
            "symbols": ["GGAL.BA"],
            "predictions": {"GGAL.BA": 8.0},
            "base_prices": {"GGAL.BA": 100.0},
            "created_date": "2024-01-01",
        }
    ]


def test_deserialize_state_payload_handles_non_dict_payload():
    assert deserialize_state_payload(None) == {}
    assert deserialize_state_payload(["bad"]) == {}
