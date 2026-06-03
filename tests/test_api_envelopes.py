from fastapi.testclient import TestClient

from backend.app.main import app


REQUIRED_KEYS = {"status", "timestamp", "source", "freshness", "data", "warnings"}


def assert_envelope(payload):
    assert REQUIRED_KEYS.issubset(payload.keys())
    assert isinstance(payload["warnings"], list)


def test_initial_endpoints_return_envelopes():
    paths = [
        "/api/v1/market/pulse",
        "/api/v1/screener?kind=acciones",
        "/api/v1/screener?kind=cedears",
        "/api/v1/portfolio/summary",
        "/api/v1/alerts",
        "/api/v1/signals",
        "/api/v1/config",
        "/api/v1/projections",
        "/api/v1/validations",
    ]
    with TestClient(app) as client:
        for path in paths:
            response = client.get(path)
            assert response.status_code == 200, path
            assert_envelope(response.json())


def test_screener_invalid_kind_returns_structured_error():
    with TestClient(app) as client:
        response = client.get("/api/v1/screener?kind=bonos")
    assert response.status_code == 400
    payload = response.json()
    assert_envelope(payload)
    assert payload["status"] == "error"

