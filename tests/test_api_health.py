from fastapi.testclient import TestClient

from backend.app.main import app


def test_health_endpoint_envelope():
    with TestClient(app) as client:
        response = client.get("/api/v1/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["source"] == "backend"
    assert payload["freshness"] == "current"
    assert "timestamp" in payload
    assert isinstance(payload["warnings"], list)

