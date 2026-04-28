from fastapi.testclient import TestClient

from sre_triage.api.app import create_app


def test_health_and_ready_endpoints():
    client = TestClient(create_app())
    health = client.get("/health")
    ready = client.get("/ready")

    assert health.status_code == 200
    assert ready.status_code == 200
    assert health.json()["status"] == "ok"
    assert ready.json()["status"] == "ready"


def test_reset_accepts_task_id():
    client = TestClient(create_app())
    response = client.post("/reset", json={"task_id": "hard_bad_secret"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["observation"]["metadata"]["task_id"] == "hard_bad_secret"


def test_metrics_endpoint_tracks_requests():
    client = TestClient(create_app())
    client.get("/health")
    metrics = client.get("/metrics")

    assert metrics.status_code == 200
    assert "request_counts" in metrics.json()
