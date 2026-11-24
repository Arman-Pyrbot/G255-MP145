"""Tests for the FastAPI service. Self-contained: no CSVs required."""

from fastapi.testclient import TestClient

from app.main import app, PredictRequest

client = TestClient(app)

VALID = {
    "age": 40,
    "screen_hours": 6.0,
    "apps_installed": 100,
    "plan_gb_limit": 3.5,
    "video_minutes": 150,
    "wifi_hours": 8.0,
    "phone_tier": "mid",
    "circle": "Urban",
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_a_number():
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    assert isinstance(r.json()["prediction"], (int, float))


def test_rejects_screen_hours_above_20():
    payload = {**VALID, "screen_hours": 21}
    assert client.post("/predict", json=payload).status_code == 422


def test_rejects_apps_installed_below_5():
    payload = {**VALID, "apps_installed": 4}
    assert client.post("/predict", json=payload).status_code == 422


def test_schema_has_exactly_the_required_fields():
    fields = set(PredictRequest.model_fields)
    assert fields == {
        "age", "screen_hours", "apps_installed", "plan_gb_limit",
        "video_minutes", "wifi_hours", "phone_tier", "circle",
    }


def test_rejects_missing_field():
    payload = {k: v for k, v in VALID.items() if k != "wifi_hours"}
    assert client.post("/predict", json=payload).status_code == 422
