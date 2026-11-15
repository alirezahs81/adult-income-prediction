import numpy as np
from fastapi.testclient import TestClient

from src import api

client = TestClient(api.app)

EXAMPLE = {
    "age": 39,
    "workclass": "State-gov",
    "fnlwgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}


class FakeModel:
    def predict_proba(
        self: "FakeModel", _frame: object
    ) -> np.ndarray:
        return np.array([[0.2, 0.8]])


def test_health_is_liveness_even_without_model(monkeypatch) -> None:
    monkeypatch.setattr(api, "model", None)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_ready_returns_503_without_model(monkeypatch) -> None:
    monkeypatch.setattr(api, "model", None)
    response = client.get("/ready")
    assert response.status_code == 503


def test_ready_succeeds_with_model(monkeypatch) -> None:
    monkeypatch.setattr(api, "model", FakeModel())
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_prediction_returns_503_without_model(monkeypatch) -> None:
    monkeypatch.setattr(api, "model", None)
    response = client.post("/predict", json=EXAMPLE)
    assert response.status_code == 503


def test_prediction_returns_class_and_probability(monkeypatch) -> None:
    monkeypatch.setattr(api, "model", FakeModel())
    response = client.post("/predict", json=EXAMPLE)
    assert response.status_code == 200
    assert response.json() == {"prediction": ">50K", "probability": 0.8}
    assert "confidence" not in response.json()


def test_invalid_age_is_rejected() -> None:
    response = client.post("/predict", json={**EXAMPLE, "age": 5})
    assert response.status_code == 422


def test_example_endpoint() -> None:
    response = client.get("/example")
    assert response.status_code == 200
    assert "example_input" in response.json()


def test_batch_limit_is_validated() -> None:
    response = client.post("/predict/batch", json={"instances": []})
    assert response.status_code == 422
}