from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "ML API is running"


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict() -> None:
    payload = {
        "alcohol": 14.23,
        "malic_acid": 1.71,
        "ash": 2.43,
        "alcalinity_of_ash": 15.6,
        "magnesium": 127.0,
        "total_phenols": 2.8,
        "flavanoids": 3.06,
        "nonflavanoid_phenols": 0.28,
        "proanthocyanins": 2.29,
        "color_intensity": 5.64,
        "hue": 1.04,
        "od280_od315_of_diluted_wines": 3.92,
        "proline": 1065.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "predicted_class" in body
    assert "predicted_label" in body
    assert "probabilities" in body
