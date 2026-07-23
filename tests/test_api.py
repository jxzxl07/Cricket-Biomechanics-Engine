import io

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_returns_model_status(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "degraded"}
    assert set(body["models"]) == {"bowling", "batting"}


def test_predict_rejects_unknown_mode(client):
    response = client.post(
        "/predict?mode=cricket",
        files={"file": ("clip.mp4", io.BytesIO(b"x"), "video/mp4")},
    )
    assert response.status_code == 422


def test_predict_rejects_wrong_file_type(client):
    response = client.post(
        "/predict?mode=bowling",
        files={"file": ("notes.txt", io.BytesIO(b"x"), "text/plain")},
    )
    assert response.status_code == 400


def test_predict_rejects_empty_file(client):
    response = client.post(
        "/predict?mode=bowling",
        files={"file": ("clip.mp4", io.BytesIO(b""), "video/mp4")},
    )
    assert response.status_code == 400