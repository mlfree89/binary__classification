
from unittest.mock import patch
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


@patch("app.main.subprocess.run")
def test_trigger_training_success(mock_run):
    response = client.post("/train")

    assert response.status_code == 200
    assert response.json() == {"status": "training_started"}


@patch("app.main.subprocess.run")
def test_trigger_training_failure(mock_run):
    mock_run.side_effect = FileNotFoundError("Docker not found")

    response = client.post("/train")

    assert response.status_code == 200
    assert response.json() == {"status": "training_started"}
