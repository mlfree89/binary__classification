from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app.main import app


def test_predict():
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    with patch("app.main.get_db_connection") as mock_db:
        mock_db.return_value.__enter__.return_value = mock_conn

        with TestClient(app) as client:
            response = client.post(
                "/predict",
                json={
                    "age": 35,
                    "workclass": "Private",
                    "fnlwgt": 100000,
                    "education": "Bachelors",
                    "education_num": 13,
                    "marital_status": "Never-married",
                    "occupation": "Sales",
                    "relationship": "Not-in-family",
                    "race": "White",
                    "sex": "Male",
                    "capital_gain": 0,
                    "capital_loss": 0,
                    "hours_per_week": 40,
                    "native_country": "United-States",
                },
            )

            assert response.status_code == 200
            assert "prediction" in response.json()
