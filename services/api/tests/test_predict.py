from fastapi.testclient import TestClient
from app.main import app


def test_predict():
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
