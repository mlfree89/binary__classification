import pandas as pd

from services.trainer.setup.data.loader import load_data
from services.trainer.setup.data.schema_validation import validate_data


COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income",
]


def mock_minio():
    train = pd.DataFrame([
        [35, "Private", 100000, "Bachelors", 13, "Never-married", "Sales", "Not-in-family", "White", "Male", 0, 0, 40, "United-States", "<=50K"],
        [45, "Self-emp", 200000, "Masters", 14, "Married", "Tech", "Husband", "White", "Male", 0, 0, 50, "United-States", ">50K"],
    ], columns=COLUMNS)

    test = pd.DataFrame([
        [25, "Private", 50000, "HS-grad", 9, "Never-married", "Sales", "Not-in-family", "White", "Female", 0, 0, 35, "United-States", "<=50K"],
    ], columns=COLUMNS)

    return train, test

def mock_postgres():
    return None


def test_load_data(monkeypatch):
    monkeypatch.setattr("services.trainer.setup.data.loader._load_from_minio", mock_minio)
    monkeypatch.setattr("services.trainer.setup.data.loader._load_from_postgres", mock_postgres)

    train, test = load_data()

    assert not train.empty
    assert not test.empty
    assert len(train.columns) == 15
    assert len(test.columns) == 15


def test_validate_data(monkeypatch):
    monkeypatch.setattr("services.trainer.setup.data.loader._load_from_minio", mock_minio)
    monkeypatch.setattr("services.trainer.setup.data.loader._load_from_postgres", mock_postgres)

    train, test = load_data()

    assert validate_data(train) is True
    assert validate_data(test) is True
