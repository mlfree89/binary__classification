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
    train = pd.DataFrame([[1] * 15, [2] * 15], columns=COLUMNS)
    test = pd.DataFrame([[3] * 15], columns=COLUMNS)
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
