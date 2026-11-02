
import pandas as pd

from services.trainer.setup.data.loader import load_data
from services.trainer.setup.data.schema_validation import validate_data


def mock_data():
    columns = [f"col_{i}" for i in range(15)]

    train = pd.DataFrame([[1] * 15, [2] * 15], columns=columns)
    test = pd.DataFrame([[3] * 15], columns=columns)

    return train, test


def test_load_data(monkeypatch):
    monkeypatch.setattr(
        "services.trainer.setup.data.loader._load_from_minio",
        mock_data
    )

    train, test = load_data()

    assert not train.empty
    assert not test.empty
    assert len(train.columns) == 15
    assert len(test.columns) == 15


def test_validate_data(monkeypatch):
    monkeypatch.setattr(
        "services.trainer.setup.data.loader._load_from_minio",
        mock_data
    )

    train, test = load_data()

    assert validate_data(train) is True
    assert validate_data(test) is True

