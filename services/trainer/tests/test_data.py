from services.trainer.setup.data.loader import load_data
from services.trainer.setup.data.schema_validation import validate_data


def test_load_data():
    train, test = load_data()

    assert not train.empty
    assert not test.empty
    assert len(train.columns) == 15
    assert len(test.columns) == 15


def test_validate_data():
    train, test = load_data()

    assert validate_data(train) is True
    assert validate_data(test) is True
