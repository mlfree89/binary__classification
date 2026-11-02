
import pandas as pd
import pytest

from services.trainer.setup.models.factory import create_model


@pytest.mark.parametrize(
    "model_name",
    ["logistic_regression", "neural_network"],
)
def test_model_can_train_and_predict(model_name):
    X = pd.DataFrame({
        "age": [25, 45, 30, 50, 22, 60, 35, 40],
        "workclass": [
            "Private", "Self-emp", "Private", "Gov",
            "Gov", "Private", "Self-emp", "Gov",
        ],
    })
    y = pd.Series([
        "<=50K", ">50K", "<=50K", ">50K",
        "<=50K", ">50K", "<=50K", ">50K",
    ])

    model = create_model(model_name, X)
    model.fit(X, y)
    predictions = model.predict(X)

    assert len(predictions) == len(y)
    assert set(predictions).issubset({"<=50K", ">50K"})
