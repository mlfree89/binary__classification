
import pandas as pd

from services.trainer.setup.preprocessing.pipeline import build_preprocessor


def test_preprocessor_handles_missing_and_categorical_values():
    X = pd.DataFrame({
        "age": [25, None, 40],
        "workclass": ["Private", "Self-emp", None],
    })

    preprocessor = build_preprocessor(X)
    transformed = preprocessor.fit_transform(X)

    assert transformed.shape[0] == 3
    assert transformed.shape[1] > 0
    assert not pd.isna(transformed).any()
