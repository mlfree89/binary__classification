
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from ..preprocessing.pipeline import build_preprocessor


def build_model(X):
    return Pipeline([
        ("preprocessor", build_preprocessor(X)),
        ("classifier", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=300,
            early_stopping=False,
            random_state=42,
        )),
    ])

