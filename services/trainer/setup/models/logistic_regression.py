from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ..preprocessing.pipeline import build_preprocessor


def build_model(X):
    return Pipeline([
        ("preprocessor", build_preprocessor(X)),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])
