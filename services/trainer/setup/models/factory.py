
from . import logistic_regression, neural_network


def create_model(name, X):
    if name == "logistic_regression":
        return logistic_regression.build_model(X)

    if name == "neural_network":
        return neural_network.build_model(X)

    raise ValueError(f"Unknown model: {name}")
