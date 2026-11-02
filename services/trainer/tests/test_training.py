
from services.trainer.setup.evaluation.metrics import calculate_metrics


def test_metrics_are_valid():
    y_true = ["<=50K", ">50K", ">50K", "<=50K"]
    y_pred = ["<=50K", ">50K", "<=50K", "<=50K"]

    metrics = calculate_metrics(y_true, y_pred)

    assert metrics["accuracy"] == 0.75
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["f1"] <= 1
