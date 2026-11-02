
from evidently_monitoring.app.main import extract_drift_flag


def test_extract_drift_flag_detects_drift():
    class FakeResult:
        def dict(self):
            return {
                "metrics": [{
                    "metric_name": "DriftedColumnsCount",
                    "value": {"count": 8, "share": 0.6},
                }]
            }

    assert extract_drift_flag(FakeResult()) is True


def test_extract_drift_flag_no_drift():
    class FakeResult:
        def dict(self):
            return {
                "metrics": [{
                    "metric_name": "DriftedColumnsCount",
                    "value": {"count": 2, "share": 0.2},
                }]
            }

    assert extract_drift_flag(FakeResult()) is False
