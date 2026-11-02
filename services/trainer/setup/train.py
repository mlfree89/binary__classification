
import os
import logging
import time

import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

from services.trainer.setup.data.loader import load_data
from services.trainer.setup.data.schema_validation import validate_data
from services.trainer.setup.data.splitting import split_data
from services.trainer.setup.models.factory import create_model
from services.trainer.setup.evaluation.evaluate import evaluate_model
from services.trainer.setup.tracking.mlflow_utils import start_run, log_results

load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "./mlruns"))

logger = logging.getLogger(__name__)


def main():
    start_time = time.perf_counter()
    success = 0

    registry = CollectorRegistry()

    duration_metric = Gauge(
        "training_duration_seconds",
        "Total training job duration in seconds",
        registry=registry,
    )

    success_metric = Gauge(
        "training_success",
        "Training job status: 1=success, 0=failure",
        registry=registry,
    )

    try:
        train, test = load_data()
        validate_data(train)
        validate_data(test)

        X_train, X_val, y_train, y_val = split_data(train)

        best_f1 = -1
        best_run_id = None

        for name in ["logistic_regression", "neural_network"]:
            model = create_model(name, X_train)

            with start_run(f"adult-income-{name}") as run:
                model.fit(X_train, y_train)
                metrics = evaluate_model(model, X_val, y_val)

                log_results({"model": name}, metrics)

                mlflow.sklearn.log_model(
                    sk_model=model,
                    name="model",
                    serialization_format="cloudpickle",
                )

                print(name, metrics)

                if metrics["f1"] > best_f1:
                    best_f1 = metrics["f1"]
                    best_run_id = run.info.run_id

        model_uri = f"runs:/{best_run_id}/model"

        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name="adult-income-classifier",
        )

        print("Best F1:", best_f1)
        print("Registered model version:", registered_model.version)

        success = 1

    finally:
        duration_metric.set(time.perf_counter() - start_time)
        success_metric.set(success)

        pushgateway_url = os.getenv(
            "PUSHGATEWAY_URL",
            "http://localhost:9091",
        )

        try:
            push_to_gateway(
                pushgateway_url,
                job="adult_income_training",
                registry=registry,
                timeout=5,
            )
            logger.info("Training metrics pushed to Pushgateway.")

        except Exception:
            logger.exception("Failed to push training metrics.")


if __name__ == "__main__":
    main()
