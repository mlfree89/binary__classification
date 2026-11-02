import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from .config import MLFLOW_TRACKING_URI, MODEL_NAME


class Predictor:
    def __init__(self):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        self.client = MlflowClient()

        self.load_latest_model()

    def load_latest_model(self):
        versions = self.client.search_model_versions(
            f"name='{MODEL_NAME}'"
        )

        if not versions:
            raise RuntimeError(f"No registered versions found for {MODEL_NAME}")

        latest = max(versions, key=lambda version: int(version.version))

        model_uri = f"models:/{MODEL_NAME}/{latest.version}"
        self.model = mlflow.pyfunc.load_model(model_uri)

        print(f"Loaded {MODEL_NAME} version {latest.version}")

    def predict(self, data: dict):
        dataframe = pd.DataFrame([data])
        return self.model.predict(dataframe)[0]
