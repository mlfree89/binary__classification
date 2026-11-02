import os
from dotenv import load_dotenv


load_dotenv(".env")
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5001",
)

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "adult-income-classifier",
)

MODEL_VERSION = os.getenv("MODEL_VERSION", "1")
