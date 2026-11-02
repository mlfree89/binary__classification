
import json
import os
import time

import psycopg
from dotenv import load_dotenv
from fastapi import FastAPI
from prometheus_client import Counter, Histogram, make_asgi_app

from .schemas import PredictionRequest, PredictionResponse
from .predictor import Predictor

load_dotenv()

app = FastAPI(title="Adult Income Prediction API")
predictor = None

REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total API requests",
    ["method", "endpoint"],
)

REQUEST_LATENCY = Histogram(
    "api_request_duration_seconds",
    "API request latency",
    ["endpoint"],
)

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


def get_db_connection():
    return psycopg.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )


@app.middleware("http")
async def monitor_requests(request, call_next):
    if request.url.path.startswith("/metrics"):
        return await call_next(request)

    start = time.perf_counter()
    response = await call_next(request)

    route = request.scope.get("route")
    endpoint = route.path if route else "unknown"

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=endpoint,
    ).inc()

    REQUEST_LATENCY.labels(
        endpoint=endpoint,
    ).observe(time.perf_counter() - start)

    return response


@app.on_event("startup")
def startup():
    global predictor

    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id BIGSERIAL PRIMARY KEY,
                input_data JSONB NOT NULL,
                prediction TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )
        """)

    predictor = Predictor()


@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/reload-model")
def reload_model():
    predictor.load_latest_model()
    return {"status": "success", "message": "Latest model loaded"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    data = request.model_dump()
    result = str(predictor.predict(data))

    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO predictions (input_data, prediction)
            VALUES (%s, %s)
            """,
            (json.dumps(data), result),
        )

    return PredictionResponse(prediction=result)
