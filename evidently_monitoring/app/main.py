
import json
import logging
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
# from services.trainer.setup.train import main as train_main
import pandas as pd
from dotenv import load_dotenv
from evidently import Report
from evidently.presets import DataDriftPreset
from psycopg_pool import ConnectionPool
import requests

load_dotenv(os.path.expanduser("~/class_app/evidently_monitoring/.env"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
]

BATCH_SIZE = int(os.getenv("NEW_PREDICTIONS_BATCH", "2"))
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL_SECONDS", "30"))
DRIFT_SHARE_THRESHOLD = 0.5


def build_pool():
    return ConnectionPool(
        conninfo=(
            f"host={os.environ['POSTGRES_HOST']} "
            f"port={os.getenv('POSTGRES_PORT', '5432')} "
            f"dbname={os.environ['POSTGRES_DB']} "
            f"user={os.environ['POSTGRES_USER']} "
            f"password={os.environ['POSTGRES_PASSWORD']}"
        ),
        min_size=1,
        max_size=5,
        timeout=10,
    )


import io
import boto3

def load_reference():
    s3 = boto3.client(
        "s3",
        endpoint_url=os.getenv("MINIO_ENDPOINT"),
        aws_access_key_id=os.getenv("MINIO_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("MINIO_SECRET_KEY"),
    )
    bucket = os.getenv("MINIO_BUCKET", "datasets")
    columns = COLUMNS + ["income"]

    def read(key, comment=None):
        obj = s3.get_object(Bucket=bucket, Key=key)
        return pd.read_csv(
            io.BytesIO(obj["Body"].read()),
            names=columns,
            skipinitialspace=True,
            na_values="?",
            comment=comment,
        )

    train = read("adult.data")
    test = read("adult.test", comment="|")

    train["income"] = train["income"].str.strip()
    test["income"] = test["income"].str.strip().str.rstrip(".")

    reference = pd.concat(
        [train[COLUMNS], test[COLUMNS]],
        ignore_index=True,
    )

    if reference.empty:
        raise ValueError("Reference dataset is empty.")

    logger.info("Loaded %d reference rows.", len(reference))
    return reference


def initialize_tracking(pool):
    with pool.connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS evidently_checked (
                id BIGINT PRIMARY KEY,
                checked_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)


def fetch_unchecked(pool):
    with pool.connection() as conn:
        return conn.execute("""
            SELECT p.id, p.input_data
            FROM predictions p
            LEFT JOIN evidently_checked c ON c.id = p.id
            WHERE c.id IS NULL
            ORDER BY p.id
        """).fetchall()


def extract_drift_flag(result):
    # Matches the Evidently output format observed in your logs.
    for metric in result.dict()["metrics"]:
        if metric["metric_name"].startswith("DriftedColumnsCount"):
            return metric["value"]["share"] >= DRIFT_SHARE_THRESHOLD

    raise ValueError("DriftedColumnsCount metric not found.")

# --------------------------------

def start_training():
    logger.info("Requesting training through HTTP.")
    try:
        requests.post(
            url=os.getenv("TRAINING_TRIGGER_URL", "http://training-trigger:8000/train"),
            timeout=3,
        )
    except requests.exceptions.ReadTimeout:
        pass  # expected — training takes longer than timeout
# ----------------
def mark_checked(pool, ids):
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO evidently_checked (id)
                VALUES (%s)
                ON CONFLICT (id) DO NOTHING
                """,
                [(pid,) for pid in ids],
            )


def start_training():
    trigger_url = os.getenv(
        "TRAINING_TRIGGER_URL",
        "http://training-trigger:8000/train",
    )
    token = os.getenv("TRAINING_TRIGGER_TOKEN")

    if not token:
        raise RuntimeError(
            "TRAINING_TRIGGER_TOKEN is missing from the environment."
        )

    logger.info("Requesting training through HTTP.")

    response = requests.post(
        trigger_url,
        headers={"Authorization": f"Bearer {token}"},
        timeout=10,
    )
    response.raise_for_status()

    logger.info("Training trigger accepted the request.")

def run_check(pool, reference, report_dir):
    rows = fetch_unchecked(pool)

    if len(rows) < BATCH_SIZE:
        logger.info(
            "%d/%d predictions collected.", len(rows), BATCH_SIZE
        )
        return

    ids = [row[0] for row in rows]
    current = pd.DataFrame([
        row[1] if isinstance(row[1], dict) else json.loads(row[1])
        for row in rows
    ])

    missing = set(COLUMNS) - set(current.columns)
    if missing:
        raise ValueError(f"Missing features: {sorted(missing)}")

    current = current[COLUMNS]

    result = Report([
        DataDriftPreset(drift_share=DRIFT_SHARE_THRESHOLD)
    ]).run(
        reference_data=reference,
        current_data=current,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = report_dir / f"drift_{timestamp}_{ids[-1]}.html"
    result.save_html(str(report_path))

    drift = extract_drift_flag(result)
    logger.info("Drift detected: %s", drift)
    logger.info("Report saved: %s", report_path)

    if drift:
        # If training fails, records remain unchecked and can be retried.
        start_training()

    mark_checked(pool, ids)


def main():
    report_dir = Path(os.getenv("REPORT_DIR", "reports"))
    report_dir.mkdir(parents=True, exist_ok=True)

    reference = load_reference()
    pool = build_pool()

    try:
        initialize_tracking(pool)
        logger.info(
            "Monitoring started; threshold=%d; interval=%ds.",
            BATCH_SIZE, CHECK_INTERVAL,
        )

        while True:
            try:
                run_check(pool, reference, report_dir)
            except Exception:
                logger.exception("Monitoring failed; will retry.")

            time.sleep(CHECK_INTERVAL)
    finally:
        pool.close()


if __name__ == "__main__":
    main()
