import io
import json
import os

import boto3
import pandas as pd
import psycopg
from dotenv import load_dotenv

load_dotenv()

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income",
]


def _s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("MINIO_ENDPOINT"),
        aws_access_key_id=os.getenv("MINIO_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("MINIO_SECRET_KEY"),
    )


def _load_from_minio() -> tuple[pd.DataFrame, pd.DataFrame]:
    s3 = _s3_client()
    bucket = os.getenv("MINIO_BUCKET", "datasets")

    def read(key, comment=None):
        obj = s3.get_object(Bucket=bucket, Key=key)
        return pd.read_csv(
            io.BytesIO(obj["Body"].read()),
            names=COLUMNS,
            skipinitialspace=True,
            na_values="?",
            comment=comment,
        )

    train = read("adult.data")
    test = read("adult.test", comment="|")

    train["income"] = train["income"].str.strip()
    test["income"] = test["income"].str.strip().str.rstrip(".")

    return train, test


def _load_from_postgres() -> pd.DataFrame | None:
    conn_str = (
        f"host={os.getenv('POSTGRES_HOST', 'localhost')} "
        f"port={os.getenv('POSTGRES_PORT', '5432')} "
        f"dbname={os.getenv('POSTGRES_DB')} "
        f"user={os.getenv('POSTGRES_USER')} "
        f"password={os.getenv('POSTGRES_PASSWORD')}"
    )

    with psycopg.connect(conn_str) as conn:
        rows = conn.execute("""
            SELECT input_data, prediction
            FROM predictions
            WHERE prediction IS NOT NULL
        """).fetchall()

    if not rows:
        return None

    records = []
    for input_data, prediction in rows:
        row = input_data if isinstance(input_data, dict) else json.loads(input_data)
        row["income"] = prediction
        records.append(row)

    df = pd.DataFrame(records)

    missing = set(COLUMNS) - set(df.columns)
    if missing:
        return None

    return df[COLUMNS]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = _load_from_minio()

    user_data = _load_from_postgres()
    if user_data is not None and not user_data.empty:
        train = pd.concat([train, user_data], ignore_index=True)

    return train, test
