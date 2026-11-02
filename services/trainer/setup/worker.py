from services.trainer.setup.train import main
import boto3
import os
import requests

def upload_data_to_minio():
    s3 = boto3.client(
        "s3",
        endpoint_url=os.getenv("MINIO_ENDPOINT"),
        aws_access_key_id=os.getenv("MINIO_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("MINIO_SECRET_KEY"),
    )
    bucket = os.getenv("MINIO_BUCKET", "datasets")

    try:
        s3.create_bucket(Bucket=bucket)
    except s3.exceptions.BucketAlreadyOwnedByYou:
        pass

    for filename in ["adult.data", "adult.test"]:
        s3.upload_file(f"/data/{filename}", bucket, filename)

if __name__ == "__main__":
    upload_data_to_minio()
    main() ## the training script 
    response=requests.post(
    "http://api:8000/reload-model",
    timeout=30)
    response.raise_for_status()
