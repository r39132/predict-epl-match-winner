from __future__ import annotations
from pathlib import Path
from google.cloud import storage

def upload_dir_to_gcs(local_dir: Path, bucket_uri: str) -> None:
    if not bucket_uri.startswith("gs://"):
        raise ValueError("bucket must be gs://bucket[/prefix]")
    path = local_dir
    client = storage.Client()
    bucket_name = bucket_uri.replace("gs://","").split("/")[0]
    prefix = "/".join(bucket_uri.replace("gs://","").split("/")[1:])
    bucket = client.bucket(bucket_name)
    for p in path.rglob("*"):
        if p.is_file():
            rel = p.relative_to(local_dir).as_posix()
            blob = bucket.blob(f"{prefix}/{rel}" if prefix else rel)
            blob.upload_from_filename(str(p))

def upload_file_to_gcs(local_file: Path, bucket_uri: str, dest_path: str) -> None:
    if not bucket_uri.startswith("gs://"):
        raise ValueError("bucket must be gs://bucket[/prefix]")
    client = storage.Client()
    bucket_name = bucket_uri.replace("gs://","").split("/")[0]
    prefix = "/".join(bucket_uri.replace("gs://","").split("/")[1:]).rstrip("/")
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"{prefix}/{dest_path}" if prefix else dest_path)
    blob.upload_from_filename(str(local_file))
