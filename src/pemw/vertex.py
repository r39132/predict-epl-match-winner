from __future__ import annotations
import os
from typing import Tuple
import numpy as np
from google.cloud import aiplatform as vertex

# Expected env:
# GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_REGION, VERTEX_STAGING_BUCKET
# Optional: VERTEX_SERVING_IMAGE_URI

def _init():
    project = os.environ["GOOGLE_CLOUD_PROJECT"]
    region = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
    vertex.init(project=project, location=region, staging_bucket=os.environ["VERTEX_STAGING_BUCKET"])

def register_model(artifacts_dir):
    _init()
    model_path = os.path.join(str(artifacts_dir), "model.joblib")
    serving_image = os.environ.get("VERTEX_SERVING_IMAGE_URI", "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-5:latest")
    model = vertex.Model.upload(
        display_name="pemw-sklearn-model",
        artifact_uri=str(artifacts_dir),
        serving_container_image_uri=serving_image,
    )
    model.wait()
    return {"model_resource_name": model.resource_name}

def deploy_endpoint():
    _init()
    models = list(vertex.Model.list(filter='display_name="pemw-sklearn-model"'))
    if not models:
        raise RuntimeError("Model not found. Run register-model first.")
    model = models[0]
    endpoint = vertex.Endpoint.create(display_name="pemw-endpoint")
    endpoint.wait()
    endpoint.deploy(model=model, traffic_percentage=100)
    return endpoint.resource_name

def predict_online(endpoint_name: str, home: str, away: str):
    _init()
    endpoint = vertex.Endpoint(endpoint_name=endpoint_name)
    # Minimal example: model expects "HomeTeam" and "AwayTeam" categoricals; set others to defaults.
    instance = {
        "HomeTeam": home, "AwayTeam": away,
        "home_elo": 1500.0, "away_elo": 1500.0, "elo_diff": 0.0,
        "exp_home": 0.5, "exp_away": 0.5,
        "home_form5": 1.0, "away_form5": 1.0,
        "home_gd5": 0.0, "away_gd5": 0.0,
    }
    resp = endpoint.predict(instances=[instance])
    preds = np.array(resp.predictions[0])  # [H,D,A] if trained that way
    labels = ["Home","Draw","Away"]
    return preds.tolist(), labels[int(np.argmax(preds))]
