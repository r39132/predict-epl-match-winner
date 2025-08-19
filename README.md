# Predict EPL Match Winner

An end-to-end ML project that downloads ~20 years of EPL results, builds features,
trains and evaluates a model, publishes to **Vertex AI**, and serves predictions via a **Streamlit** UI.

## Tech
- Python 3.12.3, `uv`
- Vertex AI (train, register, deploy, predict)
- Streamlit UI for inference
- Tooling: black, isort, ruff, mypy, pytest + coverage, pre-commit & pre-push

## Quick start
```bash
# 0) env
uv venv
uv pip install -e ".[dev]"
pre-commit install
pre-commit install --hook-type pre-push

# 1) data (last 20 seasons by default)
pemw download-data --seasons 20

# 2) features + local training
pemw prepare-data
pemw train-local
pemw evaluate-local

# 3) (optional) upload to GCS
export GCS_BUCKET=gs://your-bucket
pemw upload-data-gcs --bucket $GCS_BUCKET
pemw upload-model-gcs --bucket $GCS_BUCKET

# 4) Vertex AI setup (env)
export GOOGLE_CLOUD_PROJECT=your-project
export GOOGLE_CLOUD_REGION=us-central1
export VERTEX_STAGING_BUCKET=gs://your-staging-bucket
# Optional: custom prebuilt image for sklearn (Google provided)
# export VERTEX_SERVING_IMAGE_URI="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-5:latest"

# 5) Vertex operations
pemw vertex register-model
pemw vertex deploy-endpoint
pemw vertex predict --endpoint $VERTEX_ENDPOINT --home "Arsenal" --away "Chelsea"

# 6) Streamlit UI (local; uses Vertex if USE_VERTEX=true)
streamlit run src/pemw/ui/app.py
```
