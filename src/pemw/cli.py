from __future__ import annotations

import os
from pathlib import Path
import typer
from rich import print as rprint

from .data import download_epl_data
from .features import build_training_table
from .model import train_local, evaluate_local, save_model_artifacts
from .gcs_utils import upload_dir_to_gcs, upload_file_to_gcs
from .vertex import register_model, deploy_endpoint, predict_online

app = typer.Typer(add_completion=False, help="Predict EPL Match Winner CLI")
BASE = Path(__file__).resolve().parents[2]
DATA_RAW = BASE / "data" / "raw"
DATA_PROCESSED = BASE / "data" / "processed"
ARTIFACTS = BASE / "artifacts"


@app.command()
def download_data(seasons: int = typer.Option(20, help="How many seasons back (approx).")) -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    n = download_epl_data(DATA_RAW, seasons=seasons)
    rprint(f"[green]Downloaded {n} file(s) into {DATA_RAW}[/green]")


@app.command()
def prepare_data() -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out = build_training_table(DATA_RAW, DATA_PROCESSED)
    rprint(f"[green]Wrote features to {out}[/green]")


@app.command()
def train_local() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = train_local(DATA_PROCESSED, ARTIFACTS)
    rprint(f"[green]Saved model to {path}[/green]")


@app.command()
def evaluate_local() -> None:
    metrics = evaluate_local(DATA_PROCESSED)
    rprint("[bold cyan]Evaluation[/bold cyan]:", metrics)


@app.command()
def upload_data_gcs(bucket: str = typer.Option(..., "--bucket")) -> None:
    upload_dir_to_gcs(DATA_RAW, bucket)
    rprint("[green]Uploaded data to GCS[/green]")


@app.command()
def upload_model_gcs(bucket: str = typer.Option(..., "--bucket")) -> None:
    upload_file_to_gcs(ARTIFACTS / "model.joblib", bucket, "models/model.joblib")
    rprint("[green]Uploaded model to GCS[/green]")


@app.command()
def vertex(ctx: typer.Context):
    """Group for Vertex commands."""


@vertex.command("register-model")
def vertex_register_model() -> None:
    model_res = register_model(artifacts_dir=ARTIFACTS)
    rprint(model_res)


@vertex.command("deploy-endpoint")
def vertex_deploy_endpoint() -> None:
    endpoint_name = deploy_endpoint()
    rprint({"endpoint": endpoint_name})


@vertex.command("predict")
def vertex_predict(endpoint: str = typer.Option(..., "--endpoint"),
                   home: str = typer.Option(..., "--home"),
                   away: str = typer.Option(..., "--away")) -> None:
    probs, label = predict_online(endpoint, home, away)
    rprint({"Home": probs[0], "Draw": probs[1], "Away": probs[2], "prediction": label})
