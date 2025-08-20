from __future__ import annotations

import inspect
from pathlib import Path

import typer
from rich import print as rprint

from .data import download_epl_data
from .features import build_training_table
from .model import evaluate_local as _evaluate_local
from .model import select_and_train_best, tune_hgb, tune_logreg
from .model import train_local as _train_local

app = typer.Typer(add_completion=False, help="Predict EPL Match Winner CLI (local-only)")
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


@app.command(name="train-local")
def train_local_cmd(
    model_type: str = typer.Option("logreg", help="Model type: logreg|hgb"),
    min_team_freq: int = typer.Option(1, help="Prune teams with < freq to 'Other'"),
    calibrate: bool = typer.Option(False, help="Calibrate probabilities (isotonic, logreg only)"),
) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    sig = inspect.signature(_train_local)
    kwargs = {"model_type": model_type, "min_team_freq": min_team_freq, "calibrate": calibrate}
    if "model_type" in sig.parameters:
        path = _train_local(DATA_PROCESSED, ARTIFACTS, **kwargs)  # type: ignore[arg-type]
    else:  # compatibility with older signature / test monkeypatch
        path = _train_local(DATA_PROCESSED, ARTIFACTS)
    rprint(f"[green]Saved model to {path}[/green]")


@app.command(name="evaluate-local")
def evaluate_local_cmd(
    model_type: str = typer.Option("logreg", help="Model type: logreg|hgb"),
    min_team_freq: int = typer.Option(1, help="Prune teams with < freq to 'Other'"),
) -> None:
    sig = inspect.signature(_evaluate_local)
    kwargs = {"model_type": model_type, "min_team_freq": min_team_freq}
    if "model_type" in sig.parameters:
        metrics = _evaluate_local(DATA_PROCESSED, **kwargs)  # type: ignore[arg-type]
    else:
        metrics = _evaluate_local(DATA_PROCESSED)
    rprint("[bold cyan]Evaluation[/bold cyan]:", metrics)


@app.command(name="tune-logreg")
def tune_logreg_cmd() -> None:
    """Grid search over a small C list using time-series CV."""
    res = tune_logreg(DATA_PROCESSED)
    rprint("[bold cyan]Tuning[/bold cyan]:", res)


@app.command(name="tune-hgb")
def tune_hgb_cmd() -> None:
    """Grid search HistGradientBoosting hyperparameters."""
    res = tune_hgb(DATA_PROCESSED)
    rprint("[bold cyan]Tuning HGB[/bold cyan]:", res)


@app.command(name="auto-select")
def auto_select_cmd(
    min_team_freq: int = typer.Option(1, help="Prune teams with < freq to 'Other'"),
    calibrate: bool = typer.Option(False, help="Calibrate probabilities for logreg candidate"),
) -> None:
    """Compare logreg vs hgb (macro F1 primary) and persist best model."""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    res = select_and_train_best(
        DATA_PROCESSED,
        ARTIFACTS,
        min_team_freq=min_team_freq,
        calibrate=calibrate,
    )
    rprint("[bold cyan]Model Selection[/bold cyan]:", res)


# Cloud (GCS / Vertex) functionality permanently removed for local-only scope.
