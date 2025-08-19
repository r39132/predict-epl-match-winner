from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, List
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["home_elo","away_elo","elo_diff","exp_home","exp_away",
           "home_form5","away_form5","home_gd5","away_gd5",
           "BbAvH","BbAvD","BbAvA","AvgH","AvgD","AvgA"]
CATEG = ["HomeTeam","AwayTeam"]

@dataclass
class ModelMeta:
    numeric: List[str]
    categ: List[str]
    classes_: List[str]

def _load_feats(processed_dir: Path) -> pd.DataFrame:
    p = processed_dir / "features.parquet"
    if not p.exists():
        raise FileNotFoundError("features.parquet not found. Run prepare-data.")
    df = pd.read_parquet(p)
    return df.dropna(subset=["target"])

def _pipe(numeric: list[str], categ: list[str]) -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(with_mean=False), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categ),
    ])
    clf = LogisticRegression(max_iter=300, multi_class="multinomial")
    return Pipeline([("pre", pre), ("clf", clf)])

def train_local(processed_dir: Path, artifacts_dir: Path) -> Path:
    feats = _load_feats(processed_dir)
    numeric = [c for c in NUMERIC if c in feats.columns]
    categ = [c for c in CATEG if c in feats.columns]
    X = feats[numeric + categ]
    y = feats["target"].astype("category")
    pipe = _pipe(numeric, categ)
    pipe.fit(X, y)
    meta = ModelMeta(numeric, categ, list(pipe.named_steps["clf"].classes_))
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out = artifacts_dir / "model.joblib"
    joblib.dump({"model": pipe, "meta": meta}, out)
    return out

def evaluate_local(processed_dir: Path) -> Dict[str, float]:
    feats = _load_feats(processed_dir)
    numeric = [c for c in NUMERIC if c in feats.columns]
    categ = [c for c in CATEG if c in feats.columns]
    X = feats[numeric + categ]; y = feats["target"].astype("category")
    tscv = TimeSeriesSplit(n_splits=5)
    y_true, y_pred = [], []
    for tr, te in tscv.split(X):
        model = _pipe(numeric, categ)
        model.fit(X.iloc[tr], y.iloc[tr])
        pred = model.predict(X.iloc[te])
        y_true.extend(y.iloc[te].tolist()); y_pred.extend(pred.tolist())
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    return {"accuracy": float(acc), "f1_macro": float(f1m)}

def load_local_model(artifacts_dir: Path):
    obj = joblib.load(artifacts_dir / "model.joblib")
    return obj["model"], obj["meta"]
