from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

NUMERIC = [
    "home_elo",
    "away_elo",
    "elo_diff",
    "exp_home",
    "exp_away",
    "home_form5",
    "away_form5",
    "home_gd5",
    "away_gd5",
    "home_form10",
    "away_form10",
    "home_gd10",
    "away_gd10",
    "imp_home",
    "imp_draw",
    "imp_away",
    "rest_home",
    "rest_away",
    "elo_prob_gap",
    "BbAvH",
    "BbAvD",
    "BbAvA",
    "AvgH",
    "AvgD",
    "AvgA",
]
CATEG = ["HomeTeam", "AwayTeam"]


@dataclass
class ModelMeta:
    numeric: list[str]
    categ: list[str]
    classes_: list[str]
    model_type: str = "logreg"


def _load_feats(processed_dir: Path) -> pd.DataFrame:
    p = processed_dir / "features.parquet"
    if not p.exists():
        raise FileNotFoundError("features.parquet not found. Run prepare-data.")
    df = pd.read_parquet(p)
    return df.dropna(subset=["target"])


def _pipe(  # noqa: PLR0913 (argument count acceptable for clarity)
    numeric: list[str],
    categ: list[str],
    *,
    C: float = 1.0,
    class_weight: str | None = None,
    model_type: str = "logreg",
    hgb_params: dict[str, Any] | None = None,
) -> Pipeline:
    """Build preprocessing + model pipeline.

    model_type options:
      - logreg: LogisticRegression with scaling + one-hot
      - hgb: HistGradientBoostingClassifier with ordinal-encoded categoricals
    """
    if model_type == "logreg":
        num_pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler(with_mean=False)),
            ]
        )
        cat_pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        pre = ColumnTransformer(
            [
                ("num", num_pipe, numeric),
                ("cat", cat_pipe, categ),
            ]
        )
        clf = LogisticRegression(max_iter=1000, C=C, class_weight=class_weight)
        return Pipeline([("pre", pre), ("clf", clf)])
    elif model_type == "hgb":
        # Ordinal encode categoricals; specify categorical feature indices to HGB
        num_pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
            ]
        )
        cat_pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                (
                    "ord",
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                ),
            ]
        )
        pre = ColumnTransformer(
            [
                ("num", num_pipe, numeric),
                ("cat", cat_pipe, categ),
            ]
        )
        default_params: dict[str, Any] = {
            "max_depth": 6,
            "learning_rate": 0.05,
            "max_iter": 400,
            "random_state": 42,
            "max_leaf_nodes": 31,
        }
        if hgb_params:
            default_params.update(hgb_params)
        clf = HistGradientBoostingClassifier(**default_params)
        return Pipeline([("pre", pre), ("clf", clf)])
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def _prune_rare_categories(feats: pd.DataFrame, columns: Iterable[str], min_freq: int) -> None:
    for col in columns:
        vc = feats[col].value_counts()
        rare = set(vc[vc < min_freq].index)
        if rare:
            feats.loc[feats[col].isin(rare), col] = "Other"


def train_local(  # noqa: PLR0913 (argument count acceptable)
    processed_dir: Path,
    artifacts_dir: Path,
    *,
    model_type: str = "logreg",
    min_team_freq: int = 1,
    calibrate: bool = False,
    hgb_params: dict[str, Any] | None = None,
) -> Path:
    feats = _load_feats(processed_dir)
    # Drop numeric columns that are entirely NaN
    numeric = [c for c in NUMERIC if c in feats.columns and feats[c].notna().any()]
    categ = [c for c in CATEG if c in feats.columns]
    if min_team_freq > 1:
        _prune_rare_categories(feats, categ, min_team_freq)
    X = feats[numeric + categ]
    y = feats["target"].astype("category")
    pipe = _pipe(
        numeric,
        categ,
        class_weight="balanced" if model_type == "logreg" else None,
        model_type=model_type,
        hgb_params=hgb_params,
    )
    estimator: Any
    if calibrate:
        # Wrap full pipeline for probability calibration
        estimator = CalibratedClassifierCV(pipe, cv=3, method="isotonic")
        estimator.fit(X, y)
        classes_ = list(estimator.classes_)
    else:
        pipe.fit(X, y)
        estimator = pipe
        classes_ = list(pipe.named_steps["clf"].classes_)
    meta = ModelMeta(numeric, categ, classes_, model_type=model_type)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out = artifacts_dir / "model.joblib"
    joblib.dump({"model": estimator, "meta": meta}, out)
    return out


def evaluate_local(
    processed_dir: Path,
    *,
    model_type: str = "logreg",
    min_team_freq: int = 1,
    hgb_params: dict[str, Any] | None = None,
) -> dict[str, float]:
    feats = _load_feats(processed_dir)
    numeric = [c for c in NUMERIC if c in feats.columns and feats[c].notna().any()]
    categ = [c for c in CATEG if c in feats.columns]
    if min_team_freq > 1:
        _prune_rare_categories(feats, categ, min_team_freq)
    X = feats[numeric + categ]
    y = feats["target"].astype("category")
    tscv = TimeSeriesSplit(n_splits=5)
    y_true: list[str] = []
    y_pred: list[str] = []
    for tr, te in tscv.split(X):
        model = _pipe(
            numeric,
            categ,
            class_weight="balanced" if model_type == "logreg" else None,
            model_type=model_type,
            hgb_params=hgb_params,
        )
        model.fit(X.iloc[tr], y.iloc[tr])
        fold_pred = model.predict(X.iloc[te])
        y_true.extend(y.iloc[te].tolist())
        y_pred.extend(fold_pred.tolist())
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    return {"accuracy": float(acc), "f1_macro": float(f1m)}


def load_local_model(artifacts_dir: Path) -> tuple[Pipeline, ModelMeta]:
    obj = joblib.load(artifacts_dir / "model.joblib")
    return obj["model"], obj["meta"]


class LogRegResult(TypedDict):
    f1_macro: float
    accuracy: float
    C: float


def tune_logreg(processed_dir: Path, Cs: list[float] | None = None) -> LogRegResult:
    """Simple time-series aware C parameter search.

    Returns metrics for best C (macro F1 primary, accuracy tie-breaker).
    """
    if Cs is None:
        Cs = [0.25, 0.5, 1.0, 2.0, 4.0]
    feats = _load_feats(processed_dir)
    numeric = [c for c in NUMERIC if c in feats.columns and feats[c].notna().any()]
    categ = [c for c in CATEG if c in feats.columns]
    X = feats[numeric + categ]
    y = feats["target"].astype("category")
    tscv = TimeSeriesSplit(n_splits=5)
    best_f1: float = -1.0
    best_acc: float = -1.0
    best_C: float = Cs[0] if Cs else 1.0
    for C in Cs:
        y_true: list[str] = []
        y_pred: list[str] = []
        for tr, te in tscv.split(X):
            m = _pipe(numeric, categ, C=C, class_weight="balanced", model_type="logreg")
            m.fit(X.iloc[tr], y.iloc[tr])
            pred = m.predict(X.iloc[te])
            y_true.extend(y.iloc[te].tolist())
            y_pred.extend(pred.tolist())
        acc = accuracy_score(y_true, y_pred)
        f1m = f1_score(y_true, y_pred, average="macro")
    if f1m > best_f1 or (f1m == best_f1 and acc > best_acc):
        best_f1 = float(f1m)
        best_acc = float(acc)
        best_C = C
    return LogRegResult(f1_macro=best_f1, accuracy=best_acc, C=best_C)


def tune_hgb(
    processed_dir: Path,
    *,
    learning_rates: list[float] | None = None,
    max_depths: list[int] | None = None,
    max_leaf_nodes_list: list[int] | None = None,
    min_team_freq: int = 1,
) -> dict[str, Any]:
    """Grid search over a few HistGradientBoostingClassifier hyperparameters.

    Returns best params + metrics (macro F1 primary, accuracy tie-breaker).
    """
    if learning_rates is None:
        learning_rates = [0.02, 0.05, 0.1]
    if max_depths is None:
        max_depths = [4, 6, 8]
    if max_leaf_nodes_list is None:
        max_leaf_nodes_list = [16, 31, 63]
    feats = _load_feats(processed_dir)
    numeric = [c for c in NUMERIC if c in feats.columns and feats[c].notna().any()]
    categ = [c for c in CATEG if c in feats.columns]
    if min_team_freq > 1:
        _prune_rare_categories(feats, categ, min_team_freq)
    X = feats[numeric + categ]
    y = feats["target"].astype("category")
    tscv = TimeSeriesSplit(n_splits=5)
    best: dict[str, Any] = {"f1_macro": -1.0, "accuracy": -1.0, "params": {}}
    for lr in learning_rates:
        for md in max_depths:
            for mln in max_leaf_nodes_list:
                y_true: list[str] = []
                y_pred: list[str] = []
                params = {"learning_rate": lr, "max_depth": md, "max_leaf_nodes": mln}
                for tr, te in tscv.split(X):
                    m = _pipe(numeric, categ, model_type="hgb", hgb_params=params)
                    m.fit(X.iloc[tr], y.iloc[tr])
                    pred = m.predict(X.iloc[te])
                    y_true.extend(y.iloc[te].tolist())
                    y_pred.extend(pred.tolist())
                acc = accuracy_score(y_true, y_pred)
                f1m = f1_score(y_true, y_pred, average="macro")
                if f1m > best["f1_macro"] or (f1m == best["f1_macro"] and acc > best["accuracy"]):
                    best.update({"f1_macro": float(f1m), "accuracy": float(acc), "params": params})
    return best


def select_and_train_best(
    processed_dir: Path,
    artifacts_dir: Path,
    *,
    min_team_freq: int = 1,
    calibrate: bool = False,
    hgb_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate logistic regression vs HGB and train + persist the better model.

    Selection criterion: macro F1 (primary) then accuracy tie-breaker.

    Returns a dictionary with chosen model type, its metrics, and candidate metrics.
    """
    logreg_metrics = evaluate_local(
        processed_dir,
        model_type="logreg",
        min_team_freq=min_team_freq,
    )
    hgb_metrics = evaluate_local(
        processed_dir,
        model_type="hgb",
        min_team_freq=min_team_freq,
        hgb_params=hgb_params,
    )

    # Decide
    def _score(m: dict[str, float]) -> tuple[float, float]:
        return (m["f1_macro"], m["accuracy"])

    if _score(hgb_metrics) > _score(logreg_metrics):
        chosen = "hgb"
        chosen_metrics = hgb_metrics
    else:
        chosen = "logreg"
        chosen_metrics = logreg_metrics
    # Train final chosen model on full data
    train_local(
        processed_dir,
        artifacts_dir,
        model_type=chosen,
        min_team_freq=min_team_freq,
        calibrate=calibrate if chosen == "logreg" else False,
        hgb_params=hgb_params if chosen == "hgb" else None,
    )
    result: dict[str, Any] = {
        "chosen_model": chosen,
        "chosen_metrics": chosen_metrics,
        "logreg_metrics": logreg_metrics,
        "hgb_metrics": hgb_metrics,
        "hgb_params": hgb_params or {},
        "calibrated": bool(calibrate if chosen == "logreg" else False),
    }
    # Persist selection metadata alongside model
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    meta_path = artifacts_dir / "model_selection.json"
    try:
        import json

        with meta_path.open("w") as f:
            json.dump(result, f, indent=2)
    except Exception:  # pragma: no cover - best effort
        pass
    return result
