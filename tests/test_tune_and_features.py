from pathlib import Path

import pandas as pd
from pemw.features import compute_features
from pemw.model import evaluate_local, train_local, tune_logreg


def _sample_raw(n: int = 40) -> pd.DataFrame:
    rows = []
    teams = ["Arsenal", "Chelsea", "Liverpool", "Everton"]
    for i in range(n):
        h = teams[i % len(teams)]
        a = teams[(i + 1) % len(teams)]
        rows.append(
            {
                "Date": f"2023-08-{(i%28)+1:02d}",
                "Season": "2324",
                "HomeTeam": h,
                "AwayTeam": a,
                "FTHG": (i * 3) % 5,
                "FTAG": (i * 2) % 4,
                "FTR": ["H", "D", "A"][i % 3],
                "BbAvH": 2.0 + (i % 5) * 0.1,
                "BbAvD": 3.0 + (i % 5) * 0.1,
                "BbAvA": 3.5 + (i % 5) * 0.1,
            }
        )
    return pd.DataFrame(rows)


def test_compute_features_new_columns() -> None:
    raw = _sample_raw(15)
    feats = compute_features(raw)
    for col in [
        "home_form10",
        "away_form10",
        "home_gd10",
        "away_gd10",
        "imp_home",
        "imp_draw",
        "imp_away",
        "rest_home",
        "rest_away",
    ]:
        assert col in feats.columns
    # interaction
    assert "elo_prob_gap" in feats.columns


def test_tune_and_evaluate(tmp_path: Path) -> None:
    raw = _sample_raw(50)
    feats = compute_features(raw)
    proc = tmp_path / "processed"
    proc.mkdir()
    feats.to_parquet(proc / "features.parquet", index=False)
    artifacts = tmp_path / "artifacts"
    # train to ensure integration still works
    train_local(proc, artifacts)
    metrics = evaluate_local(proc)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    res = tune_logreg(proc, Cs=[0.5, 1.0])
    assert set(res.keys()) == {"f1_macro", "accuracy", "C"}
