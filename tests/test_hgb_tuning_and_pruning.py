from pathlib import Path

import pandas as pd
from pemw.features import compute_features
from pemw.model import evaluate_local, train_local, tune_hgb


def test_hgb_tune_and_prune(tmp_path: Path) -> None:
    teams = [f"Team{i}" for i in range(12)]
    rows = []
    for i in range(120):
        h = teams[i % len(teams)]
        a = teams[(i + 3) % len(teams)]
        rows.append(
            {
                "Date": f"2024-09-{(i%28)+1:02d}",
                "Season": "2425",
                "HomeTeam": h,
                "AwayTeam": a,
                "FTHG": (i * 7) % 5,
                "FTAG": (i * 11) % 5,
                "FTR": ["H", "D", "A"][i % 3],
                "BbAvH": 2.1 + (i % 5) * 0.1,
                "BbAvD": 3.1 + (i % 5) * 0.1,
                "BbAvA": 3.6 + (i % 5) * 0.1,
            }
        )
    raw = pd.DataFrame(rows)
    feats = compute_features(raw)
    proc = tmp_path / "processed"
    proc.mkdir()
    feats.to_parquet(proc / "features.parquet", index=False)
    # Run tuning with narrowed grids for speed
    res = tune_hgb(proc, learning_rates=[0.05], max_depths=[4], max_leaf_nodes_list=[16])
    assert "params" in res and "f1_macro" in res
    # Train with pruning
    artifacts = tmp_path / "artifacts"
    path = train_local(proc, artifacts, model_type="hgb", min_team_freq=3)
    assert path.exists()
    metrics = evaluate_local(proc, model_type="hgb", min_team_freq=3)
    assert 0.0 <= metrics["accuracy"] <= 1.0
