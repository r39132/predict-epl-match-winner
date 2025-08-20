from pathlib import Path

import pandas as pd

from pemw.features import compute_features
from pemw.model import evaluate_local, train_local


def test_hgb_training_and_eval(tmp_path: Path) -> None:
    # create small synthetic raw data
    rows = []
    for i in range(30):
        rows.append(
            {
                "Date": f"2024-08-{(i%28)+1:02d}",
                "Season": "2425",
                "HomeTeam": "Arsenal" if i % 2 == 0 else "Chelsea",
                "AwayTeam": "Chelsea" if i % 2 == 0 else "Arsenal",
                "FTHG": (i * 3) % 4,
                "FTAG": (i * 5) % 4,
                "FTR": ["H", "D", "A"][i % 3],
                "BbAvH": 2.0 + (i % 5) * 0.1,
                "BbAvD": 3.0 + (i % 5) * 0.1,
                "BbAvA": 3.5 + (i % 5) * 0.1,
            }
        )
    raw = pd.DataFrame(rows)
    feats = compute_features(raw)
    proc = tmp_path / "processed"
    proc.mkdir()
    feats.to_parquet(proc / "features.parquet", index=False)
    artifacts = tmp_path / "artifacts"
    path = train_local(proc, artifacts, model_type="hgb")
    assert path.exists()
    metrics = evaluate_local(proc, model_type="hgb")
    assert 0.0 <= metrics["accuracy"] <= 1.0
