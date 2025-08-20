from __future__ import annotations

from pathlib import Path

import pandas as pd
from pemw.features import compute_features
from pemw.model import evaluate_local, load_local_model, train_local


def _make_raw_df() -> pd.DataFrame:
    rows = []
    # Generate a small sequence of matches with varying results
    for i in range(10):
        rows.append(
            {
                "Date": f"01/08/202{i}",
                "HomeTeam": "Arsenal" if i % 2 == 0 else "Chelsea",
                "AwayTeam": "Chelsea" if i % 2 == 0 else "Arsenal",
                "FTHG": 1 + (i % 3),
                "FTAG": i % 2,
                "FTR": "H" if i % 3 == 0 else ("A" if i % 5 == 0 else "D"),
                # Provide a Season code (year start) to satisfy compute_features sorting
                "Season": f"202{i}",
            }
        )
    return pd.DataFrame(rows)


def test_train_and_evaluate(tmp_path: Path) -> None:
    raw = _make_raw_df()
    feats = compute_features(raw)
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    feats.to_parquet(processed_dir / "features.parquet", index=False)
    artifacts_dir = tmp_path / "artifacts"
    path = train_local(processed_dir, artifacts_dir)
    assert path.exists()
    model, meta = load_local_model(artifacts_dir)
    assert meta.numeric
    metrics = evaluate_local(processed_dir)
    assert 0.0 <= metrics["accuracy"] <= 1.0
