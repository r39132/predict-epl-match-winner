from pathlib import Path

import pandas as pd

from pemw.features import compute_features
from pemw.model import select_and_train_best


def _make_minimal_raw(n: int = 20) -> pd.DataFrame:
    teams = ["Arsenal", "Chelsea", "Liverpool", "Everton"]
    rows = []
    for i in range(n):
        rows.append(
            {
                "Date": f"2024-08-{(i%28)+1:02d}",
                "Season": "2425",
                "HomeTeam": teams[i % len(teams)],
                "AwayTeam": teams[(i + 1) % len(teams)],
                "FTHG": (i * 3) % 4,
                "FTAG": (i * 5) % 4,
                "FTR": ["H", "D", "A"][i % 3],
            }
        )
    return pd.DataFrame(rows)


def test_select_and_train_best(tmp_path: Path) -> None:
    # Build minimal features file in an isolated processed dir
    raw = _make_minimal_raw()
    feats = compute_features(raw)
    processed = tmp_path / "processed"
    processed.mkdir()
    feats.to_parquet(processed / "features.parquet", index=False)
    art = tmp_path / "art_sel"
    res = select_and_train_best(processed, art, min_team_freq=1)
    # Basic shape
    assert "chosen_model" in res
    assert "chosen_metrics" in res
    # Model file exists
    assert (art / "model.joblib").exists()
    if (art / "model_selection.json").exists():
        txt = (art / "model_selection.json").read_text()
        assert "chosen_model" in txt
