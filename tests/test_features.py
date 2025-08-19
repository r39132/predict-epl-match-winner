from pathlib import Path
import pandas as pd
from pemw.features import compute_features

def test_compute_features_runs(tmp_path: Path) -> None:
    df = pd.read_csv(Path(__file__).parent / "data_samples" / "tiny_E0.csv")
    feats = compute_features(df)
    assert {"home_elo","away_elo","elo_diff","target"}.issubset(set(feats.columns))
    assert len(feats) == len(df.dropna(subset=["FTR"]))
