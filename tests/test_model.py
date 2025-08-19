from pathlib import Path
import pandas as pd
from pemw.features import compute_features
from pemw.model import _pipe

def test_model_fits(tmp_path: Path) -> None:
    df = pd.read_csv(Path(__file__).parent / "data_samples" / "tiny_E0.csv")
    feats = compute_features(df)
    num_cols = [c for c in ["home_elo","away_elo","elo_diff","home_form5","away_form5","home_gd5","away_gd5"] if c in feats.columns]
    cat_cols = ["HomeTeam","AwayTeam"]
    X = feats[num_cols + cat_cols]
    y = feats["target"].astype("category")
    pipe = _pipe(num_cols, cat_cols)
    pipe.fit(X, y)
    assert set(pipe.named_steps["clf"].classes_) == {"H","D","A"}
