from __future__ import annotations

import pandas as pd

from pemw.features import compute_features


def test_compute_features_missing_scores():  # type: ignore[no-untyped-def]
    df = pd.DataFrame(
        [
            {"Date": "01/08/2020", "Season": "2020", "HomeTeam": "A", "AwayTeam": "B", "FTR": "H"},
            {"Date": "08/08/2020", "Season": "2020", "HomeTeam": "B", "AwayTeam": "A", "FTR": "D"},
        ]
    )
    feats = compute_features(df)
    EXPECT_ROWS = 2
    assert len(feats) == EXPECT_ROWS
    assert "home_gd5" in feats.columns and feats["home_gd5"].notna().all()
