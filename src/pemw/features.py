from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import pandas as pd

HOME_ADVANTAGE_ELO = 60.0
K_FACTOR = 24.0

@dataclass
class EloState:
    ratings: Dict[str, float]

def _expected(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10 ** (-(r_a - r_b) / 400.0))

def _result_to_scores(r: str) -> Tuple[float, float]:
    if r == "H": return 1.0, 0.0
    if r == "A": return 0.0, 1.0
    return 0.5, 0.5

def compute_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.sort_values(["Date","Season"]).reset_index(drop=True)
    teams = pd.unique(pd.concat([df["HomeTeam"], df["AwayTeam"]])).tolist()
    state = EloState(ratings={t: 1500.0 for t in teams})
    last5_pts = {t: [] for t in teams}
    last5_gd = {t: [] for t in teams}

    rows = []
    for _, row in df.iterrows():
        h, a = row["HomeTeam"], row["AwayTeam"]
        rh = state.ratings.get(h,1500.0); ra = state.ratings.get(a,1500.0)
        exp_h = _expected(rh + HOME_ADVANTAGE_ELO, ra)
        exp_a = 1.0 - exp_h

        feat = {
            "Date": row.get("Date"),
            "Season": row.get("Season"),
            "HomeTeam": h, "AwayTeam": a,
            "home_elo": rh, "away_elo": ra, "elo_diff": rh - ra,
            "exp_home": exp_h, "exp_away": exp_a,
            "home_form5": float(np.mean(last5_pts[h][-5:])) if last5_pts[h] else 1.0,
            "away_form5": float(np.mean(last5_pts[a][-5:])) if last5_pts[a] else 1.0,
            "home_gd5": float(np.mean(last5_gd[h][-5:])) if last5_gd[h] else 0.0,
            "away_gd5": float(np.mean(last5_gd[a][-5:])) if last5_gd[a] else 0.0,
            "target": row.get("FTR"),
        }
        for col in ("BbAvH","BbAvD","BbAvA","AvgH","AvgD","AvgA"):
            if col in df.columns:
                feat[col] = row.get(col)
        rows.append(feat)

        # update post-match
        ftr = str(row.get("FTR"))
        s_h, s_a = _result_to_scores(ftr)
        if pd.notna(row.get("FTHG")) and pd.notna(row.get("FTAG")):
            gd = float(row.get("FTHG")) - float(row.get("FTAG"))
        else:
            gd = 1.0 if ftr == "H" else -1.0 if ftr == "A" else 0.0
        last5_gd[h].append(gd); last5_gd[a].append(-gd)
        if ftr == "H":
            last5_pts[h].append(3.0); last5_pts[a].append(0.0)
        elif ftr == "A":
            last5_pts[h].append(0.0); last5_pts[a].append(3.0)
        else:
            last5_pts[h].append(1.0); last5_pts[a].append(1.0)

        e_h = _expected(rh + HOME_ADVANTAGE_ELO, ra)
        e_a = 1.0 - e_h
        state.ratings[h] = rh + K_FACTOR * (s_h - e_h)
        state.ratings[a] = ra + K_FACTOR * (s_a - e_a)

    feats = pd.DataFrame(rows).dropna(subset=["target"])
    return feats

def build_training_table(raw_dir: Path, out_dir: Path) -> Path:
    from .data import load_raw_csvs
    raw = load_raw_csvs(raw_dir)
    feats = compute_features(raw)
    out = out_dir / "features.parquet"
    feats.to_parquet(out, index=False)
    return out
