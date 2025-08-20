from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HOME_ADVANTAGE_ELO = 60.0
K_FACTOR = 24.0


@dataclass
class EloState:
    ratings: dict[str, float]


def _expected(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10 ** (-(r_a - r_b) / 400.0))


def _result_to_scores(r: str) -> tuple[float, float]:
    if r == "H":
        return 1.0, 0.0
    if r == "A":
        return 0.0, 1.0
    return 0.5, 0.5


def compute_features(
    raw: pd.DataFrame,
) -> pd.DataFrame:  # (single-pass computation for performance/readability)
    """Compute match-level features.

    Newly added vs initial version:
    - Implied probabilities from odds (BbAv*) normalized -> imp_home/draw/away
    - Rolling last10 form & goal diff (home_form10, away_form10, home_gd10, away_gd10)
    - Rest days since previous match for each team (rest_home, rest_away)
    - Interaction between elo_diff and (imp_home - imp_away) (elo_prob_gap)
    """
    # Ensure datetime for rest-day calculations
    # date dtype normalization (support nullable dtypes)
    if not pd.api.types.is_datetime64_any_dtype(raw["Date"]):
        with pd.option_context("mode.chained_assignment", None):
            raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    df = raw.sort_values(["Date", "Season"]).reset_index(drop=True)
    teams = pd.unique(pd.concat([df["HomeTeam"], df["AwayTeam"]])).tolist()
    state = EloState(ratings={t: 1500.0 for t in teams})
    last5_pts: dict[str, list[float]] = {t: [] for t in teams}
    last5_gd: dict[str, list[float]] = {t: [] for t in teams}
    last10_pts: dict[str, list[float]] = {t: [] for t in teams}
    last10_gd: dict[str, list[float]] = {t: [] for t in teams}
    last_played: dict[str, pd.Timestamp] = {}

    rows = []
    for _, row in df.iterrows():
        h, a = row["HomeTeam"], row["AwayTeam"]
        rh = state.ratings.get(h, 1500.0)
        ra = state.ratings.get(a, 1500.0)
        exp_h = _expected(rh + HOME_ADVANTAGE_ELO, ra)
        exp_a = 1.0 - exp_h

        # Odds -> implied probabilities (may be NaN)
        bb_h, bb_d, bb_a = row.get("BbAvH"), row.get("BbAvD"), row.get("BbAvA")
        imp_h = imp_d = imp_a = np.nan
        if all(col in df.columns for col in ("BbAvH", "BbAvD", "BbAvA")):
            if not (pd.isna(bb_h) or pd.isna(bb_d) or pd.isna(bb_a)):
                raw_probs = np.array([1.0 / bb_h, 1.0 / bb_d, 1.0 / bb_a], dtype=float)
                s = raw_probs.sum()
                if s > 0:
                    imp_h, imp_d, imp_a = (raw_probs / s).tolist()

        feat = {
            "Date": row.get("Date"),
            "Season": row.get("Season"),
            "HomeTeam": h,
            "AwayTeam": a,
            "home_elo": rh,
            "away_elo": ra,
            "elo_diff": rh - ra,
            "exp_home": exp_h,
            "exp_away": exp_a,
            "home_form5": float(np.mean(last5_pts[h][-5:])) if last5_pts[h] else 1.0,
            "away_form5": float(np.mean(last5_pts[a][-5:])) if last5_pts[a] else 1.0,
            "home_gd5": float(np.mean(last5_gd[h][-5:])) if last5_gd[h] else 0.0,
            "away_gd5": float(np.mean(last5_gd[a][-5:])) if last5_gd[a] else 0.0,
            "home_form10": float(np.mean(last10_pts[h][-10:])) if last10_pts[h] else 1.0,
            "away_form10": float(np.mean(last10_pts[a][-10:])) if last10_pts[a] else 1.0,
            "home_gd10": float(np.mean(last10_gd[h][-10:])) if last10_gd[h] else 0.0,
            "away_gd10": float(np.mean(last10_gd[a][-10:])) if last10_gd[a] else 0.0,
            "imp_home": imp_h,
            "imp_draw": imp_d,
            "imp_away": imp_a,
            "target": row.get("FTR"),
        }
        # Rest days
        date_val = row.get("Date")
        rest_h = rest_a = np.nan
        if isinstance(date_val, pd.Timestamp):
            if h in last_played:
                rest_h = (date_val - last_played[h]).days
            if a in last_played:
                rest_a = (date_val - last_played[a]).days
            last_played[h] = date_val
            last_played[a] = date_val
        feat["rest_home"] = rest_h
        feat["rest_away"] = rest_a
        for col in ("BbAvH", "BbAvD", "BbAvA", "AvgH", "AvgD", "AvgA"):
            if col in df.columns:
                feat[col] = row.get(col)
        rows.append(feat)

        # update post-match
        ftr = str(row.get("FTR"))
        s_h, s_a = _result_to_scores(ftr)
        fthg_raw = row.get("FTHG")
        ftag_raw = row.get("FTAG")
        if (
            fthg_raw is not None
            and ftag_raw is not None
            and pd.notna(fthg_raw)
            and pd.notna(ftag_raw)
        ):
            # Safe to convert
            fthg = float(fthg_raw)
            ftag = float(ftag_raw)
            gd = fthg - ftag
        else:
            gd = 1.0 if ftr == "H" else -1.0 if ftr == "A" else 0.0
        last5_gd[h].append(gd)
        last5_gd[a].append(-gd)
        last10_gd[h].append(gd)
        last10_gd[a].append(-gd)
        if ftr == "H":
            last5_pts[h].append(3.0)
            last5_pts[a].append(0.0)
            last10_pts[h].append(3.0)
            last10_pts[a].append(0.0)
        elif ftr == "A":
            last5_pts[h].append(0.0)
            last5_pts[a].append(3.0)
            last10_pts[h].append(0.0)
            last10_pts[a].append(3.0)
        else:
            last5_pts[h].append(1.0)
            last5_pts[a].append(1.0)
            last10_pts[h].append(1.0)
            last10_pts[a].append(1.0)

        e_h = _expected(rh + HOME_ADVANTAGE_ELO, ra)
        e_a = 1.0 - e_h
        state.ratings[h] = rh + K_FACTOR * (s_h - e_h)
        state.ratings[a] = ra + K_FACTOR * (s_a - e_a)

    feats = pd.DataFrame(rows).dropna(subset=["target"])
    # Interaction after all rows assembled
    if {"elo_diff", "imp_home", "imp_away"}.issubset(feats.columns):
        feats["elo_prob_gap"] = feats["elo_diff"] * (
            feats["imp_home"].fillna(0.0) - feats["imp_away"].fillna(0.0)
        )
    # Drop odds columns that are entirely NaN across the assembled dataset to
    # prevent downstream UserWarnings (e.g., some historical seasons lack Avg* odds).
    _odds_cols = ["BbAvH", "BbAvD", "BbAvA", "AvgH", "AvgD", "AvgA"]
    to_drop = [c for c in _odds_cols if c in feats.columns and feats[c].isna().all()]
    if to_drop:
        feats = feats.drop(columns=to_drop)
    return feats


def build_training_table(raw_dir: Path, out_dir: Path) -> Path:
    from .data import load_raw_csvs

    raw = load_raw_csvs(raw_dir)
    feats = compute_features(raw)
    out = out_dir / "features.parquet"
    feats.to_parquet(out, index=False)
    return out
