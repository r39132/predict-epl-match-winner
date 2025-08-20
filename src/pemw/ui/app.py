from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from pemw.data import load_raw_csvs
from pemw.model import NUMERIC, load_local_model

BASE = Path(__file__).resolve().parents[3]
ARTIFACTS = BASE / "artifacts"
DATA_RAW = BASE / "data" / "raw"

st.set_page_config(page_title="Predict EPL Match Winner", page_icon="⚽")

st.title("Predict EPL Match Winner")


@st.cache_data(show_spinner=False)
def _load_current_season(raw_dir: Path) -> tuple[pd.DataFrame, list[str], str]:
    try:
        raw = load_raw_csvs(raw_dir)
    except FileNotFoundError:
        return pd.DataFrame(), [], ""
    if "Season" not in raw.columns:
        return pd.DataFrame(), [], ""
    current_season = max(raw["Season"].astype(str).unique())
    cur = raw[raw["Season"].astype(str) == current_season].copy()
    teams = sorted(pd.unique(pd.concat([cur["HomeTeam"], cur["AwayTeam"]])).tolist())
    return cur, teams, current_season


raw_current, teams, season_code = _load_current_season(DATA_RAW)
if not teams:
    st.error(
        "No raw season data found. Run 'uv run pemw download-data' and 'uv run pemw prepare-data' first."
    )
    st.stop()

st.caption("Season 2025/2026")
home = st.selectbox("Home team", teams, index=0)
away_choices = [t for t in teams if t != home]
if not away_choices:
    st.warning("Need at least two distinct teams in data.")
    st.stop()
away = st.selectbox("Away team", away_choices, index=0)

# Safety: streamlit returns a value from the provided list when list non-empty
assert isinstance(home, str)
assert isinstance(away, str)


def last5_summary(team: str, df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"Team": team}
    subset = df[(df.HomeTeam == team) | (df.AwayTeam == team)].sort_values("Date")
    last5 = subset.tail(5)
    w = d = losses = gf = ga = pts = 0
    form: list[str] = []
    for _, r in last5.iterrows():
        home_team = r["HomeTeam"]
        ftr = str(r.get("FTR"))
        fthg = r.get("FTHG", 0)
        ftag = r.get("FTAG", 0)
        if home_team == team:
            _gf = fthg
            _ga = ftag
            res_map = {"H": "W", "D": "D", "A": "L"}
        else:
            _gf = ftag
            _ga = fthg
            res_map = {"H": "L", "D": "D", "A": "W"}
        res = res_map.get(ftr, "D")
        if res == "W":
            w += 1
            pts += 3
        elif res == "D":
            d += 1
            pts += 1
        else:
            losses += 1
        gf += int(_gf) if pd.notna(_gf) else 0
        ga += int(_ga) if pd.notna(_ga) else 0
        form.append(res)
    played = len(last5)
    return {
        "Team": team,
        "P": played,
        "W": w,
        "D": d,
        "L": losses,
        "GF": gf,
        "GA": ga,
        "GD": gf - ga,
        "Pts": pts,
        "Form": "".join(form),
    }


summ_df = pd.DataFrame([last5_summary(home, raw_current), last5_summary(away, raw_current)])
st.subheader("Recent Form (last 5 matches this season)")
st.dataframe(summ_df, use_container_width=True)

if st.button("Predict Winner of Match"):
    model, meta = load_local_model(ARTIFACTS)
    row: dict[str, Any] = {c: 0.0 for c in NUMERIC}
    row["HomeTeam"] = home
    row["AwayTeam"] = away
    row["exp_home"] = 0.5  # placeholder expectation features (could be improved)
    row["exp_away"] = 0.5
    X = pd.DataFrame([row])
    proba = model.predict_proba(X)[0]
    classes = meta.classes_
    order = [classes.index("H"), classes.index("D"), classes.index("A")]
    p = proba[order]
    label = ["Home", "Draw", "Away"][int(np.argmax(p))]
    outcome_team = home if label == "Home" else away if label == "Away" else "Draw"
    st.success(f"Predicted result: {label} ({outcome_team})")
    probs_df = pd.DataFrame(
        {
            "Outcome": ["Home win", "Draw", "Away win"],
            "Team": [home, "-", away],
            "Probability": [float(p[0]), float(p[1]), float(p[2])],
        }
    ).sort_values("Probability", ascending=False)
    probs_df["Probability %"] = (probs_df["Probability"] * 100).map(lambda x: f"{x:0.1f}%")
    st.dataframe(probs_df[["Outcome", "Team", "Probability %"]], use_container_width=True)
    st.caption(
        "Probabilities reflect model output (macro-F1 oriented selection). They are not betting odds and may be miscalibrated if model was trained without calibration."
    )
