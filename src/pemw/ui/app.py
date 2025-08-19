from __future__ import annotations

import os
import streamlit as st
from pathlib import Path
from typing import Tuple
from pemw.model import load_local_model
from pemw.vertex import predict_online

BASE = Path(__file__).resolve().parents[3]
ARTIFACTS = BASE / "artifacts"
DATA_RAW = BASE / "data" / "raw"

st.set_page_config(page_title="Predict EPL Match Winner", page_icon="⚽")

st.title("Predict EPL Match Winner")
home = st.text_input("Home team", "Arsenal")
away = st.text_input("Away team", "Chelsea")

use_vertex = st.checkbox("Use Vertex endpoint (online prediction)", value=False)
endpoint = st.text_input("Vertex Endpoint Resource Name", "")

if st.button("Predict Winner of Match"):
    if use_vertex:
        if not endpoint:
            st.error("Please provide a Vertex endpoint name.")
        else:
            probs, label = predict_online(endpoint, home, away)
            st.success(f"Prediction: **{label}**")
            st.write({"Home": probs[0], "Draw": probs[1], "Away": probs[2]})
    else:
        model, meta = load_local_model(ARTIFACTS)
        from pemw.model import CATEG, NUMERIC
        import pandas as pd
        # Build a single-row input using neutral defaults
        row = {c: 0.0 for c in NUMERIC}
        row.update({"HomeTeam": home, "AwayTeam": away, "exp_home": 0.5, "exp_away": 0.5})
        import numpy as np
        X = pd.DataFrame([row])
        proba = model.predict_proba(X)[0]
        classes = meta.classes_
        order = [classes.index("H"), classes.index("D"), classes.index("A")]
        p = proba[order]
        label = ["Home","Draw","Away"][int(np.argmax(p))]
        st.success(f"Prediction: **{label}**")
        st.write({"Home": float(p[0]), "Draw": float(p[1]), "Away": float(p[2])})
