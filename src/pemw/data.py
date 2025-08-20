from __future__ import annotations

import datetime
import warnings
from pathlib import Path
from typing import Final

import pandas as pd
import requests

BASE_URL: Final[str] = "https://www.football-data.co.uk/mmz4281/{season}/E0.csv"
OK_STATUS: Final[int] = 200
MIN_BYTES_THRESHOLD: Final[int] = 1000  # heuristic to filter empty / 404 pages


def _season_code(start_year: int) -> str:
    y = start_year % 100
    return f"{y:02d}{(y+1)%100:02d}"


def download_epl_data(dest_dir: Path, seasons: int = 20) -> int:
    # Try last `seasons` starting from current season back

    year = datetime.date.today().year - 1  # likely season start year
    count = 0
    for i in range(seasons):
        y = year - i
        url = BASE_URL.format(season=_season_code(y))
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == OK_STATUS and len(resp.content) > MIN_BYTES_THRESHOLD:
                out = dest_dir / f"E0_{_season_code(y)}.csv"
                out.write_bytes(resp.content)
                count += 1
        except Exception as exc:
            print(f"warn: {exc}")
    return count


def load_raw_csvs(raw_dir: Path) -> pd.DataFrame:
    files = sorted(raw_dir.glob("E0_*.csv"))
    if not files:
        raise FileNotFoundError("No raw CSVs found. Run download step first.")
    dfs: list[pd.DataFrame] = []
    for f in files:
        df = pd.read_csv(f)
        if "Date" in df.columns:
            # Parse common historical football-data date patterns without triggering
            # repeated pandas format inference warnings. Try several explicit formats
            # then fall back to a permissive parse.
            date_series = df["Date"].astype(str)
            parsed = pd.to_datetime(date_series, format="%d/%m/%Y", errors="coerce")
            mask = parsed.isna()
            if mask.any():
                parsed_alt = pd.to_datetime(date_series[mask], format="%d/%m/%y", errors="coerce")
                parsed.loc[mask] = parsed_alt
                mask = parsed.isna()
            if mask.any():
                parsed_alt2 = pd.to_datetime(date_series[mask], format="%Y-%m-%d", errors="coerce")
                parsed.loc[mask] = parsed_alt2
                mask = parsed.isna()
            if mask.any():
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="Could not infer format, so each element will be parsed individually",
                    )
                    parsed.loc[mask] = pd.to_datetime(
                        date_series[mask], errors="coerce", dayfirst=True
                    )
            df["Date"] = parsed
        # Normalize cols
        if "HomeTeam" not in df.columns and "HomeTeam" in [c.title() for c in df.columns]:
            # just in case casing differs across seasons
            df = df.rename(columns={c: c.title() for c in df.columns})
        df["Season"] = f.stem.split("_")[-1]
        dfs.append(df)
    full = pd.concat(dfs, ignore_index=True)
    cols = [
        c
        for c in [
            "Season",
            "Date",
            "HomeTeam",
            "AwayTeam",
            "FTHG",
            "FTAG",
            "FTR",
            "BbAvH",
            "BbAvD",
            "BbAvA",
            "AvgH",
            "AvgD",
            "AvgA",
        ]
        if c in full.columns
    ]
    return full[cols].dropna(subset=["HomeTeam", "AwayTeam", "FTR"], how="any")
