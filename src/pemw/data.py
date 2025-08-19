from __future__ import annotations
from pathlib import Path
import requests
import pandas as pd
from typing import List

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/E0.csv"

def _season_code(start_year: int) -> str:
    y = start_year % 100
    return f"{y:02d}{(y+1)%100:02d}"

def download_epl_data(dest_dir: Path, seasons: int = 20) -> int:
    # Try last `seasons` starting from current season back
    import datetime
    year = datetime.date.today().year - 1  # likely season start year
    count = 0
    for i in range(seasons):
        y = year - i
        url = BASE_URL.format(season=_season_code(y))
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200 and len(resp.content) > 1000:
                out = dest_dir / f"E0_{_season_code(y)}.csv"
                out.write_bytes(resp.content)
                count += 1
        except Exception as exc:  # noqa: BLE001
            print(f"warn: {exc}")
    return count

def load_raw_csvs(raw_dir: Path) -> pd.DataFrame:
    files = sorted(raw_dir.glob("E0_*.csv"))
    if not files:
        raise FileNotFoundError("No raw CSVs found. Run download step first.")
    dfs: List[pd.DataFrame] = []
    for f in files:
        df = pd.read_csv(f)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
        # Normalize cols
        if "HomeTeam" not in df.columns and "HomeTeam" in [c.title() for c in df.columns]:
            # just in case casing differs across seasons
            df = df.rename(columns={c: c.title() for c in df.columns})
        df["Season"] = f.stem.split("_")[-1]
        dfs.append(df)
    full = pd.concat(dfs, ignore_index=True)
    cols = [c for c in ["Season","Date","HomeTeam","AwayTeam","FTHG","FTAG","FTR",
                        "BbAvH","BbAvD","BbAvA","AvgH","AvgD","AvgA"] if c in full.columns]
    return full[cols].dropna(subset=["HomeTeam","AwayTeam","FTR"], how="any")
