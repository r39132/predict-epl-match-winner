from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from pemw import data as data_mod


class DummyResp:
    def __init__(self, status_code: int = 200, content: bytes = b"csvdata") -> None:
        self.status_code = status_code
        self.content = content


def test_download_epl_data(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Provide pseudo CSV content large enough to pass threshold
    content = ("Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n" + "01/08/2020,Arsenal,Chelsea,1,0,H\n") * 20

    def fake_get(url: str, timeout: int = 0) -> DummyResp:
        return DummyResp(200, content.encode())

    monkeypatch.setattr(data_mod, "requests", SimpleNamespace(get=fake_get))
    n = data_mod.download_epl_data(tmp_path, seasons=1)
    assert n == 1
    files = list(tmp_path.glob("E0_*.csv"))
    assert files, "CSV not written"


def test_load_raw_csvs_and_date_parsing(tmp_path: Path) -> None:
    # Create two CSVs with different date formats to exercise parsing logic
    csv1 = "Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n01/08/2020,Arsenal,Chelsea,1,0,H\n"
    csv2 = "Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n2020-08-08,Spurs,Liverpool,2,2,D\n"
    (tmp_path / "E0_2021.csv").write_text(csv1)
    (tmp_path / "E0_2020.csv").write_text(csv2)
    df = data_mod.load_raw_csvs(tmp_path)
    assert {"Season", "Date", "HomeTeam", "AwayTeam", "FTR"}.issubset(df.columns)
    # All rows parsed; dates coerced
    EXPECT_ROWS = 2
    assert len(df) == EXPECT_ROWS
    assert pd.api.types.is_datetime64_any_dtype(df["Date"])
