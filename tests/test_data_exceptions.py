from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from pemw import data as data_mod


class Boom(Exception):
    pass


def test_download_handles_exception(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = {"i": 0}
    line = b"Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n01/08/2020,A,B,1,0,H\n"
    content = line * 30  # ensure > MIN_BYTES_THRESHOLD (1000 bytes) in data module

    def fake_get(url: str, timeout: int = 0) -> object:
        calls["i"] += 1
        if calls["i"] == 1:
            raise Boom("net err")
        return SimpleNamespace(status_code=200, content=content)

    monkeypatch.setattr(data_mod, "requests", SimpleNamespace(get=fake_get))
    n = data_mod.download_epl_data(tmp_path, seasons=2)
    assert n == 1


def test_load_raw_csvs_no_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        data_mod.load_raw_csvs(tmp_path)
