from __future__ import annotations

from pathlib import Path

import pytest
from pemw.model import evaluate_local


def test_evaluate_local_missing_features(tmp_path: Path):  # type: ignore[no-untyped-def]
    with pytest.raises(FileNotFoundError):
        evaluate_local(tmp_path)
