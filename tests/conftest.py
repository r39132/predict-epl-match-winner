from __future__ import annotations

import pathlib

_IGNORE = {"test_cli_upload.py"}


def pytest_ignore_collect(path):  # type: ignore[no-untyped-def]
    try:
        name = pathlib.Path(str(path)).name
    except Exception:  # pragma: no cover
        return False
    return name in _IGNORE
