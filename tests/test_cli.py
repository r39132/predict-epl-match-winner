from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from pemw import cli


def test_cli_prepare_and_train(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    runner = CliRunner()
    # Patch paths inside module to point to tmp dirs
    cli.DATA_RAW = tmp_path / "data" / "raw"
    cli.DATA_PROCESSED = tmp_path / "data" / "processed"
    cli.ARTIFACTS = tmp_path / "artifacts"

    # Fake feature building & training to isolate CLI logic
    called = {"prepare": False, "train": False}

    def fake_build(raw_dir, out_dir):  # type: ignore[no-untyped-def]
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "features.parquet").write_text("fake")
        called["prepare"] = True
        return out_dir / "features.parquet"

    def fake_train(processed_dir, artifacts_dir):  # type: ignore[no-untyped-def]
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        p = artifacts_dir / "model.joblib"
        p.write_bytes(b"binary")
        called["train"] = True
        return p

    monkeypatch.setattr(cli, "build_training_table", fake_build)
    monkeypatch.setattr(cli, "_train_local", fake_train)

    result_prep = runner.invoke(cli.app, ["prepare-data"])
    assert result_prep.exit_code == 0
    result_train = runner.invoke(cli.app, ["train-local"])
    assert result_train.exit_code == 0
    assert called["prepare"] and called["train"]
