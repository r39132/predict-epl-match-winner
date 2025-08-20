from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from pemw import cli


def test_cli_upload_commands(monkeypatch, tmp_path: Path):  # type: ignore[no-untyped-def]
    runner = CliRunner()
    cli.DATA_RAW = tmp_path / "data" / "raw"
    cli.ARTIFACTS = tmp_path / "artifacts"
    cli.DATA_RAW.mkdir(parents=True, exist_ok=True)
    cli.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (cli.ARTIFACTS / "model.joblib").write_bytes(b"bin")

    called = {"dir": False, "file": False}

    def fake_upload_dir(local, bucket):  # type: ignore[no-untyped-def]
        called["dir"] = True

    def fake_upload_file(local, bucket, dest):  # type: ignore[no-untyped-def]
        called["file"] = True

    monkeypatch.setattr(cli, "upload_dir_to_gcs", fake_upload_dir)
    monkeypatch.setattr(cli, "upload_file_to_gcs", fake_upload_file)

    r1 = runner.invoke(cli.app, ["upload-data-gcs", "--bucket", "gs://b"])
    r2 = runner.invoke(cli.app, ["upload-model-gcs", "--bucket", "gs://b"])
    assert r1.exit_code == 0 and r2.exit_code == 0
    assert called["dir"] and called["file"]
