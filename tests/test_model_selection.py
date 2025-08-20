from pathlib import Path

from pemw.model import select_and_train_best


def test_select_and_train_best(tmp_path: Path) -> None:
    processed = Path("data/processed")
    assert (processed / "features.parquet").exists(), "prepare-data must run before tests"
    art = tmp_path / "art_sel"
    res = select_and_train_best(processed, art, min_team_freq=1)
    # Basic shape
    assert "chosen_model" in res
    assert "chosen_metrics" in res
    # Model file exists
    assert (art / "model.joblib").exists()
    # Metadata file optional but if present should be json-like
    if (art / "model_selection.json").exists():
        txt = (art / "model_selection.json").read_text()
        assert "chosen_model" in txt
