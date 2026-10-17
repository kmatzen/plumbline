import pytest

from paper.run_static_da3 import run


def test_existing_attempt_is_preserved_before_loading_weights(tmp_path):
    output = tmp_path / "attempt"
    output.mkdir()
    with pytest.raises(ValueError, match="preserve"):
        run(tmp_path, tmp_path, output)


def test_missing_checkpoint_rejected_before_creating_attempt(tmp_path):
    output = tmp_path / "attempt"
    with pytest.raises(ValueError, match="snapshot"):
        run(tmp_path, tmp_path / "missing", output)
    assert not output.exists()
