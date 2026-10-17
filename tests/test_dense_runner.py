import hashlib

import pytest
from paper.run_dense_dage import digest_file, run


def test_streaming_hash_covers_entire_file(tmp_path):
    data = b"a" * (1024 * 1024 + 1) + b"b"
    path = tmp_path / "frames"
    path.write_bytes(data)
    assert digest_file(path) == hashlib.sha256(data).hexdigest()


def test_existing_attempt_is_not_overwritten(tmp_path):
    sentinel = tmp_path / "manifest.json"
    sentinel.write_text("existing attempt")
    with pytest.raises(ValueError, match="preserve previous"):
        run(tmp_path, tmp_path, "scene", tmp_path)
    assert sentinel.read_text() == "existing attempt"


def test_checkpoint_requires_pinned_file(tmp_path):
    with pytest.raises(ValueError, match=r"contain model\.pt"):
        run(tmp_path, tmp_path, "scene", tmp_path / "new")
    assert not (tmp_path / "new").exists()
