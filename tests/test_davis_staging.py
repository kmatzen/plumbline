import zipfile

import pytest
from paper.stage_davis import stage


def archive(tmp_path, *, val="test", frame="00000.jpg"):
    path = tmp_path / "data.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("DAVIS/ImageSets/2017/train.txt", "dev\n")
        z.writestr("DAVIS/ImageSets/2017/val.txt", val + "\n")
        z.writestr("DAVIS/JPEGImages/480p/dev/" + frame, b"test-frame")
        z.writestr("DAVIS/JPEGImages/480p/test/00000.jpg", b"test-frame")
        z.writestr("DAVIS/Annotations/480p/dev/00000.png", b"must-not-extract")
    return path


def test_only_frames_and_splits_are_staged(tmp_path):
    output = tmp_path / "out"
    result = stage(archive(tmp_path), output)
    assert not (output / "DAVIS/Annotations").exists()
    assert result["camera_ground_truth"] is None
    assert result["capture_timestamps"] is None
    assert result["sequences"][0]["frames"][0]["frame_index"] == 0
    assert len(result["archive_sha256"]) == 64
    with pytest.raises(ValueError, match="already exists"):
        stage(tmp_path / "data.zip", output)


def test_overlap_rejected_before_extraction(tmp_path):
    with pytest.raises(ValueError, match="overlap"):
        stage(archive(tmp_path, val="dev"), tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_traversal_rejected_before_extraction(tmp_path):
    with pytest.raises(ValueError, match="unexpected frame"):
        stage(archive(tmp_path, frame="../escape.jpg"), tmp_path / "out")
    assert not (tmp_path / "out").exists()
