import hashlib
import json
import zipfile

import numpy as np
import pytest
from paper.analyze_davis_regions import labels_at
from paper.stage_davis_masks import stage


def test_sampling_preserves_void_and_outside():
    mask = np.array([[0, 1], [2, 255]], dtype=np.uint8)
    points = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [-1, 0], [2, 0]])
    assert labels_at(mask, points).tolist() == [0, 1, 2, 255, 255, 255]
    assert labels_at(mask, np.empty((0, 2))).shape == (0,)


def test_nonfinite_coordinates_rejected():
    with pytest.raises(ValueError):
        labels_at(np.zeros((2, 2)), np.array([[np.nan, 0]]))


def fixture(tmp_path, status="complete", split="train"):
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("DAVIS/ImageSets/2017/train.txt", "dev\n")
        z.writestr("DAVIS/JPEGImages/480p/dev/00000.jpg", b"image")
        z.writestr("DAVIS/Annotations/480p/dev/00000.png", b"mask")
    manifest = {
        "status": status,
        "split": split,
        "sequence": "dev",
        "frames": [
            {
                "frame_index": 0,
                "path": "DAVIS/JPEGImages/480p/dev/00000.jpg",
                "file_sha256": hashlib.sha256(b"image").hexdigest(),
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return archive, path


def test_annotation_archive_link_and_no_overwrite(tmp_path):
    archive, manifest = fixture(tmp_path)
    output = tmp_path / "masks"
    stage(archive, manifest, output)
    assert (output / "00000.png").read_bytes() == b"mask"
    assert "not static-motion truth" in json.loads((output / "masks.json").read_text())["scope"]
    with pytest.raises(ValueError, match="preserve"):
        stage(archive, manifest, output)


@pytest.mark.parametrize("status,split", [("running", "train"), ("complete", "val")])
def test_only_completed_training_runs(tmp_path, status, split):
    archive, manifest = fixture(tmp_path, status, split)
    with pytest.raises(ValueError):
        stage(archive, manifest, tmp_path / "masks")
