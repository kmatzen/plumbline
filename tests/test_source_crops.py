import json

import cv2
import numpy as np
import pytest

from paper.build_source_crops import build, centered_crop
from paper.stage_rgbd_validation import sha


def test_crop_retains_native_center_and_marks_padding():
    image = np.zeros((10, 10, 3), np.uint8)
    image[1, 2] = [10, 20, 30]
    crop, pixel = centered_crop(image, [2.2, .8], radius=3)
    assert crop.shape == (7, 7, 3)
    assert pixel == [2, 1]
    np.testing.assert_array_equal(crop[3, 3], image[1, 2])
    np.testing.assert_array_equal(crop[0, 0], [128, 128, 128])
    np.testing.assert_array_equal(crop[2, 3], image[0, 2])
    with pytest.raises(ValueError, match="outside"):
        centered_crop(image, [-5, 2])


def test_build_includes_every_initial_target_without_later_selection(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    image = source / "frame.png"
    cv2.imwrite(str(image), np.zeros((100, 100, 3), np.uint8))
    run = tmp_path / "run"
    run.mkdir()
    manifest = run / "manifest.json"
    manifest.write_text(json.dumps({"frames": [{"path": image.name, "file_sha256": sha(image)}]}))
    tracks = tmp_path / "tracks.json"
    tracks.write_text(json.dumps({"source_manifest_sha256": sha(manifest), "tracks": [
        [{"xy": [50, 50]}, {"xy": None}],
        [{"xy": None}, {"xy": [50, 50]}],
        [{"xy": [1, 1]}, {"xy": [99, 99]}],
    ]}))
    output = tmp_path / "crops"
    build(run, source, tracks, output)
    items = json.loads((output / "manifest.json").read_text())["items"]
    assert [x["status"] for x in items] == ["ready", "no_initial_target", "ready"]
    assert [x["slot"] for x in items] == [0, 1, 2]
    assert items[0]["image_sha256"] == sha(output / "slot-0.png")
    with pytest.raises(ValueError, match="preserve"):
        build(run, source, tracks, output)
    image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="source image"):
        build(run, source, tracks, tmp_path / "other")
