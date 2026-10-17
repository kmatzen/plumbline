import json

import numpy as np
import pytest
from paper.analyze_davis_image_evidence import analyze
from paper.build_reference_free_review import digest
from PIL import Image

from plumbline.image_evidence import rays_from_normalized_intrinsics


def test_pixel_center_conversion_matches_normalized_uv():
    k = np.array([[0.5, 0, 0.5], [0, 1, 0.5], [0, 0, 1]])
    rays = rays_from_normalized_intrinsics(np.array([[99.5, 49.5], [0, 0]]), k, (200, 100))
    assert np.allclose(rays, [[0, 0], [-0.995, -0.495]])


def test_empty_features_are_supported():
    assert rays_from_normalized_intrinsics(np.empty((0, 2)), np.eye(3), (10, 10)).shape == (0, 2)


@pytest.mark.parametrize("k", [np.zeros((3, 3)), np.full((3, 3), np.nan), np.eye(2)])
def test_invalid_calibration_rejected(k):
    with pytest.raises(ValueError):
        rays_from_normalized_intrinsics(np.array([[0, 0]]), k, (10, 10))


def test_no_matches_do_not_become_perfect_scores(tmp_path):
    root, run = tmp_path / "images", tmp_path / "run"
    root.mkdir()
    run.mkdir()
    frames = []
    for i in range(3):
        path = root / f"{i}.png"
        Image.new("RGB", (32, 32)).save(path)
        frames.append({"frame_index": i, "path": path.name, "file_sha256": digest(path)})
    poses = np.tile(np.eye(4), (3, 1, 1))
    poses[:, 0, 3] = [0, 1, 2]
    np.savez(
        run / "prediction.npz",
        extrinsics=poses,
        intrinsics_normalized=np.tile(np.eye(3), (3, 1, 1)),
    )
    manifest = {
        "status": "complete",
        "frames": frames,
        "prediction_file_sha256": digest(run / "prediction.npz"),
        "calibration": {"array": "intrinsics_normalized"},
    }
    (run / "manifest.json").write_text(json.dumps(manifest))
    output = tmp_path / "evidence.json"
    result = analyze(run, root, output)
    assert result["pairs"][0]["status"] == "no_matches"
    assert result["pairs"][0]["median_finite_residual"] is None
    assert result["pairs"][0]["count"] == 0
    with pytest.raises(ValueError, match="preserve"):
        analyze(run, root, output)
