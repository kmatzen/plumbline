import json

import numpy as np
import pytest
from paper.build_reference_free_review import build, digest, load_verified
from PIL import Image


def fixture(tmp_path):
    run = tmp_path / "run"
    root = tmp_path / "images"
    run.mkdir()
    root.mkdir()
    frames = []
    for i in range(3):
        path = root / f"{i}.png"
        Image.new("RGB", (24, 24), (i * 50, 0, 0)).save(path)
        frames.append({"frame_index": i * 2, "path": path.name, "file_sha256": digest(path)})
    poses = np.tile(np.eye(4), (3, 1, 1))
    np.savez(run / "prediction.npz", extrinsics=poses)
    manifest = {
        "status": "complete",
        "frames": frames,
        "prediction_file_sha256": digest(run / "prediction.npz"),
    }
    (run / "manifest.json").write_text(json.dumps(manifest))
    return run, root


def test_stationary_camera_renders_without_alignment(tmp_path):
    run, root = fixture(tmp_path)
    output = tmp_path / "review"
    result = build(run, root, output)
    assert result["functional_verdict"] == "unreviewed"
    assert result["source_frames_reverified"] == 3
    assert result["selected_frame_indices"] == [0, 2, 4]
    assert result["review_sha256"] == digest(output / "review.png")
    with pytest.raises(ValueError, match="preserve"):
        build(run, root, output)


def test_source_tamper_rejected(tmp_path):
    run, root = fixture(tmp_path)
    (root / "0.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="source frame"):
        load_verified(run, root)


def test_prediction_tamper_rejected(tmp_path):
    run, root = fixture(tmp_path)
    (run / "prediction.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="prediction hash"):
        load_verified(run, root)


@pytest.mark.parametrize("change", ["pending", "order", "escape"])
def test_invalid_manifest_rejected(tmp_path, change):
    run, root = fixture(tmp_path)
    path = run / "manifest.json"
    manifest = json.loads(path.read_text())
    if change == "pending":
        manifest["status"] = "running"
    elif change == "order":
        manifest["frames"].reverse()
    else:
        manifest["frames"][0]["path"] = "../outside.png"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_verified(run, root)
