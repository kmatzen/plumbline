import hashlib
import json

import numpy as np
import pytest
from paper.analyze_dense_run import load_verified


def fixture(tmp_path):
    poses = np.tile(np.eye(4, dtype=np.float32), (3, 1, 1))
    np.savez_compressed(tmp_path / "reference.npz", extrinsics=poses)
    np.savez_compressed(tmp_path / "prediction.npz", extrinsics=poses)
    manifest = {
        "status": "complete",
        "frames": [{"timestamp": str(i)} for i in range(3)],
        "prediction_file_sha256": hashlib.sha256(
            (tmp_path / "prediction.npz").read_bytes()
        ).hexdigest(),
        "reference_sha256": hashlib.sha256(poses.tobytes()).hexdigest(),
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_verification_and_tampered_prediction(tmp_path):
    fixture(tmp_path)
    _, prediction, reference = load_verified(tmp_path)
    np.testing.assert_array_equal(prediction, reference)
    (tmp_path / "prediction.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="prediction hash"):
        load_verified(tmp_path)


@pytest.mark.parametrize(
    "change,match",
    [
        ({"status": "running"}, "not complete"),
        ({"reference_sha256": "wrong"}, "reference hash"),
        ({"frames": [{"timestamp": "1"}] * 3}, "strictly increasing"),
        ({"frames": []}, "frame count"),
    ],
)
def test_reject_inconsistent_manifest(tmp_path, change, match):
    manifest = fixture(tmp_path)
    manifest.update(change)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=match):
        load_verified(tmp_path)
