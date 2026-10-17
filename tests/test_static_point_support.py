import json

import numpy as np
import pytest

from paper.probe_static_point_support import load_prediction, support
from paper.probe_static_rgbd_control import reference_cases
from paper.stage_static_rgbd_control import digest


def test_support_uses_common_fit_indices_without_missing_reference_frame():
    poses, track = [], []
    k = np.array([[500., 0, 320], [0, 500., 240], [0, 0, 1.]])
    for frame in range(90):
        pose = np.eye(4)
        pose[0, 3] = frame * .01
        poses.append(pose if frame else None)
        track.append({"xy": [320 - 500 * frame * .01 / 3, 240]})
    cases = reference_cases(poses)
    reference = support(track, cases["reference"], k, 4)
    rotated = support(track, cases["reference_later_yaw_5deg"], k, 4)
    assert reference["fit_original_frames"] == list(range(1, 11))
    assert reference["observed_original_frames"] == list(range(1, 90))
    assert reference["later_observed_frames"] == 79
    assert reference["all_observed"]["status"] == "positive_witness"
    assert reference["early"] == rotated["early"]
    assert rotated["all_observed"]["status"] == "solver_infeasible"
    track[20]["xy"] = None
    missing = support(track, cases["reference"], k, 4)
    assert missing["missing_track_frames"] == [20]
    assert missing["later_observed_frames"] == 78
    track[5]["xy"] = None
    assert support(track, cases["reference"], k, 4)["status"] == "insufficient_early_observations"
    with pytest.raises(ValueError, match="positive"):
        support(track, cases["reference"], k, 0)


def test_prediction_provenance_and_rigid_pose_checks(tmp_path):
    runner = tmp_path / "executed-runner.py"
    runner.write_text("# synthetic runner fixture\n")
    prediction = tmp_path / "prediction.npz"
    np.savez(prediction, extrinsics=np.repeat(np.eye(4)[None], 90, axis=0))
    record = {"status": "complete", "source_manifest_sha256": "source", "frames": ["synthetic"],
              "input_sha256": "input", "runner_sha256": digest(runner),
              "prediction_file_sha256": digest(prediction)}
    (tmp_path / "manifest.json").write_text(json.dumps(record))
    poses, _ = load_prediction(tmp_path, "source", ["synthetic"], "input")
    assert poses.shape == (90, 4, 4)
    with pytest.raises(ValueError, match="inputs"):
        load_prediction(tmp_path, "source", ["different"], "input")
    prediction.write_bytes(b"changed")
    with pytest.raises(ValueError, match="bytes"):
        load_prediction(tmp_path, "source", ["synthetic"], "input")
    poses[3, 0, 0] = 2
    np.savez(prediction, extrinsics=poses)
    record["prediction_file_sha256"] = digest(prediction)
    (tmp_path / "manifest.json").write_text(json.dumps(record))
    with pytest.raises(AssertionError, match="orthonormal"):
        load_prediction(tmp_path, "source", ["synthetic"], "input")
