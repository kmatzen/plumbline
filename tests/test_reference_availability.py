import json

import numpy as np
import pytest

from paper.audit_reference_availability import load_reference, window_availability
from paper.stage_static_rgbd_control import digest, nearest, parse_index


def test_missing_reference_is_not_replaced_or_interpolated(tmp_path):
    reference = tmp_path / "groundtruth.txt"
    reference.write_text("0 0 0 0 0 0 0 1\n1 0 0 0 0 0 0 1\n1 1 0 0 0 0 0 1\n2 0 0 0 0 0 0 0\n")
    index = parse_index(reference.read_bytes(), 8)
    manifest = {"indices": {"groundtruth.txt": digest(reference)}, "frames": [
        {"frame": i, "timestamp": t, "reference": nearest(t, index)}
        for i, t in enumerate((0, .5, 1, 2))]}
    source = tmp_path / "staging-manifest.json"
    source.write_text(json.dumps(manifest))
    poses, record = load_reference(tmp_path)
    np.testing.assert_array_equal(poses[0], np.eye(4))
    assert poses[1:] == [None, None, None]
    assert [r["status"] for r in record["rows"]] == ["associated", "outside_20ms", "ambiguous_timestamp", "invalid_reference_pose"]
    window = window_availability(poses, (0, 1), 2)
    assert window["unavailable_fit_frames"] == [1]
    assert window["query_reference_unavailable"]
    manifest["frames"][1]["reference"] = manifest["frames"][0]["reference"]
    source.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="association"):
        load_reference(tmp_path)


def test_missing_frame_zero_does_not_shift_fixed_window():
    poses = [None] + [np.eye(4) for _ in range(89)]
    assert window_availability(poses, range(1, 61), 89)["status"] == "available"
    poses[30] = None
    assert window_availability(poses, range(1, 61), 89)["unavailable_fit_frames"] == [30]
