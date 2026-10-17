import numpy as np
from paper.probe_point_support import track_support


def test_stationary_camera_can_support_point_without_identifying_depth():
    poses = np.tile(np.eye(4), (20, 1, 1))
    k = np.tile([[100, 0, 50], [0, 100, 50], [0, 0, 1]], (20, 1, 1))
    track = [{"xy": [50, 50]} for _ in poses]
    result = track_support(track, poses, k, 1)
    assert result["early"]["status"] == "positive_witness"
    assert result["all_observed"]["status"] == "positive_witness"


def test_later_inconsistency_does_not_change_early_support():
    poses = np.tile(np.eye(4), (20, 1, 1))
    k = np.tile([[100, 0, 50], [0, 100, 50], [0, 0, 1]], (20, 1, 1))
    track = [{"xy": [50, 50]} for _ in poses]
    track[-1] = {"xy": [60, 50]}
    result = track_support(track, poses, k, 1)
    assert result["early"]["status"] == "positive_witness"
    assert result["all_observed"]["status"] != "positive_witness"
    track[5] = {"xy": None}
    assert track_support(track, poses, k, 1)["status"] == "insufficient_early_observations"
