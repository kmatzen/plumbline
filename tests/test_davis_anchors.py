import copy

import numpy as np
from paper.build_anchor_preview import project
from paper.probe_davis_anchors import evaluate, grid_features, track_feature


def fixture():
    poses = np.tile(np.eye(4), (20, 1, 1))
    poses[:, 0, 3] = np.linspace(0, 1, 20)
    intrinsics = np.tile([[0.8, 0, 0.5], [0, 1, 0.5], [0, 0, 1]], (20, 1, 1))
    size = np.array([640, 480])
    point = np.array([0.2, 0.3, 4])
    track = [
        {"xy": project(point, p, k, size).tolist(), "annotation_label": 0, "crosscheck_pass": True}
        for p, k in zip(poses, intrinsics, strict=True)
    ]
    return poses, intrinsics, size, track


def test_later_observations_cannot_change_fitted_point():
    poses, k, size, track = fixture()
    original = evaluate(track, poses, k, size)
    changed = copy.deepcopy(track)
    changed[-1]["xy"][0] += 100
    result = evaluate(changed, poses, k, size)
    assert np.allclose(result["point_world"], original["point_world"])
    assert np.isclose(result["rows"][-1]["distance_pixels"], 100)
    assert np.isclose(original["rows"][-1]["distance_pixels"], 0)


def test_empty_slots_and_lost_tracks_are_not_zero_errors():
    blank = np.zeros((60, 80), np.uint8)
    slots = grid_features(blank, blank)
    assert len(slots) == 12 and all(p is None for p in slots)
    track = track_feature([blank] * 20, [blank] * 20, None)
    poses, k, size, _ = fixture()
    result = evaluate(track, poses, k, size)
    assert result["fit_status"] == "insufficient_early_background_track"
    assert all(row["distance_pixels"] is None for row in result["rows"])


def test_masked_or_lost_fit_observation_cannot_be_imputed():
    poses, k, size, track = fixture()
    track[5]["annotation_label"] = 1
    assert evaluate(track, poses, k, size)["point_world"] is None
    track[5]["annotation_label"] = 0
    track[-1]["xy"] = None
    result = evaluate(track, poses, k, size)
    assert result["point_world"] is not None
    assert result["rows"][-1]["status"] == "image_track_lost"
    assert result["rows"][-1]["distance_pixels"] is None
