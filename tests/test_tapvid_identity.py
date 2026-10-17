import copy

import numpy as np

from paper.probe_tapvid_identity import correlation, decisions, patch, predict, score


def test_patch_boundary_and_textureless_abstention():
    gray = np.zeros((32, 32), dtype=np.uint8)
    assert patch(gray, [1, 1]) is None
    assert correlation(patch(gray, [16, 16]), patch(gray, [16, 16])) is None
    rng = np.random.default_rng(0)
    a = rng.normal(size=(17, 17))
    assert np.isclose(correlation(a, a*2+3), 1)


def test_scoring_cannot_change_image_decision_and_counts_occlusion():
    rows = [{"frame": 1, "xy": [100, 50], "forward_backward_pixels": .1, "source_patch_ncc": .9},
            {"frame": 2, "xy": None, "forward_backward_pixels": None, "source_patch_ncc": None}]
    before = [decisions(r) for r in rows]
    points = np.tile([.5, .5], (3, 1))
    a = score(copy.deepcopy(rows), points, np.zeros(3, bool), 200, 100)
    b = score(rows, points, np.ones(3, bool), 200, 100)
    assert before == [decisions(r) for r in rows]
    assert a["both"]["accepted_correct"] == 1
    assert a["both"]["rejected_visible"] == 1
    assert b["both"]["accepted_occluded"] == 1
    assert a["both"]["planned"] == 2


def test_identity_video_produces_consistent_predictions_without_annotations():
    gray = np.random.default_rng(7).integers(0, 256, (64, 64), dtype=np.uint8)
    result = predict([gray]*4, 1, [32, 32])
    assert [r["frame"] for r in result] == [2, 3]
    for row in result:
        np.testing.assert_allclose(row["xy"], [32, 32], atol=.01)
        assert decisions(row)["both"]
