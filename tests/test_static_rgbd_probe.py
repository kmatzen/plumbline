import numpy as np
import pytest

from paper.probe_static_rgbd_control import evaluate_window, reference_cases


def test_common_window_keeps_original_indices_and_rotation_starts_after_fit():
    poses = []
    track = []
    k = np.array([[500 / 640, 0, 320.5 / 640], [0, 500 / 480, 240.5 / 480], [0, 0, 1.]])
    for i in range(90):
        p = np.eye(4)
        p[0, 3] = i * .01
        poses.append(p if i else None)
        track.append({"xy": [320 - 500 * i * .01 / 3, 240], "annotation_label": 0})
    cases = reference_cases(poses)
    np.testing.assert_array_equal(cases["reference_later_yaw_5deg"][10], poses[10])
    assert not np.array_equal(cases["reference_later_yaw_5deg"][11], poses[11])
    results = {name: evaluate_window(track, p, np.repeat(k[None], 90, axis=0), [640, 480])
               for name, p in cases.items()}
    for result in results.values():
        assert len(result["rows"]) == 90
        assert [r["frame"] for r in result["rows"]] == list(range(90))
        assert result["rows"][0]["distance_pixels"] is None
        assert result["rows"][10]["fit_frame"]
        assert not result["rows"][11]["fit_frame"]
    assert max(r["distance_pixels"] for r in results["reference"]["rows"][1:]) < 1e-8
    assert min(r["distance_pixels"] for r in results["reference_later_yaw_5deg"]["rows"][11:]) > 40
    poses[4] = None
    with pytest.raises(ValueError, match="incomplete"):
        reference_cases(poses)
