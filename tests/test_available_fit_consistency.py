import numpy as np

from paper.probe_available_fit_consistency import compare_available
from paper.probe_joint_query_consistency import compare_queries
from paper.render_camera_control import K, anchors, camera_pose, project_anchors


def inputs():
    cameras = np.linalg.inv(np.stack([camera_pose(t) for t in range(24)]))
    xy = [project_anchors(anchors(), t, occluder=False)[0][0].tolist() for t in range(24)]
    return cameras, xy


def test_complete_input_matches_original():
    cameras, xy = inputs()
    result = compare_available(cameras[:12], xy[:12], cameras[12:], xy[12:], K, 2)
    original = compare_queries(cameras[:12], xy[:12], cameras[12:], xy[12:], K, 2)
    assert all(result[key] == value for key, value in original.items())


def test_missing_observation_uses_remaining_views_without_inventing_coordinates():
    cameras, xy = inputs()
    xy[4] = None
    result = compare_available(cameras[:12], xy[:12], cameras[12:], xy[12:], K, 2)
    assert result["observed_fitting_indices"] == [i for i in range(12) if i != 4]
    assert result["planned_fitting_observations"] == 12
    assert result["status"] == "jointly_compatible"


def test_one_fitting_view_and_missing_queries_are_unresolved():
    cameras, xy = inputs()
    result = compare_available(cameras[:12], [xy[0]] + [None]*11, cameras[12:], xy[12:], K, 2)
    assert result["status"] == "insufficient_fitting_observations"
    result = compare_available(cameras[:12], xy[:12], cameras[12:], [None]*12, K, 2)
    assert result["status"] == "missing_query_observations"
