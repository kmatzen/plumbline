import numpy as np

from paper.build_camera_judge_corpus import fit_marker, query_marker
from paper.build_letter_camera_review import project_marker


def test_lost_later_tracks_do_not_prevent_projected_queries():
    poses = np.repeat(np.eye(4)[None], 90, axis=0)
    poses[:, 0, 3] = np.linspace(0, .5, 90)
    point = np.array([0., 0., 4.])
    track = [dict(xy=project_marker(point, p).tolist() if i <= 10 else None)
             for i, p in enumerate(poses)]
    fit = fit_marker(track, poses)
    assert fit["status"] == "fitted"
    np.testing.assert_allclose(fit["point_world"], point, atol=1e-8)
    result = query_marker(fit, poses[89])
    assert result["status"] == "renderable"
    np.testing.assert_allclose(result["candidate_xy"], project_marker(point, poses[89]), atol=1e-8)
    track[89]["xy"] = [1e6, 1e6]
    assert fit_marker(track, poses) == fit


def test_missing_prefix_and_invalid_projection_are_not_labels():
    poses = np.repeat(np.eye(4)[None], 90, axis=0)
    fit = fit_marker([dict(xy=None) for _ in poses], poses)
    assert query_marker(fit, poses[0]) == {"status": "missing_prefix_observations"}
    assert query_marker(dict(status="fitted", point_world=[0, 0, -1]), poses[0])["status"] == "nonpositive_query_depth"
    assert query_marker(dict(status="fitted", point_world=[100, 0, 1]), poses[0])["status"] == "query_outside_image"
