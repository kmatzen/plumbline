import numpy as np

from paper.probe_available_fit_consistency import compare_available
from paper.probe_evidence_sensitivity import alternatives, joint_status
from paper.render_camera_control import K, anchors, camera_pose, project_anchors


def inputs():
    cameras = np.linalg.inv(np.stack([camera_pose(t) for t in range(24)]))
    track = [project_anchors(anchors(), t, occluder=False)[0][0].tolist() for t in range(24)]
    return cameras, track


def test_profile_keeps_fitting_cameras_and_identity_unchanged():
    cameras, _ = inputs()
    changed = dict(alternatives(cameras))
    assert len(changed) == 14
    np.testing.assert_array_equal(changed["identity"], cameras)
    for name, candidate in changed.items():
        if name != "global_rigid_gauge":
            np.testing.assert_allclose(candidate[:12], cameras[:12], atol=1e-14)


def test_compact_joint_status_matches_full_check_and_gauge():
    cameras, track = inputs()
    track[4] = None
    baseline = joint_status(cameras, track)["status"]
    for name, candidate in alternatives(cameras):
        actual = joint_status(candidate, track)["status"]
        expected = compare_available(candidate[:12], track[:12], candidate[12:], track[12:], K, 2)["status"]
        assert actual == expected
        if name == "global_rigid_gauge":
            assert actual == baseline
