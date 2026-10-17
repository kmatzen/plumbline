import numpy as np
from paper.probe_tum_anchors import camera_variants


def test_controls_leave_early_cameras_unchanged_except_explicit_reversal():
    poses = np.tile(np.eye(4), (20, 1, 1))
    poses[:, 0, 3] = np.arange(20)
    original = poses.copy()
    variants = camera_variants(poses, poses.copy())
    np.testing.assert_array_equal(poses, original)
    np.testing.assert_array_equal(variants["reference_later_yaw_5deg"][:11], poses[:11])
    np.testing.assert_array_equal(variants["reference_later_yaw_5deg"][:, :3, 3], poses[:, :3, 3])
    np.testing.assert_array_equal(
        variants["reference_reversed_centers"][:, :3, 3], -poses[:, :3, 3]
    )
