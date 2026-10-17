import numpy as np

from paper.audit_rgbd_seed_transfer import project


def test_reference_transfer_uses_camera_to_world_and_positive_depth():
    k = np.array([[100., 0, 50], [0, 100, 40], [0, 0, 1]])
    pose = np.eye(4)
    assert project([0, 0, 2], pose, k) == [50, 40]
    pose[0, 3] = 1
    assert project([0, 0, 2], pose, k) == [0, 40]
    assert project([0, 0, -1], pose, k) is None
    assert project([0, 0, 0], pose, k) is None
