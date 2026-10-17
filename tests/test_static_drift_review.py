import numpy as np

from paper.render_static_drift_review import project


def test_projection_uses_camera_to_world_and_rejects_behind_camera():
    pose = np.eye(4)
    pose[0, 3] = 1
    np.testing.assert_allclose(project([1, 0, 2], pose), [320.1, 247.6])
    np.testing.assert_allclose(project([0, 0, 2], pose), [320.1 - 535.4 / 2, 247.6])
    assert project([0, 0, 0], pose) is None
    assert project([0, 0, -1], pose) is None
