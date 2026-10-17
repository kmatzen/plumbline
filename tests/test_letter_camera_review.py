import numpy as np

from paper.build_letter_camera_review import project_marker


def test_saved_nonorthonormal_rotation_uses_full_camera_inverse():
    pose = np.eye(4)
    pose[0, 0] += 2e-6
    pose[0, 3] = .2
    point = np.array([.5, .1, 4.])
    xyz = np.linalg.inv(pose) @ np.r_[point, 1.]
    expected = [535.4*xyz[0]/xyz[2]+320.1, 539.2*xyz[1]/xyz[2]+247.6]
    np.testing.assert_allclose(project_marker(point, pose), expected, rtol=0, atol=1e-12)


def test_point_behind_camera_has_no_display_marker():
    assert project_marker(np.array([0., 0., -1.]), np.eye(4)) is None
