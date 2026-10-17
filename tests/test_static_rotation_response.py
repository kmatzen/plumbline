import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from paper.probe_static_rotation_response import perturb


def test_rotation_schedules_preserve_centers_early_window_and_inputs():
    poses = [None] + [np.eye(4) for _ in range(89)]
    for i, pose in enumerate(poses[1:], 1):
        pose[:3, :3] = Rotation.from_euler("x", 20, degrees=True).as_matrix()
        pose[0, 3] = i / 100
    original = np.stack(poses[1:]).copy()
    for schedule in ("step", "ramp"):
        result = perturb(poses, -5, schedule)
        np.testing.assert_array_equal(np.stack(result[1:11]), original[:10])
        np.testing.assert_array_equal(np.stack(result[1:])[:, :3, 3], original[:, :3, 3])
        for frame in (11, 50, 89):
            relative = poses[frame][:3, :3].T @ result[frame][:3, :3]
            expected = -5 * (1 if schedule == "step" else (frame - 10) / 79)
            np.testing.assert_allclose(relative, Rotation.from_euler("y", expected, degrees=True).as_matrix(), atol=1e-14)
        np.testing.assert_array_equal(np.stack(poses[1:]), original)
        np.testing.assert_array_equal(np.stack(perturb(poses, 0, schedule)[1:]), original)
    with pytest.raises(ValueError):
        perturb(poses, np.nan, "step")
    with pytest.raises(ValueError):
        perturb(poses, 5, "unknown")
