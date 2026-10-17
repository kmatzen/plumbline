import numpy as np
import pytest

from paper.audit_rotation_witnesses import reproject


def test_independent_reprojection_uses_axis_residuals_and_camera_centers():
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    poses[1, 0, 3] = 1
    xy = [[320.1, 247.6], [320.1 - 535.4 / 2 + 3, 247.6 - 4]]
    np.testing.assert_allclose(reproject([0, 0, 2], poses, xy), [0, 4])
    with pytest.raises(ValueError, match="in front"):
        reproject([0, 0, -2], poses, xy)
