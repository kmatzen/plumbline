import numpy as np
import pytest

from paper.render_layered_camera_control import (
    K, anchors, camera_pose, intersections, project_anchors, render_frame,
)


def test_nearest_surface_and_forward_ray_only():
    rays = np.array([[-.75, 0, 3], [.75, 0, 4], [0, 0, 6], [0, 0, -1.]])
    points, ids = intersections(np.zeros(3), rays)
    np.testing.assert_array_equal(ids, [1, 2, 0, -1])
    np.testing.assert_allclose(points[:3], rays[:3])
    assert np.isnan(points[3]).all()


def test_layered_reference_projection_and_raster_visibility():
    points, ids = anchors()
    assert len(np.unique(points[:, 2])) == 3
    for frame in range(24):
        xy, visible = project_anchors(points, ids, frame)
        camera = np.linalg.inv(camera_pose(frame)) @ np.c_[points, np.ones(12)].T
        p = K @ camera[:3]
        np.testing.assert_allclose(xy, (p[:2]/p[2]).T, atol=1e-12)
        rgb, surfaces = render_frame(5, frame)
        rounded = np.rint(xy).astype(int)
        # Rounding can cross a silhouette. Compare only pixels whose full
        # 3x3 neighborhood belongs to one surface; retain analytic visibility
        # at boundary points rather than silently snapping the reference.
        interior = np.array([len(np.unique(surfaces[y-1:y+2, x-1:x+2])) == 1
                             for x, y in rounded])
        assert interior.sum() >= 8
        np.testing.assert_array_equal(visible[interior],
            (surfaces[rounded[:, 1], rounded[:, 0]] == ids)[interior])
        assert rgb.shape == (240, 320, 3)
        assert rgb.dtype == np.uint8
        assert set(np.unique(surfaces)) == {0, 1, 2}


def test_reproducible_and_bad_reference_rejected():
    a, ids = render_frame(9, 12)
    b, again = render_frame(9, 12)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(ids, again)
    points, surfaces = anchors()
    points[0, 2] = 7
    with pytest.raises(ValueError, match="declared surface"):
        project_anchors(points, surfaces, 0)


def test_background_occlusion_is_not_static_surface_motion():
    # At frame zero, the first ray crosses the left panel at x=-.75,
    # the second passes through the gap, and the third crosses the right.
    _, visible = project_anchors(np.array([[-1.5, 0, 6], [0, 0, 6], [1.125, 0, 6]]),
                                 np.zeros(3, dtype=int), 0)
    np.testing.assert_array_equal(visible, [False, True, False])
