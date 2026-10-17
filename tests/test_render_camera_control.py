import numpy as np

from paper.render_camera_control import K, anchors, camera_pose, project_anchors, render_frame


def test_control_reproducible_and_occluder_only_changes_its_pixels():
    first, mask = render_frame(7, 11)
    again, _ = render_frame(7, 11)
    clean, empty = render_frame(7, 11, occluder=False)
    np.testing.assert_array_equal(first, again)
    np.testing.assert_array_equal(first[mask == 0], clean[mask == 0])
    assert np.any(first[mask == 1] != clean[mask == 1])
    assert not np.any(empty)
    assert first.shape == (240, 320, 3) and first.dtype == np.uint8


def test_analytic_projection_matches_independent_homogeneous_transform():
    for frame in (0, 11, 23):
        points = anchors()
        xy, visible = project_anchors(points, frame, occluder=False)
        camera = np.linalg.inv(camera_pose(frame)) @ np.c_[points, np.ones(12)].T
        projected = K @ camera[:3]
        np.testing.assert_allclose(xy, (projected[:2]/projected[2]).T, atol=1e-12)
        assert visible.all()


def test_visibility_agrees_with_raster_away_from_boundary():
    visible_count = []
    for frame in range(24):
        xy, visible = project_anchors(anchors(), frame)
        _, surface = render_frame(0, frame)
        rounded = np.rint(xy).astype(int)
        # These fixed anchors are checked against the independently rasterized surface.
        np.testing.assert_array_equal(visible, surface[rounded[:, 1], rounded[:, 0]] == 0)
        visible_count.append(visible.sum())
    assert min(visible_count) < 12
    assert max(visible_count) == 12
