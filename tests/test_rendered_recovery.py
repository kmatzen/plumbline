import cv2
import numpy as np

from paper.probe_rendered_recovery import match_source
from paper.render_camera_control import render_frame


def test_identical_textured_image_preserves_queries():
    rgb, _ = render_frame(0, 0, occluder=False)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    queries = [[100., 100.], [160., 120.]]
    rows = match_source(gray, gray, queries)
    assert all(r["accepted"] for r in rows)
    np.testing.assert_allclose([r["predicted_xy"] for r in rows], queries, atol=.01)


def test_blank_target_does_not_become_recovered_correspondence():
    rgb, _ = render_frame(0, 0, occluder=False)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    rows = match_source(gray, np.zeros_like(gray), [[160., 120.]])
    assert not rows[0]["accepted"]
