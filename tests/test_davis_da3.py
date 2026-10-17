import numpy as np
import pytest
from paper.run_davis_da3 import normalized_input_intrinsics

from plumbline.image_evidence import rays_from_normalized_intrinsics


def test_resize_roundtrip_preserves_calibrated_rays():
    source_k = np.array([[[700, 0, 420], [0, 710, 230], [0, 0, 1]]], dtype=float)
    processed_k = source_k.copy()
    processed_k[:, 0] *= 252 / 854
    processed_k[:, 1] *= 140 / 480
    normalized = normalized_input_intrinsics(processed_k, (140, 252), (480, 854))
    xy = np.array([[420, 230], [100, 70], [800, 440]])
    actual = rays_from_normalized_intrinsics(xy, normalized[0], (854, 480))
    expected = np.linalg.solve(source_k[0], np.c_[xy, np.ones(3)].T).T[:, :2]
    assert np.allclose(actual, expected, atol=1e-7)


def test_bad_dimensions_rejected():
    with pytest.raises(ValueError):
        normalized_input_intrinsics(np.eye(3)[None], (0, 252), (480, 854))
