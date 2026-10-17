import numpy as np

from paper.filter_layered_image_evidence import admit


def test_admission_preserves_coordinates_and_missingness():
    rng = np.random.default_rng(1)
    image = rng.integers(0, 256, (64, 64), dtype=np.uint8)
    tracks = [[[30., 30.], [30., 30.], None]]
    retained, scores = admit([image]*3, tracks)
    assert retained == tracks
    assert scores[0][1] > .99
    assert scores[0][2] is None


def test_image_change_rejects_without_discarding_initial_query():
    image = np.random.default_rng(2).integers(0, 256, (64, 64), dtype=np.uint8)
    retained, scores = admit([image, 255-image], [[[30., 30.], [30., 30.]]])
    assert retained == [[[30., 30.], None]]
    assert scores[0][1] < -.99
