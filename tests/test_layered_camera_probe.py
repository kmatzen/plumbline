import numpy as np

from paper.probe_layered_camera_control import score_tracks, track_rgb


def test_scoring_keeps_missing_occluded_and_wrong_separate():
    tracks = [[[0, 0], None, [0, 0], [4, 0], [1, 0]]]
    reference = np.zeros((5, 1, 2))
    visible = np.array([[True], [True], [False], [True], [True]])
    assert score_tracks(tracks, reference, visible) == dict(
        planned=4, accepted=3, correct_visible_within_2px=1,
        accepted_occluded=1, accepted_visible_over_2px=1)


def test_rgb_tracker_on_repeated_image_has_no_reference_input():
    rng = np.random.default_rng(7)
    image = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    tracks = track_rgb([image]*4, np.array([[20, 20], [40, 40]]))
    np.testing.assert_allclose(tracks, [[[20, 20]]*4, [[40, 40]]*4], atol=.01)
