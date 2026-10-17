import numpy as np
import pytest

from paper.build_track_identity_pairs import pair_image


def test_duplicate_panels_are_identical_below_labels_and_centers_unobscured():
    crop = np.random.default_rng(1).integers(0, 256, (161, 161, 3), dtype=np.uint8)
    panel = pair_image(crop, crop)
    np.testing.assert_array_equal(panel[32:, :161], panel[32:, 177:338])
    np.testing.assert_array_equal(panel[112, 80], crop[80, 80])
    np.testing.assert_array_equal(panel[112, 257], crop[80, 80])
    with pytest.raises(ValueError, match="native"):
        pair_image(crop[:100], crop)
