import numpy as np
import pytest
from paper.build_anchor_lesson import build, demonstration_offset


def test_teaching_offset_is_explicit_not_estimated_camera_motion():
    for frame in range(11):
        assert np.array_equal(demonstration_offset(frame), [0, 0])
    offsets = np.array([demonstration_offset(i) for i in range(60)])
    assert np.all(np.diff(offsets[:, 0]) >= 0)
    assert np.all(offsets[:, 1] == 0)
    assert offsets[-1, 0] == 50


def test_lesson_does_not_overwrite_previous_outputs(tmp_path):
    with pytest.raises(ValueError, match="preserve"):
        build(tmp_path, tmp_path, tmp_path, tmp_path)
