import itertools

import pytest

from plumbline.placement_bounds import paired_bounds


def test_no_observations_do_not_imply_perfect_consistency():
    result = paired_bounds([None] * 3, [None] * 3, 2)
    assert result["left_minus_right_bounds"] == [-1, 1]
    assert result["ordering"] == "unresolved"


def test_complete_observations_identify_order():
    assert paired_bounds([0, 0], [3, 3], 2)["ordering"] == "left"
    assert paired_bounds([0, 0], [0, 0], 2)["ordering"] == "equal"


def test_bounds_equal_exhaustive_missing_outcome_extremes():
    left, right = [0, None, 4], [None, 3, None]
    result = paired_bounds(left, right, 2)
    differences = []
    for a, b, c in itertools.product([0, 1], repeat=3):
        differences.append(((1 + a) - (b + c)) / 3)
    assert result["left_minus_right_bounds"] == pytest.approx([min(differences), max(differences)])


def test_some_missingness_can_still_identify_order():
    assert paired_bounds([0, 0, 0, None], [4, 4, 4, None], 2)["ordering"] == "left"


@pytest.mark.parametrize(
    "left,right,tolerance",
    [([], [], 2), ([0], [], 2), ([float("nan")], [0], 2), ([-1], [0], 2), ([0], [0], -1)],
)
def test_invalid_inputs_rejected(left, right, tolerance):
    with pytest.raises(ValueError):
        paired_bounds(left, right, tolerance)
