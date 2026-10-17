import pytest

from paper.analyze_layered_camera_ranking import compare, median_bracket


def bracket(lo, hi):
    return dict(status="numerical_bracket", lower=lo, upper=hi)


def test_interval_median_and_unknown_bounds():
    result = median_bracket([bracket(1, 2), bracket(3, 4)])
    assert result["lower"] == 2 and result["upper"] == 3
    unknown = dict(status="numerically_unresolved")
    result = median_bracket([bracket(1, 2), unknown])
    assert result["lower"] == .5 and result["upper"] is None
    assert result["planned_tracks"] == 2
    assert result["unresolved_tracks"] == 1


def test_unknown_does_not_become_favorable_score():
    original = dict(lower=1, upper=2)
    assert compare(original, dict(lower=0, upper=None)) == "not_separated"
    assert compare(original, dict(lower=3, upper=None)) == "changed_higher_error"
    assert compare(original, dict(lower=0, upper=.5)) == "changed_lower_error"
    assert compare(original, dict(lower=2, upper=3)) == "not_separated"


def test_invalid_and_empty_rosters_rejected():
    with pytest.raises(ValueError):
        median_bracket([])
    with pytest.raises(ValueError):
        median_bracket([bracket(2, 1)])
