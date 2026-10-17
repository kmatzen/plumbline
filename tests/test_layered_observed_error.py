import numpy as np

from paper.probe_layered_observed_error import assess, compare_brackets


def test_bracket_comparison_retains_overlap_and_uncertainty():
    def bracket(a, b):
        return dict(status="numerical_bracket", lower=a, upper=b)
    original = bracket(1, 2)
    assert compare_brackets(original, bracket(3, 4)) == "changed_higher_error"
    assert compare_brackets(original, bracket(0, .5)) == "changed_lower_error"
    assert compare_brackets(original, bracket(2, 3)) == "overlapping_brackets"
    assert compare_brackets(original, dict(status="numerically_unresolved")) == "unresolved"


def test_missing_observations_are_not_zero_error():
    result = assess(np.repeat(np.eye(4)[None], 24, axis=0), [[10, 10]]+[None]*23)
    assert result == dict(status="insufficient_observations", indices=[0])
