import pytest

from paper.analyze_camera_evidence_coverage import classify


def record(indices, lo=0, hi=.01):
    return dict(status="numerical_bracket", indices=indices, lower=lo, upper=hi)


def test_missing_changed_views_is_not_a_tie():
    result = classify(record([0, 1]), record([0, 1]), [12, 13])
    assert result["verdict"] == "no_changed_view_evidence"
    assert result["raw_bracket_comparison"] == "overlapping_brackets"


def test_supported_tie_and_unresolved_are_preserved():
    assert classify(record([0, 12]), record([0, 12]), [12])["verdict"] == "overlapping_brackets"
    unresolved = dict(status="numerically_unresolved", indices=[0, 12])
    assert classify(record([0, 12]), unresolved, [12])["verdict"] == "unresolved"
    assert classify(record([0, 12]), record([0, 12], 1, 2), [12])["verdict"] == "changed_higher_error"


def test_mismatched_or_duplicate_indices_rejected():
    with pytest.raises(ValueError):
        classify(record([0, 1]), record([0, 12]), [12])
    with pytest.raises(ValueError):
        classify(record([0, 0]), record([0, 0]), [12])


def test_identity_is_not_a_discrimination_trial():
    assert classify(record([0, 1]), record([0, 1]), [])["verdict"] == "no_camera_change"
