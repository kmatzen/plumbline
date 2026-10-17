"""Synthetic accounting fixtures, not experimental VLM or human labels."""

import pytest

from plumbline.tracking_failure import summarize_failure_review


def judgment(key, verdict):
    return {"attempt_id": key, "verdict": verdict, "evidence_id": "packet-hash", "reason": "test"}


def test_rejecting_bad_results_cannot_erase_attempts():
    attempts = [
        {"attempt_id": "a", "status": "complete", "metrics": {"ate": 1.0}},
        {"attempt_id": "b", "status": "complete", "metrics": {"ate": 99.0}},
        {"attempt_id": "c", "status": "crash"},
        {"attempt_id": "d", "status": "pending"},
    ]
    result = summarize_failure_review(
        attempts, [judgment("a", "usable"), judgment("b", "unusable")], metric="ate"
    )
    assert result["n_attempts"] == 4
    assert result["all_complete_metric_mean"] == 50
    assert result["accepted_metric_mean"] == 1
    assert result["failure_rate_bounds"] == [0.5, 0.75]
    assert result["functional_dnf_rate"] == 0.25
    assert result["mechanical_dnf_rate"] == 0.25


def test_missing_and_abstaining_judgments_never_pass():
    attempts = [{"attempt_id": str(i), "status": "complete"} for i in range(2)]
    result = summarize_failure_review(attempts, [judgment("0", "abstain")], metric="ate")
    assert result["accepted_metric_n"] == 0
    assert result["accepted_metric_mean"] is None
    assert result["unresolved_rate"] == 1
    assert result["failure_rate_bounds"] == [0, 1]


def test_empty_roster_is_not_perfect_performance():
    result = summarize_failure_review([], [], metric="ate")
    assert result["failure_rate_bounds"] is None
    assert result["mechanical_dnf_rate"] is None


def test_reject_malformed_accounting():
    row = {"attempt_id": "a", "status": "complete", "metrics": {"ate": float("nan")}}
    with pytest.raises(ValueError, match="finite"):
        summarize_failure_review([row], [], metric="ate")
    with pytest.raises(ValueError, match="distinct"):
        summarize_failure_review([row, row], [], metric="ate")
    with pytest.raises(ValueError, match="unknown"):
        summarize_failure_review([], [judgment("a", "usable")], metric="ate")
    with pytest.raises(ValueError, match="only complete"):
        summarize_failure_review(
            [{"attempt_id": "a", "status": "crash"}], [judgment("a", "usable")], metric="ate"
        )
