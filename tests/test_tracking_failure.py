"""Synthetic accounting fixtures, not experimental VLM or human labels."""

import pytest

from plumbline.tracking_failure import compare_functional_reviews, summarize_failure_review


def test_review_comparison_keeps_all_cells_and_denominators():
    labels = ["usable", "unusable", "abstain", "missing"]
    ids, predictions, references = [], [], []
    for r in labels:
        for p in labels:
            key = f"{r}-{p}"
            ids.append(key)
            if r != "missing":
                references.append(judgment(key, r))
            if p != "missing":
                predictions.append(judgment(key, p))
    result = compare_functional_reviews(ids, predictions, references)
    assert result["n_attempts"] == 16
    assert all(n == 1 for row in result["confusion_reference_rows_prediction_columns"].values()
               for n in row.values())
    for row in result["by_reference_class"].values():
        assert row == {"reference_n": 4, "decided_n": 2, "error_n": 1,
                       "unresolved_n": 2, "coverage": .5,
                       "error_rate_all_references": .25, "error_rate_decided": .5,
                       "error_rate_bounds": [.25, .75]}


def test_all_abstaining_judge_has_no_conditional_accuracy_claim():
    result = compare_functional_reviews(
        ["a"], [judgment("a", "abstain")], [judgment("a", "usable")])
    row = result["by_reference_class"]["usable"]
    assert row["coverage"] == 0
    assert row["error_rate_decided"] is None
    assert row["error_rate_bounds"] == [0, 1]
    assert result["by_reference_class"]["unusable"]["error_rate_bounds"] is None


def test_unresolved_reference_is_never_an_inferred_failure_label():
    result = compare_functional_reviews(["a"], [judgment("a", "unusable")], [])
    assert result["confusion_reference_rows_prediction_columns"]["missing"]["unusable"] == 1
    assert all(row["reference_n"] == 0 for row in result["by_reference_class"].values())


@pytest.mark.parametrize("ids,predictions,references", [
    (["a", "a"], [], []),
    ([""], [], []),
    ([], [{"attempt_id": "a"}], []),
    (["a"], [], [{"attempt_id": "a", "verdict": "usable"}]),
])
def test_review_comparison_rejects_invalid_inputs(ids, predictions, references):
    with pytest.raises(ValueError):
        compare_functional_reviews(ids, predictions, references)


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
