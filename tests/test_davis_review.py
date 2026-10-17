import copy

import pytest
from paper.analyze_davis_review import analyze


def fixture():
    study = {
        "study_id": "s",
        "assets": [{"asset_id": "a"}, {"asset_id": "b"}],
        "presentations": [
            {"presentation_id": "p1", "asset_id": "a"},
            {"presentation_id": "p2", "asset_id": "b"},
            {"presentation_id": "p3", "asset_id": "a"},
        ],
    }
    submission = {
        "schema": "davis-usability-pilot-v1",
        "study_id": "s",
        "reviewer": "test",
        "responses": [
            {
                "presentation_id": "p1",
                "verdict": "abstain",
                "reason": "Insufficient evidence",
                "frame_indices_text": "",
                "recorded_at": "2026-10-17T01:00:00Z",
            }
        ],
    }
    return study, submission


def test_partial_review_does_not_impute_missing_or_count_repeats_twice():
    study, submission = fixture()
    result = analyze(study, submission)
    assert result["unique_output_status_counts"] == {"abstain": 1, "unreviewed": 1}
    assert result["repeat_checks"][0]["agreement"] is None
    second = copy.deepcopy(submission["responses"][0])
    second["presentation_id"] = "p3"
    submission["responses"].append(second)
    result = analyze(study, submission)
    assert result["presentation_verdict_counts"] == {"abstain": 2}
    assert result["unique_output_status_counts"] == {"abstain": 1, "unreviewed": 1}
    assert result["repeat_checks"][0]["agreement"] is True
    second["verdict"] = "unusable"
    result = analyze(study, submission)
    assert result["unique_output_status_counts"] == {"inconsistent": 1, "unreviewed": 1}
    assert result["repeat_checks"][0]["agreement"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("presentation_id", "unknown"),
        ("verdict", "DNF"),
        ("reason", " "),
        ("recorded_at", "2026-10-17"),
        ("frame_indices_text", [1, 2]),
    ],
)
def test_invalid_answer_rejected(field, value):
    study, submission = fixture()
    submission["responses"][0][field] = value
    with pytest.raises(ValueError):
        analyze(study, submission)


def test_duplicate_and_wrong_study_rejected():
    study, submission = fixture()
    submission["responses"] *= 2
    with pytest.raises(ValueError, match="duplicate"):
        analyze(study, submission)
    submission["study_id"] = "other"
    with pytest.raises(ValueError, match="mismatch"):
        analyze(study, submission)


def test_empty_submission_retains_all_unreviewed():
    study, submission = fixture()
    submission["responses"] = []
    assert analyze(study, submission)["unique_output_status_counts"] == {"unreviewed": 2}
