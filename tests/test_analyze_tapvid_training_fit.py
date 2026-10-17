import pytest

from paper.analyze_tapvid_training_fit import analyze


def fixture():
    item = dict(id="x", sequence="a", target_for_scoring_only="close",
                raw_response="far", parsed="far")
    report = dict(status="complete", partition="train", results=dict(adapted=[item]))
    training = dict(status="complete", training_ids=["x"])
    corpus = dict(records=[dict(id="x", sequence="a", label="close", partition="train", status="labeled")])
    return report, training, corpus


def test_training_fit_keeps_errors_and_marks_in_sample_scope():
    r = analyze(*fixture())
    assert r["all"]["accuracy_all_examples"] == 0
    assert r["training_majority_accuracy"] == 1
    assert "in-sample" in r["scope"]


def test_training_fit_rejects_partial_or_wrong_partition():
    for key, value in (("status", "evaluating"), ("partition", "validation")):
        report, training, corpus = fixture()
        report[key] = value
        with pytest.raises(ValueError, match="complete training-fit"):
            analyze(report, training, corpus)


def test_training_fit_rejects_duplicate_and_repaired_answers():
    report, training, corpus = fixture()
    report["results"]["adapted"] *= 2
    with pytest.raises(ValueError, match="duplicate"):
        analyze(report, training, corpus)
    report, training, corpus = fixture()
    report["results"]["adapted"][0]["raw_response"] = "maybe far"
    with pytest.raises(ValueError, match="parser mismatch"):
        analyze(report, training, corpus)
