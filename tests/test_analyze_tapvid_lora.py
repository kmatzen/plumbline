import pytest

from paper.analyze_tapvid_lora import compare, metrics


def test_abstentions_and_invalids_count_as_not_correct():
    rows = [dict(target="close", answer="close"), dict(target="far", answer="uncertain"),
            dict(target="occluded", answer="invalid")]
    m = metrics(rows)
    assert m["accuracy_all_examples"] == 1/3
    assert m["balanced_accuracy"] == 1/3
    assert m["answered_examples"] == 1
    assert m["uncertain_examples"] == m["invalid_examples"] == 1
    assert m["close_precision"] == 1
    assert m["answer_coverage"] == 1/3
    assert m["accuracy_answered_only"] == 1


def test_zero_close_coverage_is_undefined_not_perfect():
    m = metrics([dict(target="close", answer="uncertain")])
    assert m["close_precision"] is None
    assert m["balanced_accuracy"] is None
    assert m["accuracy_all_examples"] == 0
    assert m["answer_coverage"] == 0
    assert m["accuracy_answered_only"] is None


def test_format_compliance_is_not_class_recognition():
    m = metrics([dict(target="close", answer="far"),
                 dict(target="far", answer="close"),
                 dict(target="occluded", answer="far")])
    assert m["answer_coverage"] == 1
    assert m["accuracy_answered_only"] == 0
    assert m["accuracy_all_examples"] == 0


def test_empty_metrics_have_no_accuracy_or_coverage():
    m = metrics([])
    assert m["accuracy_all_examples"] is None
    assert m["answer_coverage"] is None
    assert m["accuracy_answered_only"] is None


def test_partial_run_cannot_be_summarized_as_complete():
    with pytest.raises(ValueError, match="incomplete"):
        compare(dict(status="evaluating"), {}, {})
    with pytest.raises(ValueError, match="missing"):
        compare(dict(status="complete", results=dict(base=[], adapted=[])),
                dict(records=[dict(id="x", partition="validation", status="labeled")]), {})
