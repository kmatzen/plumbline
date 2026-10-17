"""Artificial labels here test accounting only; they are not research data."""

import pytest
from paper.analyze_review_study import summarize


def fixture():
    return {
        "study_id": "s",
        "unique_pairs": 1,
        "presentations": [
            {
                "presentation_id": "p",
                "pair_key": "k",
                "system_a": "x",
                "system_b": "y",
                "repeat": False,
            },
            {
                "presentation_id": "r",
                "pair_key": "k",
                "system_a": "y",
                "system_b": "x",
                "repeat": True,
            },
        ],
    }


def session(reviewer, first, repeated):
    return {
        "study_id": "s",
        "reviewer": reviewer,
        "responses": [
            {"presentation_id": "p", "winner": first},
            {"presentation_id": "r", "winner": repeated},
        ],
    }


def test_repeats_are_canonicalized_and_do_not_add_votes():
    result = summarize(fixture(), [session("one", "a", "b"), session("two", "a", "b")])
    assert result["consensus"][0]["counts"] == {"x": 2}
    assert result["consensus"][0]["verdict"] == "x"
    assert all(r["agreement"] for r in result["repeat_checks"])


def test_divided_or_single_reviewers_do_not_create_consensus():
    assert summarize(fixture(), [session("one", "a", "b")])["resolved_pairs"] == 0
    result = summarize(fixture(), [session("one", "a", "b"), session("two", "b", "a")])
    assert result["resolved_pairs"] == 0
    with pytest.raises(ValueError, match="distinct"):
        summarize(fixture(), [session("same", "a", "b"), session("same", "a", "b")])


def test_abstention_does_not_count_as_tie():
    result = summarize(
        fixture(), [session("one", "abstain", "abstain"), session("two", "tie", "tie")]
    )
    assert result["abstentions"] == 2
    assert result["consensus"][0]["counts"] == {"tie": 1}
    assert result["resolved_pairs"] == 0


def test_partial_overlap_and_abstention_agreement():
    first = session("one", "a", "b")
    second = session("two", "a", "b")
    second["responses"] = second["responses"][:1]
    result = summarize(fixture(), [first, second])
    agreement = result["inter_reviewer_agreement"][0]
    assert agreement["shared_original_pairs"] == 1
    assert agreement["jointly_usable_pairs"] == 1
    assert agreement["exact_agreement_rate"] == 1
    second["responses"][0]["winner"] = "abstain"
    agreement = summarize(fixture(), [first, second])["inter_reviewer_agreement"][0]
    assert agreement["shared_original_pairs"] == 1
    assert agreement["exact_agreement_rate"] is None


def test_repeat_only_partial_session_does_not_create_consensus():
    partial = session("one", "a", "b")
    partial["responses"] = partial["responses"][1:]
    result = summarize(fixture(), [partial])
    assert result["responses"] == 1
    assert result["unreviewed_pairs"] == 1
    assert result["resolved_pairs"] == 0


def test_manifest_cannot_count_one_reviewer_twice():
    study = fixture()
    study["presentations"][1]["repeat"] = False
    with pytest.raises(ValueError, match="multiple original"):
        summarize(study, [session("one", "a", "b")])


def test_repeat_systems_must_match_original():
    study = fixture()
    study["presentations"][1]["system_b"] = "z"
    with pytest.raises(ValueError, match="inconsistent systems"):
        summarize(study, [])
