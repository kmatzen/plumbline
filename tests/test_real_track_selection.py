import pytest

from paper.audit_real_track_selection import audit_pair


def tracks(values):
    return [dict(slot=i, indices=[1, 2], status="numerical_bracket", lower=v, upper=v)
            for i, v in enumerate(values)]


def test_enumerates_all_subsets_and_detects_one_removal_reversal():
    result = audit_pair(tracks([0, 0, 10]), tracks([2, 2, 2]))
    assert len(result["subsets"]) == 7
    assert result["full_roster_verdict"] == "first_lower"
    assert result["minimum_removals_to_reverse_preference"] == 1
    assert result["by_retained_count"][1]["counts"] == {"first_lower": 1, "second_lower": 2}


def test_unresolved_tracks_are_not_discarded():
    first = tracks([0, 0])
    first[1]["status"] = "numerically_unresolved"
    result = audit_pair(first, tracks([2, 2]))
    assert result["full_roster_verdict"] == "not_separated"
    assert result["subsets"][0]["first"]["unresolved_tracks"] == 1
    assert result["subsets"][0]["first"]["upper"] is None


def test_mismatched_evidence_and_duplicate_rosters_rejected():
    first, second = tracks([0, 1]), tracks([2, 3])
    second[0]["indices"] = [1, 3]
    with pytest.raises(ValueError, match="indices"):
        audit_pair(first, second)
    with pytest.raises(ValueError, match="rosters"):
        audit_pair(first+first, first+first)
