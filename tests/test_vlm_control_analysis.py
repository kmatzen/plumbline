import json

import pytest
from paper.analyze_vlm_controls import assess


def result(verdict):
    parsed = {"verdict": verdict, "reason": "visible relation", "evidence_panels": ["1"]}
    return {
        "status": "complete",
        "raw_response": json.dumps(parsed),
        "parsed_response": parsed,
        "parse_error": None,
    }


def test_abstention_and_failed_execution_are_not_correct_answers():
    assert assess(result("abstain"), "consistent")["matches_control"] is None
    assert assess({"status": "failed"}, "consistent")["matches_control"] is None
    assert assess(result("consistent"), "consistent")["matches_control"] is True
    assert assess(result("inconsistent"), "consistent")["matches_control"] is False


def test_fenced_answer_is_not_repaired():
    r = result("consistent")
    r.update(
        raw_response="```json\n" + r["raw_response"] + "\n```",
        parsed_response=None,
        parse_error="invalid_json",
    )
    assert assess(r, "consistent")["matches_control"] is None
    r["parse_error"] = None
    with pytest.raises(ValueError, match="strict reparse"):
        assess(r, "consistent")
