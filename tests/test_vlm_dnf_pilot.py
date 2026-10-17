import json

import pytest
from paper.run_vlm_dnf_pilot import parse_response


def test_abstention_retains_reason_without_inventing_evidence():
    row = {"verdict": "abstain", "reason": "Source video is missing", "evidence_panels": []}
    assert parse_response(json.dumps(row)) == (row, None)


@pytest.mark.parametrize(
    "raw,error",
    [
        ("```json\n{}\n```", "invalid_json"),
        ("[]", "invalid_keys"),
        (json.dumps({"verdict": [], "reason": "x", "evidence_panels": []}), "invalid_verdict"),
        (
            json.dumps({"verdict": "unusable", "reason": "x", "evidence_panels": []}),
            "missing_evidence_panels",
        ),
        (
            json.dumps({"verdict": "usable", "reason": "", "evidence_panels": ["top"]}),
            "missing_reason",
        ),
    ],
)
def test_invalid_model_output_is_not_silently_repaired(raw, error):
    assert parse_response(raw) == (None, error)
