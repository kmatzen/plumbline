import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from paper.run_vlm_dnf_pilot import audit_processed_inputs, parse_response, run


def test_description_preserves_raw_text_without_treating_it_as_a_camera_label():
    assert parse_response("A blue square.", "description") == (
        {"description": "A blue square."},
        None,
    )
    assert parse_response("  ", "description") == (None, "empty_description")


def test_input_audit_checks_payload_and_placeholder_count():
    import torch

    processor = SimpleNamespace(image_processor=SimpleNamespace(merge_size=2))
    model = SimpleNamespace(config=SimpleNamespace(image_token_id=42))
    inputs = {
        "pixel_values": torch.arange(16).reshape(4, 4).float(),
        "image_grid_thw": torch.tensor([[1, 4, 4]]),
        "input_ids": torch.tensor([[0, 42, 42, 42, 42, 1]]),
    }
    audit = audit_processed_inputs(inputs, processor, model)
    assert audit["pixel_finite"] and audit["placeholder_count_matches"]
    assert audit["actual_image_tokens"] == audit["expected_image_tokens"] == 4
    inputs["pixel_values"][0, 0] = 99
    assert (
        audit_processed_inputs(inputs, processor, model)["pixel_float32_sha256"]
        != audit["pixel_float32_sha256"]
    )
    inputs["input_ids"] = torch.tensor([[0, 42, 1]])
    assert not audit_processed_inputs(inputs, processor, model)["placeholder_count_matches"]
    inputs["pixel_values"][0, 0] = float("nan")
    assert not audit_processed_inputs(inputs, processor, model)["pixel_finite"]


def test_abstention_retains_reason_without_inventing_evidence():
    row = {"verdict": "abstain", "reason": "Source video is missing", "evidence_panels": []}
    assert parse_response(json.dumps(row)) == (row, None)


def test_unknown_quantization_rejected_before_model_loading(tmp_path):
    with pytest.raises(ValueError, match="unknown quantization"):
        run(Path("missing"), Path("missing.png"), tmp_path / "output.json", quantization="int8")
    assert not (tmp_path / "output.json").exists()


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


def test_geometry_verdict_is_not_a_dnf_label():
    row = {"verdict": "consistent", "reason": "centers meet lines", "evidence_panels": ["1"]}
    assert parse_response(json.dumps(row), "geometry") == (row, None)
    assert parse_response(json.dumps(row), "dnf") == (None, "invalid_verdict")
    row["verdict"] = "usable"
    assert parse_response(json.dumps(row), "geometry") == (None, "invalid_verdict")


def test_source_identity_does_not_accept_camera_verdicts_or_unknown_panels():
    row = {"verdict": "identifiable", "reason": "marked beam", "evidence_panels": ["detail_crop"]}
    assert parse_response(json.dumps(row), "source_identity") == (row, None)
    assert parse_response(json.dumps(row), "dnf") == (None, "invalid_verdict")
    row["evidence_panels"] = ["trajectory"]
    assert parse_response(json.dumps(row), "source_identity") == (None, "invalid_evidence_panels")
    row["verdict"] = "usable"
    assert parse_response(json.dumps(row), "source_identity") == (None, "invalid_verdict")
