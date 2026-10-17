import copy
import json
from pathlib import Path

import pytest

from paper.compare_crop_scale import compare_pair


def test_retained_pair_and_invalid_comparisons():
    root = Path(__file__).resolve().parents[1] / "paper/results"
    original = json.loads((root / "source-crops-v1/slot-0.json").read_text())
    scaled = json.loads((root / "source-crops-scale-v1/slot-0.json").read_text())
    result = compare_pair(original, scaled, original["image_sha256"])
    assert result["visual_correctness"] == "not_assessed"
    assert result["original_response"] == original["raw_response"]
    for key, value in [("prompt_sha256", "changed"), ("status", "running"),
                       ("image_sha256", "changed"), ("seed", 42)]:
        bad = copy.deepcopy(scaled)
        bad[key] = value
        with pytest.raises(ValueError):
            compare_pair(original, bad, original["image_sha256"])
    bad = copy.deepcopy(scaled)
    bad["settings"]["do_sample"] = True
    with pytest.raises(ValueError, match="settings"):
        compare_pair(original, bad, original["image_sha256"])
    bad = copy.deepcopy(scaled)
    bad["processed_input_audit"]["actual_image_tokens"] = 36
    with pytest.raises(ValueError, match="realized"):
        compare_pair(original, bad, original["image_sha256"])
