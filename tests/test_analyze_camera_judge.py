import copy
from collections import Counter

import pytest

from paper.analyze_camera_judge import summarize


def fixture():
    records = [dict(id=f"{m}:{s}:{f}", model=m, slot=s, frame=f, status="missing_prefix_observations")
               for m in ("dage", "da3") for s in range(12) for f in (30, 60, 89)]
    records[0]["status"] = "renderable"
    row = dict(id=records[0]["id"], target_for_scoring_only=None, parsed="uncertain",
               raw_response="uncertain", processed_pixel_sha256="same", image_tokens=609)
    report = dict(status="complete", partition="camera", planned_examples=72, renderable_examples=1,
                  unavailable_counts=dict(Counter(r["status"] for r in records)),
                  results=dict(base=[row], adapted=[copy.deepcopy(row)]))
    return report, dict(records=records)


def test_unavailable_and_uncertain_are_not_failures():
    report, corpus = fixture()
    result = summarize(report, corpus)
    assert result["planned_presentations"] == 72
    assert result["arms"]["adapted"]["dage"]["predicted_labels"] == {"uncertain": 1}
    assert result["arms"]["adapted"]["da3"]["presented"] == 0
    assert result["arms"]["adapted"]["da3"]["unavailable"] == {"missing_prefix_observations": 36}
    assert "accuracy" not in result


@pytest.mark.parametrize("change", ["pixel", "target", "duplicate", "parser", "incomplete"])
def test_invalid_comparisons_rejected(change):
    report, corpus = fixture()
    row = report["results"]["adapted"][0]
    if change == "pixel":
        row["processed_pixel_sha256"] = "different"
    elif change == "target":
        row["target_for_scoring_only"] = "close"
    elif change == "duplicate":
        report["results"]["adapted"].append(copy.deepcopy(row))
    elif change == "parser":
        row["parsed"] = "close"
    else:
        report["status"] = "evaluating"
    with pytest.raises(ValueError):
        summarize(report, corpus)
