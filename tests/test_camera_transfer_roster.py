import copy
import json

import pytest

from paper.combine_tapvid_camera_transfer import combine
from paper.evaluate_tapvid_lora import validate_camera_transfer_roster


def manifest():
    names = [f"video{i}" for i in range(20)]
    rows = [dict(sequence=n, model=m, track=0, lag=lag, status="unavailable", label=None,
                 geometry_status="missing_prefix_observations")
            for n in names for m in ("dage", "da3") for lag in (15, 30, 60)]
    return dict(partitions={n:"test" for n in names}, track_counts={n:1 for n in names},
                planned_presentations=120, records=rows)


def test_complete_roster_can_have_no_evaluable_labels():
    result = manifest()
    validate_camera_transfer_roster(result)
    result["records"][0].update(status="labeled", label="close", geometry_status="renderable")
    validate_camera_transfer_roster(result)


@pytest.mark.parametrize("change", ["drop", "duplicate", "invent_label", "invalid_geometry", "missing_video"])
def test_roster_or_label_corruption_rejected(change):
    result = manifest()
    if change == "drop":
        result["records"].pop()
    elif change == "duplicate":
        result["records"][-1] = copy.deepcopy(result["records"][0])
    elif change == "invent_label":
        result["records"][0]["label"] = "far"
    elif change == "invalid_geometry":
        result["records"][0].update(status="labeled", label="close")
    else:
        result["track_counts"].pop("video0")
    with pytest.raises(ValueError):
        validate_camera_transfer_roster(result)


def test_missing_per_video_corpus_does_not_create_partial_test(tmp_path):
    names = sorted(f"video{i}" for i in range(20))
    (tmp_path/"manifest.json").write_text(json.dumps(dict(sequences={n:dict(split="held_out") for n in names})))
    plan = tmp_path/"protocol.json"
    plan.write_text(json.dumps(dict(sequences=names, methods=["dage", "da3"])))
    output = tmp_path/"combined"
    with pytest.raises(FileNotFoundError):
        combine(tmp_path, tmp_path/"corpora", plan, output)
    assert not output.exists()
