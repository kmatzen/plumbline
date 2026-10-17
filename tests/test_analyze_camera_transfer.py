import copy

import pytest

from paper.analyze_camera_transfer import analyze


def fixture():
    names = [f"video{i}" for i in range(20)]
    rows = [dict(id=f"{n}:{m}:{lag}", sequence=n, partition="test", model=m, track=0, lag=lag,
                 status="unavailable", label=None, geometry_status="missing_prefix_observations",
                 source_patch_ncc=None, patch_baseline_prediction="uncertain")
            for n in names for m in ("dage", "da3") for lag in (15, 30, 60)]
    for index, label in ((0,"far"),(6,"occluded")):
        rows[index].update(status="labeled", label=label, geometry_status="renderable",
                           source_patch_ncc=.9, patch_baseline_prediction="close")
    corpus = dict(application="camera_projection_transfer", records=rows, prompt="frozen",
                  source_manifest_sha256="source", partitions={n:"test" for n in names},
                  track_counts={n:1 for n in names}, planned_presentations=120)
    development = dict(prompt="frozen", source_manifest_sha256="source",
                       partitions={f"dev{i}":"train" for i in range(10)},
                       records=[dict(id="train1", partition="train", status="labeled", label="far")])
    base = [dict(id=rows[i]["id"], sequence=rows[i]["sequence"], target_for_scoring_only=rows[i]["label"],
                 parsed="far", raw_response="far", processed_pixel_sha256="pixels", image_tokens=609)
            for i in (0,6)]
    adapted = copy.deepcopy(base)
    adapted[0].update(parsed="uncertain", raw_response="uncertain")
    report = dict(status="complete", partition="test", application="camera_projection_transfer",
                  planned_examples=120, labeled_examples=2, results=dict(base=base, adapted=adapted))
    return report, corpus, development


def test_missing_classes_and_videos_remain_explicit():
    result = analyze(*fixture())
    assert result["roster"]["planned_presentations"] == 120
    assert result["roster"]["labeled_presentations"] == 2
    assert result["arms"]["base"]["all"]["accuracy_all_examples"] == .5
    adapted = result["arms"]["adapted"]
    assert adapted["all"]["accuracy_all_examples"] == 0
    assert adapted["all"]["uncertain_examples"] == 1
    assert adapted["all"]["balanced_accuracy"] is None
    assert len(adapted["videos_without_labeled_examples"]) == 18
    assert result["paired_video_uncertainty"]["video_count"] == 2
    assert result["arms"]["source_patch_ncc"]["all"]["false_close_predictions"] == 2
    assert result["arms"]["source_patch_ncc"]["all"]["close_precision"] == 0


@pytest.mark.parametrize("change",["pixel","target","parser","baseline","roster","counts"])
def test_corrupted_accuracy_comparison_rejected(change):
    report, corpus, dev = fixture()
    if change == "pixel":
        report["results"]["adapted"][0]["processed_pixel_sha256"] = "different"
    elif change == "target":
        report["results"]["base"][0]["target_for_scoring_only"] = "close"
    elif change == "parser":
        report["results"]["base"][0]["parsed"] = "close"
    elif change == "baseline":
        corpus["records"][0]["patch_baseline_prediction"] = "far"
    elif change == "roster":
        corpus["records"].pop()
    else:
        report["labeled_examples"] = 3
    with pytest.raises(ValueError):
        analyze(report,corpus,dev)
