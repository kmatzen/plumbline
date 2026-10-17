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


def test_availability_breakdowns_preserve_every_slot():
    report, corpus, dev = fixture()
    corpus["records"][1].update(status="borderline_unlabeled", geometry_status="renderable")
    result = analyze(report, corpus, dev)
    assert result["roster"]["statuses"] == dict(labeled=2, borderline_unlabeled=1, unavailable=117)
    assert result["roster"]["unavailable_reasons"] == dict(missing_prefix_observations=117)
    for field in ("availability_by_video", "availability_by_candidate_method", "availability_by_query_lag"):
        groups = result[field].values()
        assert sum(g["planned_presentations"] for g in groups) == 120
        assert sum(g["labeled_presentations"] for g in groups) == 2
    assert result["availability_by_video"]["video2"]["label_coverage"] == 0
    assert result["availability_by_candidate_method"]["da3"]["labeled_presentations"] == 0
    assert result["availability_by_query_lag"]["60"]["planned_presentations"] == 40
    assert result["arms"]["adapted"]["by_query_lag"]["60"]["accuracy_all_examples"] is None
    assert result["arms"]["adapted"]["by_query_lag"]["15"]["uncertain_examples"] == 1


def test_no_labeled_presentations_is_not_perfect_accuracy():
    report, corpus, dev = fixture()
    for row in corpus["records"]:
        row.update(status="unavailable", label=None, geometry_status="missing_prefix_observations")
    report.update(labeled_examples=0, results=dict(base=[], adapted=[]))
    result = analyze(report, corpus, dev)
    assert result["roster"]["label_coverage"] == 0
    assert result["paired_video_uncertainty"] is None
    for arm in result["arms"].values():
        assert arm["all"]["accuracy_all_examples"] is None
        assert arm["macro_accuracy_on_videos_with_labels"] is None
        assert len(arm["videos_without_labeled_examples"]) == 20
        assert arm["answered_fraction_of_planned"] == 0


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
