import copy

import pytest

from paper.analyze_camera_transfer_lk import analyze, predict


def observation():
    return dict(xy=[20., 20.], forward_backward_pixels=.5, source_patch_ncc=.9)


@pytest.mark.parametrize("xy,answer", [([22.,20.],"close"),([25.,20.],"uncertain"),([28.,20.],"far")])
def test_fixed_native_pixel_thresholds(xy, answer):
    assert predict(xy, observation()) == (answer, xy[0]-20)


def test_rejected_track_does_not_mean_far_or_occluded():
    for changes in (dict(xy=None), dict(forward_backward_pixels=1.01),
                    dict(source_patch_ncc=.79), dict(source_patch_ncc=None)):
        assert predict([100.,100.], dict(observation(), **changes)) == ("uncertain", None)
    assert predict([100.,100.], None) == ("uncertain", None)


def test_tracker_scoring_metadata_is_not_used():
    a = dict(observation(), scoring_only=dict(visible=False, error_at_256_pixels=900))
    b = dict(observation(), scoring_only=dict(visible=True, error_at_256_pixels=0))
    assert predict([21.,20.], a) == predict([21.,20.], b)


def fixture():
    names = [f"video{i}" for i in range(20)]
    rows = [dict(id=f"{n}:{m}:{lag}", sequence=n, partition="test", model=m, track=0, lag=lag,
                 source_frame=0, source_xy=[20.,20.], frame=lag, candidate_xy=[21.,20.],
                 status="labeled", label="close", geometry_status="renderable")
            for n in names for m in ("dage", "da3") for lag in (15,30,60)]
    corpus = dict(records=rows, source_manifest_sha256="source", partitions={n:"test" for n in names},
                  track_counts={n:1 for n in names}, planned_presentations=120)
    tracker = dict(split="held_out", source_manifest_sha256="source", sequences={n:dict(tracks=[
        dict(track=0, query_frame=0, query_xy=[20.,20.], records=[dict(observation(),frame=f) for f in (15,30,60)])]) for n in names})
    return corpus, tracker


def test_predictions_fixed_before_labels_and_unavailable_retained():
    corpus, tracker = fixture()
    corpus["records"][0].update(status="unavailable", label=None, geometry_status="query_beyond_video")
    a = analyze(corpus, tracker)
    changed = copy.deepcopy(corpus)
    for row in changed["records"][1:]:
        row["label"] = "far"
    b = analyze(changed, tracker)
    assert a["predictions"] == b["predictions"]
    assert a["all_planned_answers"] == dict(uncertain=1, close=119)
    assert a["labeled_metrics"]["accuracy_all_examples"] == 1
    assert b["labeled_metrics"]["accuracy_all_examples"] == 0
    assert a["planned_presentations"] == 120


@pytest.mark.parametrize("change",["source", "query", "duplicate"])
def test_bad_binding_rejected(change):
    corpus, tracker = fixture()
    if change == "source":
        tracker["source_manifest_sha256"] = "changed"
    elif change == "query":
        corpus["records"][0]["source_xy"] = [21.,20.]
    else:
        track = tracker["sequences"]["video0"]["tracks"][0]
        track["records"].append(copy.deepcopy(track["records"][0]))
    with pytest.raises(ValueError):
        analyze(corpus, tracker)
