import pytest

from paper.analyze_tapvid_external import analyze, paired_video_interval


def test_paired_video_interval_identical_or_constant_effect():
    base = {"a":.2,"b":.8}
    zero = paired_video_interval(base,base)
    assert zero["macro_accuracy_difference"] == 0
    assert zero["percentile_interval_95"] == [0,0]
    shifted = paired_video_interval(base,{"a":.3,"b":.9})
    assert shifted["macro_accuracy_difference"] == pytest.approx(.1)
    assert shifted["percentile_interval_95"] == pytest.approx([.1,.1])
    assert shifted == paired_video_interval(base,{"a":.3,"b":.9})


def test_unmatched_or_missing_videos_rejected():
    with pytest.raises(ValueError,match="matching"):
        paired_video_interval({}, {})
    with pytest.raises(ValueError,match="matching"):
        paired_video_interval({"a":1},{"b":1})


def test_complete_analysis_retains_unscored_roster_and_training_only_prior():
    names = [f"test{i}" for i in range(20)]
    corpus = dict(partitions=dict.fromkeys(names,"test"),prompt="same",source_manifest_sha256="same",
                  records=[dict(id=n,sequence=n,partition="test",status="labeled",label="close",
                                track=0,frame=5,candidate_xy=[1,2]) for n in names])
    corpus["records"].append(dict(id="margin",sequence=names[0],partition="test",status="borderline_unlabeled"))
    development = dict(partitions={f"dev{i}":"train" for i in range(10)},prompt="same",
                       source_manifest_sha256="same",records=[dict(id="train",partition="train",status="labeled",label="far")])
    report = dict(status="complete",partition="test",results={})
    for arm, answer in (("base","close"),("adapted","uncertain")):
        report["results"][arm] = [dict(id=n,sequence=n,target_for_scoring_only="close",
                                      raw_response=answer,parsed=answer,processed_pixel_sha256="same") for n in names]
    tracker = {"sequences": {n:{"tracks":[dict(track=0,records=[dict(frame=5,xy=[1,2],
                 forward_backward_pixels=0,source_patch_ncc=1)])]} for n in names}}
    result = analyze(report,corpus,development,tracker)
    assert result["training_majority_label"] == "far"
    assert result["roster"]["planned_examples"] == 21
    assert result["arms"]["base"]["answered_fraction_of_planned"] == 20/21
    assert result["arms"]["adapted"]["answered_fraction_of_planned"] == 0
    assert result["paired_video_uncertainty"]["percentile_interval_95"] == [-1,-1]
    assert result["arms"]["adapted"]["all"]["uncertain_examples"] == 20
