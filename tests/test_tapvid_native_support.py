import numpy as np

from paper.audit_tapvid_native_support import accepted, audit_track, summarize
from paper.probe_tapvid_identity import decisions


def row(frame, xy):
    return dict(frame=frame, xy=xy, forward_backward_pixels=.5, source_patch_ncc=.9)


def test_native_box_units_and_whole_track_error():
    track = dict(track=0, query_frame=0, records=[row(1, [401, 200]), row(2, [405, 200])])
    result = audit_track(track, np.full((3, 2), .5), np.zeros(3, bool), 800, 400)
    four = result["native_box_tolerances"]["4"]
    assert four["accepted_in_native_box"] == 1
    assert four["all_accepted_in_native_box"] is False
    assert result["native_box_tolerances"]["8"]["all_accepted_in_native_box"] is True
    assert summarize([result])["tolerances"]["4"]["whole_track_fraction_among_two_or_more"] == 0


def test_zero_and_one_observation_do_not_become_clean_geometry_tracks():
    points = np.full((2, 2), .5)
    empty = audit_track(dict(track=0, query_frame=0, records=[]), points, np.zeros(2, bool), 800, 400)
    one = audit_track(dict(track=1, query_frame=0, records=[row(1, [400, 200])]), points, np.ones(2, bool), 800, 400)
    result = summarize([empty, one])
    assert result["zero_accepted_tracks"] == 1 and result["one_accepted_tracks"] == 1
    assert result["accepted_occluded"] == 1
    assert result["tolerances"]["8"]["whole_track_fraction_among_two_or_more"] is None
    assert result["tolerances"]["8"]["accepted_in_box_fraction"] == 0


def test_decision_exactly_matches_frozen_rule():
    for xy in (None, [2, 3]):
        for fb in (None, 0, 1, 1.001):
            for ncc in (None, .799, .8, 1):
                r = dict(xy=xy, forward_backward_pixels=fb, source_patch_ncc=ncc)
                assert accepted(r) == decisions(r)["both"]
