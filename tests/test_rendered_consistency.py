import pytest

from paper.probe_rendered_consistency import tracks_for


def case():
    return {"occluder": False, "rows": [
        {"frame": t, "slot": s, "accepted": t != 7, "predicted_xy": [s, t]}
        for t in range(1, 24) for s in range(12)]}


def test_observed_arm_does_not_need_reference_fields():
    data = case()
    tracks = tracks_for(data, "observed")
    assert len(tracks) == 12 and all(len(t) == 24 for t in tracks)
    assert tracks[3][6] == [3, 6]
    assert tracks[3][7] is None
    for row in data["rows"]:
        row.update(reference_xy=[-100, -100], visible=False, error_pixels=999)
    assert tracks_for(data, "observed") == tracks
    assert all(t[1:] == [None]*23 for t in tracks_for(data, "exact_visible"))


def test_roster_cannot_silently_drop_or_duplicate_observations():
    data = case()
    data["rows"].pop()
    with pytest.raises(ValueError, match="all planned"):
        tracks_for(data, "observed")
    data["rows"].append(data["rows"][0])
    with pytest.raises(ValueError, match="duplicate"):
        tracks_for(data, "observed")
