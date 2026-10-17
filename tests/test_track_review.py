from paper.build_track_review import select_slots


def test_selection_depends_only_on_first_frame_not_later_survival():
    tracks = [
        [{"xy": None}],
        [{"xy": [1, 2]}, {"xy": None}],
        [{"xy": [3, 4]}, {"xy": [8, 9]}],
        [{"xy": [5, 6]}],
        [{"xy": [7, 8]}],
    ]
    assert select_slots(tracks) == [1, 2, 3]
    tracks[1][1]["xy"] = [100, 200]
    tracks[2][1]["xy"] = None
    assert select_slots(tracks) == [1, 2, 3]
