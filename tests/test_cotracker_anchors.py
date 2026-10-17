import numpy as np
import pytest
from paper.analyze_cotracker_anchors import observations
from paper.run_cotracker_anchors import frozen_queries


def test_queries_preserve_slot_ids_without_using_later_tracking_success():
    baseline = {"tracks": [[{"xy": [5, 6]}, {"xy": None}], [{"xy": None}], [{"xy": [7, 8]}]]}
    slots, queries = frozen_queries(baseline)
    assert slots == [0, 2]
    assert np.array_equal(queries, [[0, 5, 6], [0, 7, 8]])
    with pytest.raises(ValueError):
        frozen_queries({"tracks": [[{"xy": None}]]})


def test_full_video_early_coordinates_cannot_enter_fitting_observations():
    prefix = np.tile([5.0, 6.0], (11, 1))
    full = np.tile([7.0, 8.0], (20, 1))
    full[:11] = [30, 35]
    masks = [np.zeros((40, 40), np.uint8)] * 20
    lk = [{"xy": [5, 6]}] * 20
    rows = observations(prefix, full, np.ones(11, bool), np.ones(20, bool), masks, lk)
    assert all(r["xy"] == [5, 6] for r in rows[:11])
    assert all(r["xy"] == [7, 8] for r in rows[11:])


def test_invisible_and_outside_points_stay_unavailable():
    prefix = np.tile([5.0, 6.0], (11, 1))
    full = np.tile([7.0, 8.0], (20, 1))
    full[-1] = [-1, 8]
    visible = np.ones(20, bool)
    visible[12] = False
    rows = observations(
        prefix,
        full,
        np.ones(11, bool),
        visible,
        [np.zeros((40, 40), np.uint8)] * 20,
        [{"xy": None}] * 20,
    )
    assert rows[12]["xy"] is None
    assert rows[-1]["xy"] is None
    assert rows[-1]["raw_xy"] == [-1, 8]
