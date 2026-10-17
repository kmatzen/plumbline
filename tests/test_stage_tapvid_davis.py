import io
import pickle

import numpy as np
import pytest

from paper.stage_tapvid_davis import ArrayUnpickler, assign_splits, validate_sequence


def example():
    return {"video": np.zeros((3, 4, 5, 3), dtype=np.uint8),
            "points": np.zeros((2, 3, 2), dtype=np.float32),
            "occluded": np.zeros((2, 3), dtype=bool)}


def test_restricted_reader_roundtrips_arrays_and_rejects_code_globals():
    item = example()
    restored = ArrayUnpickler(io.BytesIO(pickle.dumps(item, protocol=4))).load()
    for name in item:
        np.testing.assert_array_equal(item[name], restored[name])
    with pytest.raises(pickle.UnpicklingError):
        ArrayUnpickler(io.BytesIO(pickle.dumps(eval))).load()


def test_split_is_sequence_disjoint_and_order_independent():
    names = [f"video-{i}" for i in range(30)]
    splits = assign_splits(names)
    assert splits == assign_splits(names[::-1])
    assert list(splits.values()).count("development") == 10
    assert list(splits.values()).count("held_out") == 20


def test_annotations_are_preserved_not_silently_filtered():
    item = example()
    item["occluded"][0] = True
    item["points"][1, 0, 0] = 1.01
    result = validate_sequence("test-clip", item)
    assert result["all_occluded_tracks"] == 1
    assert result["visible_outside_unit_square"] == 1
    assert result["visible_observations"] == 3
    with pytest.raises(ValueError, match="unsafe"):
        validate_sequence("../bad", item)
    item["occluded"] = item["occluded"].astype(np.uint8)
    with pytest.raises(ValueError, match="boolean"):
        validate_sequence("test-clip", item)
