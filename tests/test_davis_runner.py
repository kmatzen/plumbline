import pytest
from paper.run_davis_dage import select_frames


def inventory(split="train"):
    return {
        "sequences": [
            {"sequence": "s", "split": split, "frames": [{"frame_index": i} for i in range(4)]}
        ]
    }


def test_prefix_selection_preserves_indices_and_does_not_invent_time():
    frames = select_frames(inventory(), "s", 3)
    assert frames == [{"frame_index": 0}, {"frame_index": 1}, {"frame_index": 2}]


def test_validation_cannot_be_used_for_development():
    with pytest.raises(ValueError, match="validation is reserved"):
        select_frames(inventory("val"), "s", 3)


def test_bad_frame_order_is_rejected():
    data = inventory()
    data["sequences"][0]["frames"].reverse()
    with pytest.raises(ValueError, match="ordered"):
        select_frames(data, "s", 3)


@pytest.mark.parametrize("limit", [0, 1, 91])
def test_frame_budget_is_explicit(limit):
    with pytest.raises(ValueError, match="frame limit"):
        select_frames(inventory(), "s", limit)
