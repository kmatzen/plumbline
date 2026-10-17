import pytest

from paper.build_tapvid_external_corpus import test_names as select_names


def test_external_roster_excludes_every_development_video():
    held = {f"test{i}": {"split":"held_out"} for i in range(20)}
    staging = {"sequences": {**held, "development": {"split":"development"}}}
    predictions = {"sequences": dict.fromkeys(held)}
    assert select_names(staging, predictions) == sorted(held)
    predictions["sequences"]["development"] = None
    with pytest.raises(ValueError, match="twenty"):
        select_names(staging, predictions)


def test_partial_test_roster_rejected():
    staging = {"sequences": {f"test{i}": {"split":"held_out"} for i in range(20)}}
    with pytest.raises(ValueError, match="twenty"):
        select_names(staging, {"sequences": {}})
