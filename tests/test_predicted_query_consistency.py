import copy
import numpy as np
import pytest

from paper.probe_predicted_query_consistency import evaluate


def test_query_change_affects_comparison_not_fitted_prediction_bounds():
    poses = np.repeat(np.eye(4)[None], 90, axis=0)
    poses[:, 0, 3] = np.arange(90) * .01
    track = [{"xy": [320.1-535.4*i*.01/3, 247.6]} for i in range(90)]
    # Preserve twelve slots but only evaluate one; the others are explicitly missing.
    tracks = [track] + [[{"xy": None} for _ in range(90)] for _ in range(11)]
    before = evaluate(poses, tracks)
    changed = copy.deepcopy(tracks)
    changed[0][89]["xy"] = [1e6, 1e6]
    after = evaluate(poses, changed)
    for a, b in zip(before, after, strict=True):
        assert a["slots"][0]["prediction_bounds"] == b["slots"][0]["prediction_bounds"]
        assert a["slots"][0]["query_comparison"]["status"] == "compatible"
        assert b["slots"][0]["query_comparison"]["status"] == "query_contradiction"
        assert a["counts"]["insufficient_fitting_observations"] == 11
    with pytest.raises(ValueError, match="retain"):
        evaluate(poses, tracks[:1])
