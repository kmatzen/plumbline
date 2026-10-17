import numpy as np
from paper.audit_ore_depth_support import (
    camera_depths,
    choose_two_grid_minimum,
    positive_depth_interval,
)


def test_depth_check_detects_later_negative_depth():
    poses = np.tile(np.eye(4), (3, 1, 1))
    poses[1, 2, 3] = -20
    assert np.allclose(camera_depths(poses, 0, np.array([0, 0, 5])), [5, -15, 5])


def test_depth_check_uses_anchor_frame_not_world_origin():
    poses = np.tile(np.eye(4), (3, 1, 1))
    poses[:, 2, 3] = [-1, -3, -6]
    assert np.allclose(camera_depths(poses, 1, np.array([0, 0, 5])), [7, 5, 2])


def test_released_tie_policy_is_preserved():
    assert choose_two_grid_minimum([np.array([1, 0, 0]), np.array([0, 0])]) == (1, 0)
    assert choose_two_grid_minimum([np.array([1, 0, 0]), np.array([1, 2])]) == (0, 1)
    assert choose_two_grid_minimum([np.array([np.inf]), np.array([np.inf])]) == (1, 0)


def test_continuous_positive_interval_distinguishes_grid_from_geometry():
    result = positive_depth_interval([1, -1], [-201, 300])
    assert result == {"feasible": True, "lower_exclusive": 201.0, "upper_exclusive": 300.0}
    assert not positive_depth_interval([1, -1], [-2, 1])["feasible"]
    assert not positive_depth_interval([0], [0])["feasible"]
    assert positive_depth_interval([0], [1]) == {
        "feasible": True,
        "lower_exclusive": 0.0,
        "upper_exclusive": None,
    }
