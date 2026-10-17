import numpy as np
import pytest

from paper.evaluate_movi_cameras import evaluate, reference_poses


def test_reference_axes_and_reject_invalid_quaternions():
    q = np.tile([1., 0, 0, 0], (24, 1))
    p = np.zeros((24, 3))
    poses = reference_poses(p, q)
    np.testing.assert_array_equal(poses[0, :3, :3], np.diag([1, -1, -1]))
    np.testing.assert_allclose(reference_poses(p, -q), poses)
    with pytest.raises(ValueError):
        reference_poses(p, q*2)


def test_static_alignment_unresolved_but_rotation_available():
    poses = np.tile(np.eye(4), (24, 1, 1))
    result = evaluate(poses, poses)
    assert result['alignment_status'] == 'unresolved'
    assert result['adjacent_pairs'] == 23
    assert result['adjacent_rotation_rmse_degrees'] == 0
    with pytest.raises(ValueError):
        evaluate(poses[:-1], poses)


def test_similarity_invariance():
    poses = np.tile(np.eye(4), (24, 1, 1))
    t = np.linspace(0, 2, 24)
    poses[:, :3, 3] = np.stack([t, t*t, np.sin(t)], axis=1)
    changed = poses.copy()
    rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
    changed[:, :3, :3] = rotation
    changed[:, :3, 3] = 3.5*poses[:, :3, 3] @ rotation.T + [17, -8, 4]
    result = evaluate(changed, poses)
    assert result['alignment_status'] == 'complete'
    assert result['ate_rmse_scene_units'] < 1e-10
    assert result['adjacent_translation_rmse_scene_units'] < 1e-10


def test_rotation_roundoff_does_not_hide_motion():
    from scipy.spatial.transform import Rotation
    poses = np.tile(np.eye(4), (24, 1, 1))
    changed = poses.copy()
    changed[:, :3, :3] = Rotation.from_euler('z', np.arange(24)[:, None]*0.01, degrees=True).as_matrix()*1.000001
    result = evaluate(changed, poses)
    assert result['adjacent_rotation_rmse_degrees'] == pytest.approx(0.01, abs=1e-6)
