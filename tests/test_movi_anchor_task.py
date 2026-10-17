import numpy as np
import json
from pathlib import Path

from paper.evaluate_movi_anchor_task import measure, register_prefix
from paper.decode_movi_geometry_pilot import digest


def trajectory():
    poses = np.tile(np.eye(4), (24, 1, 1))
    poses[:, 0, 3] = np.arange(24)*.02
    return poses


def test_registration_uses_no_later_reference_poses_and_removes_gauge():
    reference = trajectory()
    predicted = reference.copy()
    rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
    predicted[:, :3, :3] = rotation
    predicted[:, :3, 3] = 3.5*reference[:, :3, 3] @ rotation.T+[17, -8, 4]
    aligned, registration = register_prefix(predicted, reference)
    np.testing.assert_allclose(aligned, reference, atol=1e-12)
    reference[10:, :3, 3] += 100
    changed, changed_registration = register_prefix(predicted, reference)
    np.testing.assert_array_equal(changed, aligned)
    assert changed_registration == registration


def test_static_scale_is_unresolved_not_a_perfect_task_score():
    poses = np.tile(np.eye(4), (24, 1, 1))
    aligned, registration = register_prefix(poses, poses)
    assert aligned is None and registration['status'] == 'unresolved'


def test_behind_and_offscreen_projections_remain_accounted():
    poses = np.tile(np.eye(4), (24, 1, 1))
    intrinsics = np.tile([[1., 0, 2.], [0, 1, .5], [0, 0, 1]], (24, 1, 1))
    reference = np.full((2, 24, 2), 127.5)
    records, summary = measure(np.array([[0., 0, 4], [0, 0, -4]]), poses, intrinsics,
                               reference, np.ones((2, 24), bool))
    assert summary['planned_later_point_frames'] == summary['reference_eligible'] == 28
    assert summary['nonpositive_or_undefined_candidate'] == 14
    assert summary['outside_candidate_image'] == summary['measured'] == 14
    assert summary['mean_pixel_error'] == 384
    assert records['reference_eligible'][:, :10].sum() == 0


def test_archived_task_arrays_match_summaries_and_hashes():
    root = Path(__file__).resolve().parents[1]/'paper/results/movi-camera-pilot-v1/anchor-task-v1'
    report = json.loads((root/'manifest.json').read_text())
    assert report['status'] == 'complete' and len(report['attempts']) == 8
    assert report['script_sha256'] == digest(Path(__file__).resolve().parents[1]/'paper/evaluate_movi_anchor_task.py')
    total = 0
    for attempt in report['attempts']:
        assert attempt['status'] == 'evaluated'
        for arm in attempt['calibration_arms'].values():
            assert digest(root/arm['file']) == arm['sha256']
            with np.load(root/arm['file'], allow_pickle=False) as a:
                selected = a['reference_eligible'] & a['candidate_positive']
                values = a['error_pixels'][selected]
                np.testing.assert_allclose(np.quantile(values, [0, .5, .95, 1]), arm['pixel_error_quantiles_0_50_95_100'])
                assert int(selected.sum()) == arm['measured']
                assert arm['reference_eligible']+arm['unavailable_reference'] == arm['planned_later_point_frames']
        total += attempt['calibration_arms']['predicted_intrinsics']['planned_later_point_frames']
    assert total == 5460
