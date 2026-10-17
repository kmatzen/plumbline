"""Separate reference-pose evaluation of every fixed MOVi camera attempt."""

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from paper.decode_movi_geometry_pilot import digest
from plumbline.conventions import assert_valid_extrinsics
from plumbline.metrics import pose


def reference_poses(positions, wxyz):
    if positions.shape != (24, 3) or wxyz.shape != (24, 4):
        raise ValueError('all 24 reference frames required')
    if not np.isfinite(wxyz).all() or not np.allclose(np.linalg.norm(wxyz, axis=1), 1, atol=1e-6, rtol=0):
        raise ValueError('invalid reference quaternion')
    result = np.tile(np.eye(4), (24, 1, 1))
    # Blender camera front -Z/up +Y -> OpenCV front +Z/down +Y.
    result[:, :3, :3] = Rotation.from_quat(wxyz[:, [1, 2, 3, 0]]).as_matrix() @ np.diag([1, -1, -1])
    result[:, :3, 3] = positions
    assert_valid_extrinsics(result)
    return result


def evaluate(predicted, reference):
    if predicted.shape != (24, 4, 4) or reference.shape != predicted.shape:
        raise ValueError('all 24 frames required without truncation')
    assert_valid_extrinsics(predicted)
    assert_valid_extrinsics(reference)
    # Model matrices have finite-precision orthogonality error. Normalize once
    # before angular metrics; trace/arccos otherwise clips small errors to zero.
    predicted = predicted.copy()
    reference = reference.copy()
    corrections = []
    for poses in (predicted, reference):
        raw = poses[:, :3, :3].copy()
        poses[:, :3, :3] = Rotation.from_matrix(raw).as_matrix()
        corrections.append(float(np.max(np.abs(raw-poses[:, :3, :3]))))
    rp = np.swapaxes(predicted[:-1, :3, :3], 1, 2) @ predicted[1:, :3, :3]
    rr = np.swapaxes(reference[:-1, :3, :3], 1, 2) @ reference[1:, :3, :3]
    angles = pose.rotation_error_degrees(rp, rr)
    result = dict(frames=24, adjacent_pairs=23, rotation_projection_max_abs_correction=corrections,
                  adjacent_rotation_error_degrees=angles.tolist(),
                  adjacent_rotation_rmse_degrees=float(np.sqrt(np.mean(angles**2))))
    # A line or point does not identify a unique position-based world rotation.
    ranks = [int(np.linalg.matrix_rank(p[:, :3, 3] - p[:, :3, 3].mean(axis=0)))
             for p in (predicted, reference)]
    result['centered_position_ranks'] = ranks
    if min(ranks) < 2:
        result.update(alignment_status='unresolved', alignment_reason='position trajectory rank below two')
        return result
    try:
        ate = pose.trajectory_ate_rmse_sim3(predicted, reference)
        trans, rot = pose.trajectory_rpe_rmse_sim3(predicted, reference)
        if not np.isfinite([ate, trans, rot]).all():
            raise ValueError('nonfinite aligned metrics')
        result.update(alignment_status='complete', ate_rmse_scene_units=ate,
                      adjacent_translation_rmse_scene_units=trans,
                      aligned_adjacent_rotation_rmse_degrees=rot)
    except (ValueError, ArithmeticError, np.linalg.LinAlgError) as error:
        result.update(alignment_status='unresolved', alignment_reason=str(error))
    return result


def run(root, runs, protocol_path):
    protocol = json.loads(protocol_path.read_text())
    manifest_path = root/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    batch_path = runs/'batch.json'
    batch = json.loads(batch_path.read_text())
    roster = [(s, m) for s in protocol['sequences'] for m in protocol['methods']]
    if (digest(manifest_path) != protocol['input_manifest_sha256']
            or batch['protocol_sha256'] != digest(protocol_path)
            or batch['status'] != 'finished_all_attempts'
            or [(a['sequence'], a['method']) for a in batch['attempts']] != roster
            or [s['name'] for s in manifest['scenes']] != protocol['sequences']):
        raise ValueError('fixed roster or provenance mismatch')
    rows = []
    for attempt in batch['attempts']:
        row = dict(sequence=attempt['sequence'], method=attempt['method'], process_status=attempt['status'])
        rows.append(row)
        directory = (runs/attempt['output']).resolve()
        if not directory.is_relative_to(runs.resolve()):
            raise ValueError('attempt outside run directory')
        saved = directory/'manifest.json'
        row['manifest_sha256'] = digest(saved)
        prediction = json.loads(saved.read_text())
        if attempt['status'] != 'process_complete':
            row['evaluation_status'] = 'process_failed'
            continue
        scene = next(s for s in manifest['scenes'] if s['name'] == attempt['sequence'])
        reference_path = (root/scene['reference_file']).resolve()
        prediction_path = directory/'prediction.npz'
        if (not reference_path.is_relative_to(root.resolve())
                or digest(reference_path) != scene['reference_sha256']
                or prediction['status'] != 'complete'
                or prediction['sequence'] != row['sequence'] or prediction['method'] != row['method']
                or prediction['input_manifest_sha256'] != digest(manifest_path)
                or prediction['decoded_rgb_sha256'] != scene['decoded_rgb_sha256']
                or prediction['rgb_file_sha256'] != scene['rgb_sha256']
                or digest(prediction_path) != prediction['prediction_sha256']):
            raise ValueError('reference or prediction provenance mismatch')
        with np.load(reference_path, allow_pickle=False) as arrays:
            reference = reference_poses(arrays['camera_positions'], arrays['camera_quaternions'])
        with np.load(prediction_path, allow_pickle=False) as arrays:
            metrics = evaluate(arrays['extrinsics'].astype(float), reference)
        row.update(evaluation_status='evaluated', reference_sha256=digest(reference_path),
                   prediction_sha256=digest(prediction_path), metrics=metrics)
    return dict(scope='four-scene development reference validation; not DNF or evaluator validation',
                policy='all eight attempts, all 24 frames, 23 adjacent pairs; Sim3 position alignment; units are dataset scene units',
                reference_convention='wxyz camera-to-world, right multiplied by diag(1,-1,-1,1)',
                protocol_sha256=digest(protocol_path), batch_sha256=digest(batch_path),
                decoded_manifest_sha256=digest(manifest_path), script_sha256=digest(Path(__file__)),
                metric_source_sha256=digest(Path(inspect.getfile(pose))), attempts=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'runs', 'protocol', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    result = run(args.root, args.runs, args.protocol)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
