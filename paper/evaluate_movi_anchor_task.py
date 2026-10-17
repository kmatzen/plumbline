"""Reference-only static-anchor task outcome, separate from RGB evaluator inputs."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from paper.decode_movi_geometry_pilot import digest
from paper.evaluate_movi_cameras import reference_poses
from plumbline.conventions import assert_valid_extrinsics


def register_prefix(predicted, reference):
    """Fix orientation/origin at frame zero; estimate scale from frames 0..9."""
    if predicted.shape != (24, 4, 4) or reference.shape != predicted.shape:
        raise ValueError('all 24 poses required')
    assert_valid_extrinsics(predicted)
    assert_valid_extrinsics(reference)
    predicted = predicted.astype(float).copy()
    reference = reference.astype(float).copy()
    for poses in (predicted, reference):
        poses[:, :3, :3] = Rotation.from_matrix(poses[:, :3, :3]).as_matrix()
    rotation = reference[0, :3, :3] @ predicted[0, :3, :3].T
    p = (predicted[:10, :3, 3]-predicted[0, :3, 3]) @ rotation.T
    q = reference[:10, :3, 3]-reference[0, :3, 3]
    energy = float(np.sum(p*p))
    if energy <= np.finfo(float).tiny:
        return None, dict(status='unresolved', reason='no predicted prefix translation to identify scale')
    scale = float(np.sum(p*q)/energy)
    if not np.isfinite(scale) or scale <= 0:
        return None, dict(status='unresolved', reason='positive prefix scale unavailable', scale=scale if np.isfinite(scale) else None)
    translation = reference[0, :3, 3]-scale*(rotation @ predicted[0, :3, 3])
    aligned = predicted.copy()
    aligned[:, :3, :3] = rotation @ predicted[:, :3, :3]
    aligned[:, :3, 3] = scale*predicted[:, :3, 3] @ rotation.T+translation
    return aligned, dict(status='registered', scale=scale, rotation=rotation.tolist(), translation=translation.tolist(),
        prefix_center_rmse_scene_units=float(np.sqrt(np.mean(np.sum((scale*p-q)**2, axis=1)))))


def measure(world, aligned, calibration, reference_xy, reference_eligible):
    if calibration.shape != (24, 3, 3) or not np.isfinite(calibration).all():
        raise ValueError('invalid intrinsics')
    local = np.einsum('tji,tnj->tni', aligned[:, :3, :3], world[None]-aligned[:, None, :3, 3])
    homogeneous = np.einsum('tij,tnj->tni', calibration, local)
    defined = (local[..., 2] > 0) & np.isfinite(homogeneous).all(axis=-1) & (np.abs(homogeneous[..., 2]) > 1e-12)
    xy = np.full(local.shape[:2]+(2,), np.nan)
    xy[defined] = homogeneous[defined, :2]/homogeneous[defined, 2, None]*256-.5
    error = np.linalg.norm(xy.transpose(1, 0, 2)-reference_xy, axis=-1)
    positive = defined.T
    inside = positive & ((xy >= 0) & (xy <= 255)).all(axis=-1).T
    later = np.zeros(24, bool); later[10:] = True
    eligible = reference_eligible & later[None]
    measured = eligible & positive
    values = error[measured]
    summary = dict(planned_later_point_frames=int(len(world)*14),
        reference_eligible=int(eligible.sum()), unavailable_reference=int((later[None] & ~reference_eligible).sum()),
        nonpositive_or_undefined_candidate=int((eligible & ~positive).sum()),
        outside_candidate_image=int((measured & ~inside).sum()),
        measured=int(measured.sum()),
        pixel_error_quantiles_0_50_95_100=np.quantile(values, [0, .5, .95, 1]).tolist() if len(values) else None,
        mean_pixel_error=float(values.mean()) if len(values) else None)
    # Positive-depth projections outside the image remain in the error summary.
    return dict(candidate_xy=xy.transpose(1, 0, 2), error_pixels=error,
                candidate_positive=positive, candidate_inside=inside, reference_eligible=eligible), summary


def run(decoded, camera_root, background_root, output):
    dm_path, bm_path = decoded/'manifest.json', background_root/'manifest.json'
    dm, bm = [json.loads(p.read_text()) for p in (dm_path, bm_path)]
    if (dm['status'] != 'complete' or bm['status'] != 'complete'
            or bm['decoded_manifest_sha256'] != digest(dm_path)
            or [s['name'] for s in dm['scenes']] != ['3835', '5397', '4385', '4970']):
        raise ValueError('complete fixed pilot required')
    output.mkdir(parents=True, exist_ok=False)
    result = dict(status='running', scope='reference-based virtual-anchor development outcome, not reference-free evaluator or human DNF',
        protocol='all eight methods/scenes and all 195 fixed background points; first-frame orientation/origin registration and least-squares positive scale on frames 0..9 only; score frames 10..23; predicted intrinsics primary and reference-intrinsics ablation; retain off-image/behind-camera and unresolved registration',
        reference_policy='background depth/mask non-rejection; no independent visibility certification; no marker fitting or later reference poses in registration',
        timing='new development outcome defined after conventional pose and marker audits; no held-out claim',
        claim_policy='continuous application outcome; no functional DNF threshold or new-method advantage asserted',
        source_sha256=dict(decoded=digest(dm_path), background=digest(bm_path)),
        script_sha256=digest(Path(__file__)), attempts=[])
    try:
        for scene in dm['scenes']:
            name = scene['name']
            bscene = next(s for s in bm['scenes'] if s['name'] == name)
            ref_path, bg_path = decoded/scene['reference_file'], background_root/bscene['file']
            if digest(ref_path) != scene['reference_sha256'] or digest(bg_path) != bscene['sha256']:
                raise ValueError('reference hashes changed')
            with np.load(ref_path, allow_pickle=False) as a:
                reference = reference_poses(a['camera_positions'], a['camera_quaternions'])
                f = float(a['camera_focal_length'][0]/a['camera_sensor_width'][0])
            with np.load(bg_path, allow_pickle=False) as a:
                world, reference_xy = a['world_point'], a['projected_xy']
                eligible = a['in_image'] & ~a['depth_rejected'] & ~a['mask_rejected']
            reference_k = np.tile([[f, 0, .5], [0, f, .5], [0, 0, 1]], (24, 1, 1))
            for method in ('dage', 'da3'):
                directory = camera_root/f'{name}-{method}'
                path = directory/'manifest.json'
                saved = json.loads(path.read_text())
                row = dict(sequence=name, method=method, camera_manifest_sha256=digest(path),
                           reference_sha256=digest(ref_path), background_sha256=digest(bg_path))
                result['attempts'].append(row)
                if saved['status'] == 'failed':
                    row['status'] = 'camera_run_failed'
                    continue
                prediction_path = directory/'prediction.npz'
                if (saved['status'] != 'complete' or saved['sequence'] != name or saved['method'] != method
                        or saved['input_manifest_sha256'] != digest(dm_path)
                        or digest(prediction_path) != saved['prediction_sha256']):
                    raise ValueError('camera input or prediction changed')
                with np.load(prediction_path, allow_pickle=False) as a:
                    predicted, calibration = a['extrinsics'], a['intrinsics_normalized']
                aligned, registration = register_prefix(predicted, reference)
                row.update(registration=registration, prediction_sha256=digest(prediction_path))
                if aligned is None:
                    row['status'] = 'registration_unresolved'
                    continue
                row.update(status='evaluated', calibration_arms={})
                for arm, intrinsics in (('predicted_intrinsics', calibration), ('reference_intrinsics', reference_k)):
                    arrays, summary = measure(world, aligned, intrinsics, reference_xy, eligible)
                    target = output/f'{name}-{method}-{arm}.npz'
                    np.savez_compressed(target, **arrays)
                    row['calibration_arms'][arm] = dict(file=target.name, sha256=digest(target), **summary)
        result['status'] = 'complete'
    except Exception as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        (output/'manifest.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('decoded', 'cameras', 'background', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.decoded, args.cameras, args.background, args.output)
