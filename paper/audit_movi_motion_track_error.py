"""Separate tracking error from heldout epipolar ambiguity using frozen fits."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from paper.audit_movi_oracle_motion_control import unique_points
from paper.decode_movi_geometry_pilot import digest
from paper.probe_movi_rigid_support import sampson_distance


def residuals(matrix, source, predicted, reference):
    f = np.asarray(matrix)
    source = np.asarray(source, float)[None]
    predicted = np.asarray(predicted, float)[None]
    reference = np.asarray(reference, float)[None]
    return dict(tracking_error_pixels=float(np.linalg.norm(predicted-reference)),
                predicted_sampson_pixels=float(sampson_distance(f, source, predicted)[0]),
                oracle_sampson_pixels=float(sampson_distance(f, source, reference)[0]))


def summary(rows):
    checked = [r for r in rows if r['status'] == 'compared']
    return dict(planned=len(rows), compared=len(checked),
                statuses=dict(Counter(r['status'] for r in rows)),
                by_reference={label: dict(Counter(
                    r['predicted_compatibility']+' -> '+r['oracle_compatibility']
                    for r in checked if r['reference'] == label))
                    for label in ('static_background', 'moving_foreground', 'other_supported_foreground')})


def run(camera_root, geometry_root, verifier_root):
    paths = dict(motion=camera_root/'marker-motion-audit-v1.json',
                 background=camera_root/'background-marker-audit-v1.json',
                 temporal=verifier_root/'manifest.json',
                 support=verifier_root/'crossfit-support-v1.json',
                 baseline=verifier_root/'rigid-support-v1.json',
                 reference_manifest=geometry_root/'temporal-audit-v1/manifest.json')
    data = {k: json.loads(p.read_text()) for k, p in paths.items()}
    if (data['support']['source_sha256']['baseline'] != digest(paths['baseline'])
            or data['support']['source_sha256']['temporal'] != digest(paths['temporal'])
            or data['motion']['temporal_manifest_sha256'] != digest(paths['reference_manifest'])
            or any(data['baseline']['source_sha256'][k] != digest(paths[k])
                   for k in ('motion', 'background', 'temporal'))):
        raise ValueError('source provenance mismatch')
    motion = unique_points(data['motion']['records'])
    bg = {r['id']: r for r in data['background']['records']}
    predictions = {(r['sequence'], r['frame'], r['track']): r
                   for r in data['temporal']['records'] if r['method'] == 'dage'}
    fits = {(r['sequence'], r['frame']): r for r in data['support']['fits']}
    if len(predictions) != 768 or set(predictions) != set(motion):
        raise ValueError('full roster required')
    oracle = {}
    array_hashes = {}
    for scene in data['reference_manifest']['scenes']:
        path = paths['reference_manifest'].parent/scene['records_file']
        sha = digest(path)
        if sha != scene['records_sha256'] or sha != data['motion']['source_sha256'][scene['name']]['temporal']:
            raise ValueError('reference array changed')
        array_hashes[scene['name']] = sha
        with np.load(path, allow_pickle=False) as a:
            oracle[scene['name']] = {tuple(xy-.5): uv-.5
                                    for xy, uv in zip(a['source_xy'], a['projected_xy'])}
    rows = []
    for key, p in predictions.items():
        m, b = motion[key], bg[p['id']]
        ref, label = None, 'unresolved'
        if b.get('reference_depth_mask_not_rejected'):
            ref, label = b['reference_xy'], 'static_background'
        elif m.get('visibility_evidence') == 'supported':
            ref = oracle[key[0]][tuple(m['source_xy'])][key[1]]
            label = ('moving_foreground' if m['counterfactual_static_error_pixels'] >= 8
                     else 'other_supported_foreground')
        row = dict(sequence=key[0], frame=key[1], track=key[2], reference=label,
                   status='reference_unresolved', predicted_visible=p['predicted_visible'])
        rows.append(row)
        if ref is None:
            continue
        fold = 2*((key[2]//8)%2)+(key[2]%2)
        fit = fits[key[:2]]['fits'][fold]
        if key[2] not in fit['heldout_slots'] or key[2] in fit['training_slots']:
            raise ValueError('own-point fitting')
        if fit['fit']['status'] != 'estimated':
            row['status'] = 'matrix_unavailable'
            continue
        row.update(residuals(fit['fit']['fundamental_matrix'], m['source_xy'], p['predicted_xy'], ref))
        row.update(status='compared', source_xy=m['source_xy'], predicted_xy=p['predicted_xy'],
                   reference_xy=np.asarray(ref).tolist(),
                   predicted_compatibility='compatible' if row['predicted_sampson_pixels'] <= 1 else 'incompatible',
                   oracle_compatibility='compatible' if row['oracle_sampson_pixels'] <= 1 else 'incompatible')
    return dict(scope='development correspondence intervention with frozen image-fitted matrices; not a new evaluator, staticness truth, or DNF',
                protocol='all 768 unique point/frame slots; substitute only the heldout target coordinate with its reference projection; matrix, source coordinate and 1px threshold unchanged; retain predictor-invisible cases separately',
                limitations='matrices still contain errors from other tracks and can be degenerate; oracle reference visibility is diagnostic; hidden predictions are compared for mechanism diagnosis, not counted as deployed answers; no retraining or heldout-scene claim',
                source_sha256={k: digest(p) for k, p in paths.items()}, reference_array_sha256=array_hashes,
                script_sha256=digest(Path(__file__)),
                results=dict(all=summary(rows), predicted_visible=summary([r for r in rows if r['predicted_visible']]),
                             predicted_invisible=summary([r for r in rows if not r['predicted_visible']])), records=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('camera', 'geometry', 'verifier', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    result = run(args.camera, args.geometry, args.verifier)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result['results'], indent=2))
