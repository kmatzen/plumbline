"""Score frozen replication evidence without changing inference or thresholds."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from paper.audit_movi_paired_placement import aggregate, preference
from paper.decode_movi_geometry_pilot import digest

ARMS = ('tracking', 'tracking_motion', 'tracking_motion_cycle')


def pixel_label(distance):
    return 'unresolved' if distance is None else 'close' if distance <= 2 else 'far' if distance >= 8 else 'borderline'


def predicted_label(distance):
    label = pixel_label(distance)
    return 'uncertain' if label == 'borderline' else label


def accepts(row, arm):
    return bool(row['predicted_visible'] and
                (arm == 'tracking' or row['motion_predicted'] == 'compatible') and
                (arm != 'tracking_motion_cycle' or row['cycle_predicted'] == 'consistent'))


def placement_summary(rows, arm):
    return dict(planned=len(rows), answers=dict(Counter(r['answers'][arm] for r in rows)),
                reference_counts=dict(Counter(r['reference_label'] for r in rows)),
                confusion={label: dict(Counter(r['answers'][arm] for r in rows if r['reference_label'] == label))
                           for label in ('close', 'far', 'borderline', 'unresolved')})


def checked_arrays(root, entry, filename, checksum):
    path = root/entry[filename]
    if digest(path) != entry[checksum]:
        raise ValueError('reference arrays changed')
    with np.load(path, allow_pickle=False) as a:
        return {key: a[key] for key in a.files}


def run(root):
    paths = dict(evidence=root/'evidence-v1/manifest.json', decoded=root/'decoded-manifest.json',
                 background=root/'background-audit-v1/manifest.json',
                 temporal=root/'temporal-audit-v1/manifest.json', visibility=root/'visibility-audit-v1/manifest.json',
                 protocol=root/'protocol.json')
    data = {k: json.loads(p.read_text()) for k, p in paths.items()}
    sha = digest(paths['decoded'])
    e = data['evidence']
    if (e['status'] != 'complete' or e['input_manifest_sha256'] != sha
            or e['protocol_sha256'] != digest(paths['protocol'])
            or data['background']['status'] != 'complete'
            or data['background']['decoded_manifest_sha256'] != sha
            or data['temporal']['source_manifest_sha256'] != sha
            or data['visibility']['source_sha256']['decoded'] != sha
            or data['visibility']['source_sha256']['temporal'] != digest(paths['temporal'])):
        raise ValueError('reference/inference provenance mismatch')
    names = [s['name'] for s in data['decoded']['scenes']]
    if names != e['sequences'] or any([s['name'] for s in data[k]['scenes']] != names
                                    for k in ('background', 'temporal', 'visibility')):
        raise ValueError('scene roster mismatch')
    points, placements, pairs, geometry_hashes = [], [], [], {}
    for name in names:
        gp = root/f'marker-geometry-{name}-v1.json'
        g = json.loads(gp.read_text())
        es = next(s for s in e['scenes'] if s['sequence'] == name)
        if digest(gp) != es['geometry_sha256']:
            raise ValueError('candidate geometry changed')
        geometry_hashes[name] = digest(gp)
        b, t, v = [checked_arrays(paths[k].parent, next(s for s in data[k]['scenes'] if s['name'] == name), fn, ck)
                   for k, fn, ck in (('background', 'file', 'sha256'), ('temporal', 'records_file', 'records_sha256'),
                                     ('visibility', 'records_file', 'records_sha256'))]
        bg = {int(slot): i for i, slot in enumerate(b['slots'])}
        fg = {tuple(xy-.5): i for i, xy in enumerate(t['source_xy'])}
        candidates = {(r['track'],r['frame'],r['method']): r for r in g['records']}
        if len(candidates) != 384:
            raise ValueError('full candidate roster required')
        scene_rows = [r for r in e['records'] if r['sequence'] == name]
        if {(r['frame'],r['track']) for r in scene_rows} != {(f,i) for f in (12,18,23) for i in range(64)} or len(scene_rows) != 192:
            raise ValueError('full point roster required')
        for r in scene_rows:
            i, frame = r['track'], r['frame']
            reference, motion, xy = 'unresolved', None, None
            if i in bg:
                j = bg[i]
                if not np.array_equal(r['source_xy'],b['source_xy'][j]):
                    raise ValueError('source pixel mismatch')
                if b['in_image'][j,frame] and not b['depth_rejected'][j,frame] and not b['mask_rejected'][j,frame]:
                    reference, motion, xy = 'static_background', 0., b['projected_xy'][j,frame]
            else:
                j = fg[tuple(r['source_xy'])]
                if v['supported'][j,frame]:
                    motion = float(t['counterfactual_static_error_pixels'][j,frame])
                    reference = 'moving_foreground' if motion >= 8 else 'other_supported_foreground'
                    xy = t['projected_xy'][j,frame]-.5
            distance = None if xy is None else float(np.linalg.norm(np.asarray(r['predicted_xy'])-xy))
            point = dict(sequence=name,frame=frame,track=i,reference=reference,
                tracking_error_pixels=distance,tracking_reference=pixel_label(distance),
                counterfactual_motion_pixels=motion,cycle_predicted=r['cycle_predicted'],
                retained={arm: accepts(r,arm) for arm in ARMS})
            points.append(point)
            pair = dict(sequence=name,frame=frame,track=i,retained=point['retained'],
                        estimated_errors={},reference_errors=None,common_renderable=True)
            reference_errors = {}
            for method in ('dage','da3'):
                c = candidates[i,frame,method]
                renderable = c['status'] == 'renderable'
                estimate = float(np.linalg.norm(np.asarray(c['candidate_xy'])-r['predicted_xy'])) if renderable else None
                actual = float(np.linalg.norm(np.asarray(c['candidate_xy'])-xy)) if renderable and xy is not None else None
                placements.append(dict(id=c['id'], sequence=name, method=method, frame=frame, track=i,
                    status=c['status'],reference=reference,reference_error_pixels=actual,estimated_error_pixels=estimate,
                    reference_label=pixel_label(actual), answers={arm:
                        predicted_label(estimate) if accepts(r,arm) and renderable else 'unresolved' for arm in ARMS}))
                pair['common_renderable'] &= renderable
                pair['estimated_errors'][method] = estimate
                reference_errors[method] = actual
            if pair['common_renderable'] and reference == 'static_background':
                pair['reference_errors'] = reference_errors
            pairs.append(pair)
    pair_results = {}
    for name in names:
        scene = [r for r in pairs if r['sequence'] == name]
        common = [r for r in scene if r['common_renderable']]
        full = [r for r in common if r['reference_errors'] is not None]
        variants = {}
        for arm in ARMS:
            selected = [r for r in common if r['retained'][arm]]
            scored = [r for r in selected if r['reference_errors'] is not None]
            variants[arm] = dict(accepted=len(selected),reference_scored=len(scored),reference_unresolved=len(selected)-len(scored),
                deployed_score=aggregate(selected,'estimated_errors'),matched_estimate=aggregate(scored,'estimated_errors'),
                matched_reference=aggregate(scored,'reference_errors'),
                per_point_preference_agreement=sum(preference(r['estimated_errors']) == preference(r['reference_errors']) for r in scored))
        pair_results[name] = dict(planned=len(scene),common_renderable=len(common),
            full_common_reference=aggregate(full,'reference_errors'),variants=variants)
    return dict(scope='prospective four-scene replication of frozen baseline checks; reference visibility remains diagnostic, not certified camera-DNF truth',
        limitations='dependent points/frames; no new camera-method holdout; pretraining overlap unknown; placement includes initialization/calibration errors; compatible points are not certified static; no threshold changes',
        source_sha256={k:digest(p) for k,p in paths.items()}, geometry_sha256=geometry_hashes,
        script_sha256=digest(Path(__file__)),
        tracking_confusion={label:dict(Counter(r['cycle_predicted'] for r in points if r['tracking_reference'] == label))
            for label in ('close','far','borderline','unresolved')},
        motion_support={arm:{label:dict(Counter('retained' if r['retained'][arm] else 'unresolved' for r in points if r['reference'] == label))
            for label in ('static_background','moving_foreground','other_supported_foreground','unresolved')} for arm in ARMS},
        placement_results={arm:placement_summary(placements,arm) for arm in ARMS},
        paired_placement=pair_results,points=points,placements=placements)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    result = run(args.root)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('tracking_confusion','motion_support','placement_results')},indent=2))
