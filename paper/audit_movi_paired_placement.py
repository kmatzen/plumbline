"""Matched camera-placement comparisons with candidate-independent evidence gates."""

import argparse
import json
from pathlib import Path

import numpy as np

from paper.decode_movi_geometry_pilot import digest


def preference(errors):
    difference = errors['dage']-errors['da3']
    return 'tie' if abs(difference) <= 1e-9 else 'dage' if difference < 0 else 'da3'


def aggregate(rows, field):
    if not rows:
        return dict(count=0, mean=None, median=None, mean_preference=None, median_preference=None)
    mean = {m: float(np.mean([r[field][m] for r in rows])) for m in ('dage', 'da3')}
    median = {m: float(np.median([r[field][m] for r in rows])) for m in ('dage', 'da3')}
    return dict(count=len(rows), mean=mean, median=median,
                mean_preference=preference(mean), median_preference=preference(median))


def run(camera, verifier, reverse):
    paths = dict(corpus=camera/'judge-corpus-v1/manifest.json',
                 background=camera/'background-marker-audit-v1.json',
                 forward=verifier/'manifest.json', support=verifier/'crossfit-support-v1.json',
                 baseline=verifier/'rigid-support-v1.json', reverse=reverse/'manifest.json')
    data = {k: json.loads(p.read_text()) for k, p in paths.items()}
    if (data['forward']['status'] != 'complete' or data['reverse']['status'] != 'complete'
            or data['forward']['corpus_sha256'] != digest(paths['corpus'])
            or data['reverse']['forward_sha256'] != digest(paths['forward'])
            or data['support']['source_sha256']['temporal'] != digest(paths['forward'])
            or data['support']['source_sha256']['baseline'] != digest(paths['baseline'])
            or data['baseline']['source_sha256']['background'] != digest(paths['background'])):
        raise ValueError('source provenance mismatch')
    key = lambda r: (r['sequence'], r['frame'], r['track'])
    support = {key(r): r for r in data['support']['records']}
    cycle = {key(r): r for r in data['reverse']['records']}
    forward = {r['id']: r for r in data['forward']['records']}
    bg = {r['id']: r for r in data['background']['records']}
    groups = {}
    for r in data['corpus']['records']:
        groups.setdefault(key(r), []).append(r)
    if len(groups) != 768 or set(groups) != set(support) or set(groups) != set(cycle):
        raise ValueError('full paired roster required')
    rows = []
    for identity, pair in groups.items():
        if len(pair) != 2 or {r['method'] for r in pair} != {'dage', 'da3'}:
            raise ValueError('two method proposals required')
        f = [forward[r['id']] for r in pair]
        if (f[0]['predicted_xy'] != f[1]['predicted_xy']
                or f[0]['predicted_visible'] != f[1]['predicted_visible']):
            raise ValueError('candidate-dependent correspondence')
        row = dict(sequence=identity[0], frame=identity[1], track=identity[2],
                   ids={r['method']: r['id'] for r in pair},
                   reference_available=all(bg[r['id']].get('reference_depth_mask_not_rejected', False) for r in pair),
                   common_renderable=all(r['status'] == 'renderable' for r in pair),
                   forward_visible=f[0]['predicted_visible'],
                   motion_compatible=support[identity]['predicted'] == 'compatible',
                   cycle_consistent=cycle[identity]['predicted'] == 'consistent',
                   estimated_errors=None, reference_errors=None)
        if row['common_renderable']:
            row['estimated_errors'] = {r['method']: float(np.linalg.norm(
                np.asarray(r['candidate_xy'])-forward[r['id']]['predicted_xy'])) for r in pair}
            if row['reference_available']:
                row['reference_errors'] = {r['method']: bg[r['id']]['candidate_to_background_pixels'] for r in pair}
        rows.append(row)
    results = {}
    for scene in sorted({r['sequence'] for r in rows}):
        planned = [r for r in rows if r['sequence'] == scene]
        common = [r for r in planned if r['common_renderable']]
        full_reference = [r for r in common if r['reference_errors'] is not None]
        variants = {}
        for arm in ('tracking', 'tracking_motion', 'tracking_motion_cycle'):
            accepted = [r for r in common if r['forward_visible']
                        and (arm == 'tracking' or r['motion_compatible'])
                        and (arm != 'tracking_motion_cycle' or r['cycle_consistent'])]
            scored = [r for r in accepted if r['reference_errors'] is not None]
            variants[arm] = dict(accepted=len(accepted), reference_scored=len(scored),
                reference_unresolved=len(accepted)-len(scored),
                deployed_score=aggregate(accepted, 'estimated_errors'),
                matched_estimate=aggregate(scored, 'estimated_errors'),
                matched_reference=aggregate(scored, 'reference_errors'),
                per_point_preference_agreement=sum(preference(r['estimated_errors']) == preference(r['reference_errors']) for r in scored))
        results[scene] = dict(planned=len(planned), common_renderable=len(common),
                             full_common_reference=aggregate(full_reference, 'reference_errors'), variants=variants)
    return dict(scope='development matched RGB-initialized placement comparison, not isolated pose error, camera DNF, or heldout ranking validation',
                protocol='all 768 paired slots; both candidates must be renderable; same candidate-independent forward visibility, epipolar and cycle gates for both; continuous pixel error; both mean and median reported without choosing an aggregation winner',
                limitations='gates change evidence coverage; deployed scores include unresolved reference cases; matched reference scores use supported backgrounds only; point initialization and calibration remain part of task error; four development scenes cannot establish a ranking metric',
                source_sha256={k: digest(p) for k, p in paths.items()}, script_sha256=digest(Path(__file__)),
                results=results, records=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('camera', 'verifier', 'reverse', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    result = run(args.camera, args.verifier, args.reverse)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    for scene, value in result['results'].items():
        print(scene, 'full reference', value['full_common_reference']['mean_preference'], value['full_common_reference']['median_preference'])
        for arm, r in value['variants'].items():
            print(arm, r['accepted'], r['reference_scored'],
                  'deployed', r['deployed_score']['mean_preference'], r['deployed_score']['median_preference'],
                  'matched estimate', r['matched_estimate']['mean_preference'], r['matched_estimate']['median_preference'],
                  'matched reference', r['matched_reference']['mean_preference'], r['matched_reference']['median_preference'])
