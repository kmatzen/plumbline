"""Score reverse consistency against retained development reference cues."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from paper.decode_movi_geometry_pilot import digest
from paper.probe_movi_reverse_tracking import cycle_answer


def key(row):
    return row['sequence'], row['frame'], row['track']


def summarize(rows):
    return dict(planned=len(rows), cycle_answers=dict(Counter(r['cycle_answer'] for r in rows)),
                correspondence_confusion={label: dict(Counter(r['cycle_answer'] for r in rows
                    if r['tracking_reference'] == label))
                    for label in ('close', 'far', 'borderline', 'unresolved')},
                motion_support={arm: {label: dict(Counter(r[arm] for r in rows if r['motion_reference'] == label))
                    for label in ('static_background', 'moving_foreground', 'unresolved')}
                    for arm in ('epipolar_only', 'epipolar_and_cycle')})


def run(reverse_path, verifier):
    paths = dict(reverse=reverse_path, tracking=verifier/'motion-track-error-v1.json',
                 support=verifier/'crossfit-support-v1.json', forward=verifier/'manifest.json')
    data = {k: json.loads(p.read_text()) for k, p in paths.items()}
    if (data['reverse']['status'] != 'complete'
            or data['reverse']['forward_sha256'] != digest(paths['forward'])
            or data['tracking']['source_sha256']['temporal'] != digest(paths['forward'])
            or data['tracking']['source_sha256']['support'] != digest(paths['support'])):
        raise ValueError('incomplete or mismatched experiment')
    indexed = {name: {key(r): r for r in data[name]['records']} for name in ('reverse', 'tracking', 'support')}
    if any(len(indexed[name]) != 768 or len(data[name]['records']) != 768 for name in indexed):
        raise ValueError('all unique planned slots required')
    if any(set(indexed[name]) != set(indexed['reverse']) for name in indexed):
        raise ValueError('roster mismatch')
    rows = []
    for identity, r in indexed['reverse'].items():
        tracking, support = indexed['tracking'][identity], indexed['support'][identity]
        error = r['cycle_error_pixels']
        if r['reverse_attempted']:
            recomputed = float(np.linalg.norm(np.asarray(r['reverse_xy'])-r['source_xy']))
            if not np.isclose(error, recomputed, rtol=0, atol=1e-10):
                raise ValueError('cycle distance mismatch')
        elif error is not None or r['reverse_xy'] is not None or r['reverse_visible']:
            raise ValueError('unattempted reverse has evidence')
        answer = cycle_answer(r['forward_visible'], r['reverse_visible'], error)
        if answer != r['predicted']:
            raise ValueError('cycle policy mismatch')
        label = 'unresolved'
        if tracking['status'] == 'compared':
            if r['forward_xy'] != tracking['predicted_xy']:
                raise ValueError('forward coordinates changed')
            distance = tracking['tracking_error_pixels']
            label = 'close' if distance <= 2 else 'far' if distance >= 8 else 'borderline'
        only = 'retained' if support['predicted'] == 'compatible' else 'unresolved'
        rows.append(dict(sequence=identity[0], frame=identity[1], track=identity[2],
                         forward_visible=r['forward_visible'], cycle_answer=answer,
                         cycle_error_pixels=error, tracking_reference=label,
                         motion_reference=support['reference'], epipolar_only=only,
                         epipolar_and_cycle=only if answer == 'consistent' else 'unresolved'))
    strata = dict(all=rows, forward_visible=[r for r in rows if r['forward_visible']])
    strata.update({scene: [r for r in rows if r['sequence'] == scene] for scene in sorted({r['sequence'] for r in rows})})
    return dict(scope='development reverse-consistency diagnostic, not staticness or camera DNF accuracy',
                protocol='fixed 1px cycle check; 2/8px reference correspondence bins; all 768 slots retained; combine existing epipolar compatibility and cycle consistency by conjunction, no refitting',
                limitations='same model correlated errors; four dependent development scenes; reference visibility cues are not certified; cycle consistency cannot establish a static surface; conditional counts are not generalization evidence',
                source_sha256={k: digest(p) for k, p in paths.items()}, script_sha256=digest(Path(__file__)),
                results={k: summarize(v) for k, v in strata.items()}, records=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reverse', 'verifier', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    result = run(args.reverse, args.verifier)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result['results']['all'], indent=2))
