"""Correct-camera control with oracle initialization and surface correspondence.

Only the motion gate uses predicted image tracks. Oracle placement discrepancy
is a control, not a deployable reference-free score or camera-failure label.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from paper.audit_movi_joint_evidence import decide
from paper.decode_movi_geometry_pilot import digest


def unique_points(records):
    groups = {}
    for row in records:
        key = (row['sequence'], row['frame'], row['track'])
        groups.setdefault(key, []).append(row)
    result = {}
    for key, pair in groups.items():
        if len(pair) != 2 or {r['method'] for r in pair} != {'dage', 'da3'}:
            raise ValueError('exactly two original method records required')
        # Candidate geometry is intentionally ignored in this oracle control.
        fields = ('source_xy', 'reference_status', 'visibility_evidence',
                  'counterfactual_static_error_pixels')
        if any(pair[0].get(k) != pair[1].get(k) for k in fields):
            raise ValueError('method-independent reference mismatch')
        result[key] = pair[0]
    return result


def summarize(rows):
    return dict(planned=len(rows),
                references=dict(Counter(r['reference'] for r in rows)),
                answers={arm: dict(Counter(r[arm] for r in rows))
                         for arm in ('ungated', 'crossfit_gated')},
                reference_by_answer={arm: {ref: dict(Counter(r[arm] for r in rows
                    if r['reference'] == ref)) for ref in
                    ('static_background', 'supported_foreground', 'unresolved')}
                    for arm in ('ungated', 'crossfit_gated')})


def run(root, support_path):
    paths = dict(motion=root/'marker-motion-audit-v1.json',
                 support=support_path,
                 baseline=support_path.with_name('rigid-support-v1.json'))
    data = {k: json.loads(p.read_text()) for k, p in paths.items()}
    if (data['support']['source_sha256']['baseline'] != digest(paths['baseline'])
            or data['baseline']['source_sha256']['motion'] != digest(paths['motion'])):
        raise ValueError('reference provenance mismatch')
    motion = unique_points(data['motion']['records'])
    support = {(r['sequence'], r['frame'], r['track']): r
               for r in data['support']['records']}
    if len(support) != 768 or len(data['support']['records']) != 768 or set(support) != set(motion):
        raise ValueError('complete unique point roster required')
    rows = []
    for key, s in support.items():
        m = motion[key]
        error = None
        reference = 'unresolved'
        if s['reference'] == 'static_background':
            # With correct camera, calibration, world point, and correspondence,
            # the projection of a static point is the corresponding point.
            error, reference = 0., 'static_background'
        elif m.get('visibility_evidence') == 'supported':
            error = m['counterfactual_static_error_pixels']
            reference = 'supported_foreground'
        placement = ('unavailable' if error is None else 'close' if error <= 2
                     else 'far' if error >= 8 else 'uncertain')
        rows.append(dict(sequence=key[0], frame=key[1], track=key[2],
                         reference=reference, oracle_discrepancy_pixels=error,
                         motion_gate=s['predicted'],
                         ungated=decide(placement, s['predicted'], False),
                         crossfit_gated=decide(placement, s['predicted'], True)))
    return dict(scope='oracle correct-camera motion-confound control on development scenes; not a deployed evaluator or DNF accuracy',
                protocol='all 768 unique grid point/frame slots; correct camera, calibration, initial world point and surface correspondence; source-frozen points; retained continuous reference motion; unchanged 2/8px placement thresholds and image-only crossfit 1px epipolar gate',
                limitations='oracle correspondences remove tracking errors and reference visibility remains diagnostic; background zero is definitional, not a measured numerical test; far on a moving surface does not imply camera error; four development scenes, not generalization; oracle control changes initialization and correspondence together and is not a causal ablation of either alone',
                source_sha256={k: digest(p) for k, p in paths.items()},
                script_sha256=digest(Path(__file__)),
                results=dict(all=summarize(rows), **{scene: summarize([r for r in rows if r['sequence'] == scene])
                    for scene in sorted({r['sequence'] for r in rows})}), records=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'support', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    result = run(args.root, args.support)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result['results']['all'], indent=2))
