"""Audit joint placement/motion evidence; never infer camera DNF from a point."""

import argparse
import json
from collections import Counter
from pathlib import Path

from paper.analyze_movi_judge import references, index
from paper.decode_movi_geometry_pilot import digest


def decide(placement, support, gated):
    if placement not in ('close', 'far'):
        return 'unresolved'
    if gated and support != 'compatible':
        return 'unresolved'
    return placement


def summarize(rows, arm):
    counts = Counter(r[arm] for r in rows)
    return dict(planned=len(rows), answers=dict(counts),
                by_reference={label: dict(Counter(r[arm] for r in rows
                    if r['motion_reference'] == label))
                    for label in ('static_background', 'moving_foreground', 'unresolved')},
                static_placement_confusion={label: dict(Counter(r[arm] for r in rows
                    if r['motion_reference'] == 'static_background'
                    and r['placement_reference'] == label))
                    for label in ('close', 'far', 'borderline', 'unresolved')})


def run(root, temporal_path, support_path):
    paths = dict(corpus=root/'judge-corpus-v1/manifest.json',
                 background=root/'background-marker-audit-v1.json',
                 motion=root/'marker-motion-audit-v1.json',
                 temporal=temporal_path, support=support_path,
                 support_baseline=support_path.with_name('rigid-support-v1.json'))
    data = {k: json.loads(p.read_text()) for k, p in paths.items()}
    if (data['temporal']['status'] != 'complete'
            or data['temporal']['corpus_sha256'] != digest(paths['corpus'])
            or data['support']['source_sha256']['temporal'] != digest(temporal_path)
            or data['support']['source_sha256']['baseline'] != digest(paths['support_baseline'])
            or any(data['support_baseline']['source_sha256'][k] != digest(paths[k])
                   for k in ('temporal', 'background', 'motion'))):
        raise ValueError('source mismatch')
    for scene, sha in data['corpus']['source_geometry_sha256'].items():
        if (data['background']['sources'][scene]['geometry_sha256'] != sha
                or data['motion']['source_sha256'][scene]['geometry'] != sha
                or data['support_baseline']['source_sha256']['geometry'][scene] != sha):
            raise ValueError('geometry mismatch')
    ref = index(references(data['corpus'], data['background'], data['motion']))
    temporal = index(data['temporal']['records'])
    support = {(r['sequence'], r['frame'], r['track']): r
               for r in data['support']['records']}
    if len(support) != 768 or set(temporal) != set(ref):
        raise ValueError('complete unique rosters required')
    rows = []
    for identity, r in ref.items():
        prediction = temporal[identity]
        if any(prediction[k] != r[k] for k in ('sequence', 'method', 'frame', 'track')):
            raise ValueError('identity mismatch')
        s = support[(r['sequence'], r['frame'], r['track'])]
        # Both decisions are computed before reference labels are joined.
        arms = {name: decide(prediction['predicted'], s['predicted'], gated)
                for name, gated in (('ungated', False), ('crossfit_gated', True))}
        rows.append(dict(id=identity, sequence=r['sequence'], method=r['method'],
                         frame=r['frame'], track=r['track'], **arms,
                         placement_reference=r['diagnostic_label'] or
                             ('borderline' if r['scoring_status'] == 'borderline' else 'unresolved'),
                         motion_reference=s['reference']))
    strata = dict(all=rows)
    strata.update({f'{scene}/{method}': [r for r in rows
                   if r['sequence'] == scene and r['method'] == method]
                   for scene in sorted({r['sequence'] for r in rows})
                   for method in sorted({r['method'] for r in rows})})
    return dict(scope='development joint-evidence audit, not camera failure labels or a novel evaluator',
                policy='unchanged 2/8px placement thresholds and heldout-point 1px epipolar gate; only close/far are determinate; occluded, unavailable and uncertain stay unresolved; all 1536 method-point-frame slots retained',
                limitations='moving points can pass the gate; static points can fail; placement discrepancy is joint camera/calibration/initialization/tracking error; references are diagnostic visibility cues; four dependent development scenes; no DNF or generalization claim',
                source_sha256={k: digest(p) for k, p in paths.items()},
                script_sha256=digest(Path(__file__)),
                results={arm: {k: summarize(v, arm) for k, v in strata.items()}
                         for arm in ('ungated', 'crossfit_gated')}, records=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'temporal', 'support', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    result = run(args.root, args.temporal, args.support)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v['all'] for k, v in result['results'].items()}, indent=2))
