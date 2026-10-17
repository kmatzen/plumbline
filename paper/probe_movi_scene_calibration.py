"""Retrospective scene-level uncertainty calibration feasibility, not a guarantee."""

import argparse
import json
from decimal import Decimal, ROUND_CEILING
from pathlib import Path

import numpy as np

from paper.decode_movi_geometry_pilot import digest


def order_statistic(scores, coverage):
    if not scores or not 0 < coverage < 1 or not np.isfinite(scores).all() or min(scores) < 0:
        raise ValueError('finite nonnegative scene scores and coverage in (0,1) required')
    rank = int((Decimal(len(scores)+1)*Decimal(str(coverage))).to_integral_value(rounding=ROUND_CEILING))
    return dict(scenes=len(scores), nominal_coverage=coverage, rank=rank,
                multiplier=float(sorted(scores)[rank-1]) if rank <= len(scores) else None,
                status='finite' if rank <= len(scores) else 'unbounded')


def scene_scores(rows, mode):
    output = []
    for name in sorted({r['sequence'] for r in rows}):
        all_rows = [r for r in rows if r['sequence'] == name]
        eligible = [r for r in all_rows if r['eligible']]
        labeled = [r for r in eligible if r['error'] is not None]
        if not labeled:
            raise ValueError('scene has no calibration labels; cannot silently omit it')
        scale = lambda r: 1. if mode == 'constant' else max(1.,r['cycle'])
        output.append(dict(sequence=name,planned=len(all_rows),eligible=len(eligible),labeled=len(labeled),
                           unlabeled_eligible=len(eligible)-len(labeled),
                           score=max(r['error']/scale(r) for r in labeled)))
    return output


def evaluate(rows, multiplier, mode):
    result = []
    for name in sorted({r['sequence'] for r in rows}):
        eligible = [r for r in rows if r['sequence'] == name and r['eligible']]
        labeled = [r for r in eligible if r['error'] is not None]
        radii = [multiplier*(1. if mode == 'constant' else max(1.,r['cycle'])) for r in eligible] if multiplier is not None else None
        misses = ([r for r in labeled if r['error'] > multiplier*(1. if mode == 'constant' else max(1.,r['cycle']))]
                  if multiplier is not None else [])
        result.append(dict(sequence=name,eligible=len(eligible),labeled=len(labeled),
            unlabeled_eligible=len(eligible)-len(labeled), labeled_misses=len(misses),
            all_labeled_covered=not misses if labeled else None,
            radius_median=float(np.median(radii)) if radii else None,
            radius_max=max(radii) if radii else None,
            missed_slots=[[r['frame'],r['track']] for r in misses],
            radius_status='finite' if multiplier is not None else 'unbounded'))
    return result


def run(root):
    paths = dict(old_reverse=root/'movi-reverse-tracking-v1/manifest.json',
                 old_reference=root/'movi-temporal-verifier-v1/motion-track-error-v1.json',
                 new_evidence=root/'movi-replication-shard01-v1/evidence-v1/manifest.json',
                 new_reference=root/'movi-replication-shard01-v1/score-v1.json')
    data = {k:json.loads(p.read_text()) for k,p in paths.items()}
    if (data['old_reverse']['status'] != 'complete' or data['new_evidence']['status'] != 'complete'
            or data['old_reference']['source_sha256']['temporal'] != data['old_reverse']['forward_sha256']
            or data['new_reference']['source_sha256']['evidence'] != digest(paths['new_evidence'])):
        raise ValueError('source lineage mismatch')
    key = lambda r:(r['sequence'],r['frame'],r['track'])
    old_ref = {key(r):r for r in data['old_reference']['records']}
    new_ref = {key(r):r for r in data['new_reference']['points']}
    old, new = [], []
    for label,records,reference,target in (
        ('old',data['old_reverse']['records'],old_ref,old),
        ('new',data['new_evidence']['records'],new_ref,new)):
        if len(records) != 768 or {key(r) for r in records} != set(reference):
            raise ValueError('complete roster required')
        for r in records:
            ref = reference[key(r)]
            target.append(dict(sequence=r['sequence'],frame=r['frame'],track=r['track'],
                eligible=bool(r['forward_visible' if label == 'old' else 'predicted_visible'] and r['reverse_visible']),
                cycle=r['cycle_error_pixels'],error=ref.get('tracking_error_pixels')))
    if {r['sequence'] for r in old} & {r['sequence'] for r in new}:
        raise ValueError('scene overlap')
    methods = {}
    for mode in ('constant','cycle_scaled'):
        calibration = scene_scores(old,mode)
        methods[mode] = dict(calibration=calibration,levels=[])
        for coverage in (.8,.9,.95):
            fitted = order_statistic([r['score'] for r in calibration],coverage)
            methods[mode]['levels'].append(dict(**fitted,replication=evaluate(new,fitted['multiplier'],mode)))
    return dict(scope='retrospective calibration-feasibility diagnostic, not conformal coverage validation or novel methodology',
        policy='original four scenes provide one maximum normalized error each; replication scenes used only for reporting; both fixed constant and max(1px,cycle_error) scales; forward and reverse visibility required; all 768 slots per split accounted for; .8,.9,.95 nominal levels all shown',
        limitations='data already used in development; calibration and score choices not preregistered; scene exchangeability unverified; reference visibility is diagnostic and unlabeled accepted points remain; scene-level containment applies only to labeled eligible points, not all tracks, staticness, task success, or camera DNF',
        source_sha256={k:digest(p) for k,p in paths.items()},script_sha256=digest(Path(__file__)),methods=methods)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();result=run(args.root)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for mode,r in result['methods'].items():
        print(mode,[s['score'] for s in r['calibration']])
        for level in r['levels']:
            print(level['nominal_coverage'],level['status'],level['multiplier'],[(s['sequence'],s['labeled_misses'],s['radius_median'],s['radius_max']) for s in level['replication']])
