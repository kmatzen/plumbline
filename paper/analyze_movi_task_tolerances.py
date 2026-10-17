"""Paired empirical task-success curves; no selected DNF tolerance."""

import argparse
import json
from pathlib import Path

import numpy as np

from paper.decode_movi_geometry_pilot import digest


def compare(first, second):
    a, b = np.asarray(first,float), np.asarray(second,float)
    if a.shape != b.shape or a.ndim != 1 or not len(a):
        raise ValueError('nonempty paired vectors required')
    if np.isnan(a).any() or np.isnan(b).any() or (a < 0).any() or (b < 0).any():
        raise ValueError('nonnegative errors or positive infinity required')
    thresholds = np.unique(np.r_[0., a[np.isfinite(a)], b[np.isfinite(b)]])
    ca = np.searchsorted(np.sort(a), thresholds, side='right')
    cb = np.searchsorted(np.sort(b), thresholds, side='right')
    delta = ca-cb
    return dict(paired=len(a), unavailable_dage=int(np.isinf(a).sum()), unavailable_da3=int(np.isinf(b).sum()),
        dage_weakly_dominates=bool((delta>=0).all()), da3_weakly_dominates=bool((delta<=0).all()),
        curves_cross=bool((delta>0).any() and (delta<0).any()),
        largest_dage_success_fraction_advantage=float(delta.max()/len(a)),
        largest_da3_success_fraction_advantage=float(-delta.min()/len(a)),
        thresholds_pixels=thresholds.tolist(), dage_success_counts=ca.tolist(), da3_success_counts=cb.tolist(),
        fixed_tolerance_success_counts={str(t):dict(dage=int((a<=t).sum()), da3=int((b<=t).sum())) for t in (1,2,4,8,16)})


def run(root):
    anchor_path=root/'anchor-task-v1/manifest.json'
    prefix_path=root/'prefix-tracking-ablation-v1.json'
    anchor=json.loads(anchor_path.read_text())
    prefix=json.loads(prefix_path.read_text())
    assert anchor['status']=='complete'
    results=[]
    for name in ('3835','5397','4385','4970'):
        for calibration in ('predicted_intrinsics','reference_intrinsics'):
            values, masks=[],[]
            for method in ('dage','da3'):
                attempt=next(a for a in anchor['attempts'] if a['sequence']==name and a['method']==method)
                item=attempt['calibration_arms'][calibration]
                path=root/'anchor-task-v1'/item['file']
                assert digest(path)==item['sha256']
                with np.load(path,allow_pickle=False) as a:
                    mask=a['reference_eligible']
                    masks.append(mask)
                    errors=a['error_pixels'].copy()
                    errors[~a['candidate_positive']]=np.inf
                    values.append(errors[mask])
            assert np.array_equal(*masks)
            results.append(dict(sequence=name, task='known_world_anchor', calibration=calibration, **compare(*values)))
        records={m:{(r['track'],r['frame']):r for r in prefix['records'] if r['sequence']==name and r['method']==m} for m in ('dage','da3')}
        assert records['dage'].keys()==records['da3'].keys()
        eligible=[]
        for key,a in records['dage'].items():
            b=records['da3'][key]
            assert (a['prefix_reference_eligible'],a['query_reference_eligible'])==(b['prefix_reference_eligible'],b['query_reference_eligible'])
            if a['prefix_reference_eligible'] and a['query_reference_eligible']:
                eligible.append(key)
        for field, task in (('rgb_error_pixels','rgb_initialized_surface_marker'),('oracle_error_pixels','oracle_initialized_surface_marker')):
            values=[[records[m][key][field] if records[m][key][field] is not None else np.inf for key in eligible] for m in ('dage','da3')]
            results.append(dict(sequence=name,task=task,calibration='predicted_intrinsics', **compare(*values)))
    return dict(scope='paired development task-success curves, not reference-free metric, human usability, or DNF',
        policy='same reference-eligible slots per paired task; unavailable candidate placements never counted as successes; exact empirical curve breakpoints and fixed 1/2/4/8/16 pixel summaries; off-image finite errors retained; no tolerance chosen from results',
        limitation='unavailable placements are not camera-failure labels; task rosters and initialization differ; reference visibility diagnostic; four scenes cannot establish generalization or statistical significance',
        script_sha256=digest(Path(__file__)), source_sha256=dict(anchor=digest(anchor_path),prefix=digest(prefix_path)),results=results)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path); parser.add_argument('output',type=Path)
    args=parser.parse_args(); result=run(args.root)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for r in result['results']:
        print(r['sequence'],r['task'],r['calibration'],'n=',r['paired'],'cross=',r['curves_cross'],r['fixed_tolerance_success_counts'])
