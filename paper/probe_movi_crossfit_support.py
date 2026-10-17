"""Fixed spatial holdout check of the epipolar support baseline."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from paper.decode_movi_geometry_pilot import digest
from paper.probe_movi_rigid_support import estimate,sampson_distance


def crossfit(first,second,visible):
    if np.shape(first)!=(64,2) or np.shape(second)!=(64,2) or np.shape(visible)!=(64,):
        raise ValueError('fixed 64-slot grid required')
    # Four interleaved spatial groups, defined without outcomes.
    folds=np.array([2*((i//8)%2)+(i%2) for i in range(64)])
    answers=['unresolved']*64;distances=[None]*64;fits=[]
    for fold in range(4):
        train=np.asarray(visible,bool)&(folds!=fold)
        fitted=estimate(first,second,train)
        fits.append(dict(fold=fold,training_slots=np.flatnonzero(train).tolist(),heldout_slots=np.flatnonzero(folds==fold).tolist(),fit=fitted))
        if fitted['status']!='estimated':continue
        values=sampson_distance(np.asarray(fitted['fundamental_matrix']),first,second)
        for i in np.flatnonzero((folds==fold)&visible):
            if np.isfinite(values[i]):
                distances[i]=float(values[i]);answers[i]='compatible' if values[i]<=1 else 'incompatible'
    return dict(answers=answers,distances=distances,fits=fits)


def run(root,temporal_path,baseline_path):
    temporal=json.loads(temporal_path.read_text());baseline=json.loads(baseline_path.read_text())
    assert baseline['source_sha256']['temporal']==digest(temporal_path)
    records=[];fits=[]
    for name in ('3835','5397','4385','4970'):
        path=root/f'marker-geometry-{name}-v1.json'
        assert digest(path)==baseline['source_sha256']['geometry'][name]
        first=np.array([r['query_xy'] for r in json.loads(path.read_text())['tracks']])
        for frame in (12,18,23):
            rows=sorted([r for r in temporal['records'] if r['sequence']==name and r['frame']==frame and r['method']=='dage'],key=lambda r:r['track'])
            assert [r['track'] for r in rows]==list(range(64))
            second=np.array([r['predicted_xy'] for r in rows]);visible=np.array([r['predicted_visible'] for r in rows])
            result=crossfit(first,second,visible);fits.append(dict(sequence=name,frame=frame,**result))
            for i in range(64):records.append(dict(sequence=name,frame=frame,track=i,predicted=result['answers'][i],sampson_pixels=result['distances'][i]))
    reference={(r['sequence'],r['frame'],r['track']):r for r in baseline['records']}
    for r in records:
        old=reference[(r['sequence'],r['frame'],r['track'])]
        r.update(reference=old['reference'],in_sample_prediction=old['predicted'])
    return dict(scope='development heldout-point support diagnostic; not independent scenes, staticness truth, novel method, or DNF',
        policy='four fixed interleaved grid groups; fit visible points from other three groups; unchanged RANSAC and 1px Sampson threshold; no own-fold observations fit the judged matrix',
        limitation='spatially dependent tracks, shared tracker, planar degeneracy, and coherent motion remain; point holdout is not scene-disjoint validation',
        script_sha256=digest(Path(__file__)),helper_sha256=digest(Path(__file__).with_name('probe_movi_rigid_support.py')),
        source_sha256=dict(temporal=digest(temporal_path),baseline=digest(baseline_path)),
        confusion={label:dict(Counter(r['predicted'] for r in records if r['reference']==label)) for label in ('static_background','moving_foreground','unresolved')},
        changed_decisions=sum(r['predicted']!=r['in_sample_prediction'] for r in records),records=records,fits=fits)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','temporal','baseline','output'):parser.add_argument(name,type=Path)
    args=parser.parse_args();result=run(args.root,args.temporal,args.baseline)
    with args.output.open('x') as stream:stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(result['confusion'],result['changed_decisions'])
