"""Camera-independent epipolar support baseline; inliers do not certify staticness."""

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from paper.decode_movi_geometry_pilot import digest


def sampson_distance(f, first, second):
    a=np.c_[first,np.ones(len(first))];b=np.c_[second,np.ones(len(second))]
    fa=a@f.T;fb=b@f
    numerator=np.abs(np.sum(b*fa,axis=1))
    denominator=np.sqrt(np.sum(fa[:,:2]**2,axis=1)+np.sum(fb[:,:2]**2,axis=1))
    return np.divide(numerator,denominator,out=np.full(len(a),np.inf),where=denominator>1e-12)


def estimate(first,second,visible):
    indices=np.flatnonzero(visible)
    answers=['unresolved']*len(first);distances=[None]*len(first)
    if len(indices)<8:return dict(status='insufficient_visible_tracks',answers=answers,distances=distances)
    cv2.setRNGSeed(0)
    f,mask=cv2.findFundamentalMat(np.asarray(first)[indices],np.asarray(second)[indices],cv2.FM_RANSAC,1.,.99,2000)
    if f is None or f.shape!=(3,3) or not np.isfinite(f).all():
        return dict(status='no_finite_fundamental_matrix',answers=answers,distances=distances)
    values=sampson_distance(f,np.asarray(first),np.asarray(second))
    for i in indices:
        if np.isfinite(values[i]):
            distances[i]=float(values[i]);answers[i]='compatible' if values[i]<=1 else 'incompatible'
    return dict(status='estimated',fundamental_matrix=f.tolist(),answers=answers,distances=distances,
                ransac_inliers=int(mask.sum()))


def run(root, temporal_path):
    temporal=json.loads(temporal_path.read_text());assert temporal['status']=='complete'
    bg_path=root/'background-marker-audit-v1.json';fg_path=root/'marker-motion-audit-v1.json'
    bg={r['id']:r for r in json.loads(bg_path.read_text())['records']}
    fg={r['id']:r for r in json.loads(fg_path.read_text())['records']}
    records=[];fits=[];sources={}
    for name in ('3835','5397','4385','4970'):
        path=root/f'marker-geometry-{name}-v1.json';geometry=json.loads(path.read_text())
        assert digest(path)==next(s for s in temporal['scenes'] if s['sequence']==name)['geometry_sha256']
        sources[name]=digest(path)
        first=np.array([t['query_xy'] for t in geometry['tracks']])
        for frame in (12,18,23):
            pair={m:sorted([r for r in temporal['records'] if r['sequence']==name and r['frame']==frame and r['method']==m],key=lambda r:r['track']) for m in ('dage','da3')}
            assert [r['track'] for r in pair['dage']]==list(range(64))
            assert [(r['predicted_xy'],r['predicted_visible']) for r in pair['dage']]==[(r['predicted_xy'],r['predicted_visible']) for r in pair['da3']]
            rows=pair['dage'];second=np.array([r['predicted_xy'] for r in rows]);visible=np.array([r['predicted_visible'] for r in rows])
            fitted=estimate(first,second,visible);fits.append(dict(sequence=name,frame=frame,**fitted))
            # Reference annotations enter only after the estimate is fixed.
            for i,row in enumerate(rows):
                b,g=bg[row['id']],fg[row['id']]
                reference='unresolved'
                if b.get('reference_depth_mask_not_rejected'):reference='static_background'
                elif g.get('visibility_evidence')=='supported' and g['counterfactual_static_error_pixels']>=8:reference='moving_foreground'
                records.append(dict(sequence=name,frame=frame,track=i,predicted=fitted['answers'][i],reference=reference,
                    sampson_pixels=fitted['distances'][i]))
    return dict(scope='development camera-independent epipolar-support baseline, not a staticness certificate, new method, or DNF',
        protocol='64 fixed source/temporal correspondences; predictor visibility gate; per-pair RANSAC 1px confidence .99 max2000 seed0; Sampson distance <=1px compatible; three query frames; no candidate cameras or reference labels in fitting; method duplicates removed',
        limitation='dominant motion can belong to moving objects; planar/rotational degeneracy and motion along epipolar lines can retain moving points; reference visibility remains diagnostic; no generalization claim',
        script_sha256=digest(Path(__file__)),opencv_version=cv2.__version__,source_sha256=dict(temporal=digest(temporal_path),background=digest(bg_path),motion=digest(fg_path),geometry=sources),
        confusion={label:dict(Counter(r['predicted'] for r in records if r['reference']==label)) for label in ('static_background','moving_foreground','unresolved')},fits=fits,records=records)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','temporal','output'):parser.add_argument(name,type=Path)
    args=parser.parse_args();result=run(args.root,args.temporal)
    with args.output.open('x') as stream:stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result['confusion'],indent=2))
