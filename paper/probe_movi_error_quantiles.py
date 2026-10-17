"""Fixed upper-quantile error regression; predictions are not calibrated bounds."""

import argparse
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import linprog

from paper.decode_movi_geometry_pilot import digest
from paper.probe_movi_error_regression import predict


def fit(rows, columns, quantile=.9, penalty=.1):
    if not 0 < quantile < 1 or penalty < 0:
        raise ValueError('valid quantile and nonnegative penalty required')
    labeled = [r for r in rows if r['eligible'] and r['error'] is not None]
    if not labeled:
        raise ValueError('training labels required')
    w = np.array([1/sum(t['sequence']==r['sequence'] for t in labeled) for r in labeled])
    x = np.array([r['features'] for r in labeled])[:,columns]
    y = np.log1p([r['error'] for r in labeled])
    mean = np.average(x,axis=0,weights=w)
    scale = np.sqrt(np.average((x-mean)**2,axis=0,weights=w))
    scale = np.where(scale>1e-12,scale,1.)
    z = np.column_stack([np.ones(len(x)),(x-mean)/scale])
    n,p = z.shape
    # beta, positive residual, negative residual, absolute non-intercept beta.
    c = np.r_[np.zeros(p),quantile*w,(1-quantile)*w,np.full(p-1,penalty)]
    equality = np.column_stack([z,np.eye(n),-np.eye(n),np.zeros((n,p-1))])
    upper = np.zeros((2*(p-1),len(c)))
    for j in range(1,p):
        upper[2*(j-1),j] = 1
        upper[2*(j-1)+1,j] = -1
        upper[2*(j-1):2*j,p+2*n+j-1] = -1
    solution = linprog(c,A_eq=equality,b_eq=y,A_ub=upper if p>1 else None,
        b_ub=np.zeros(len(upper)) if p>1 else None,
        bounds=[(None,None)]*p+[(0,None)]*(len(c)-p),method='highs')
    if not solution.success:
        raise ValueError('quantile optimizer failed: '+solution.message)
    beta = solution.x[:p]
    residual = y-z@beta
    objective = float(np.sum(w*np.maximum(quantile*residual,(quantile-1)*residual))+penalty*np.abs(beta[1:]).sum())
    if not np.isclose(objective,solution.fun,rtol=1e-7,atol=1e-8):
        raise ValueError('optimizer objective mismatch')
    return dict(columns=columns,mean=mean.tolist(),scale=scale.tolist(),coefficients=beta.tolist(),
                quantile=quantile,l1_penalty=penalty,training_points=n,
                training_scenes=sorted({r['sequence'] for r in labeled}),objective=objective)


def summarize(rows,quantile):
    scenes = []
    for name in sorted({r['sequence'] for r in rows}):
        eligible = [r for r in rows if r['sequence']==name and r['eligible']]
        scored = [r for r in eligible if r['error'] is not None]
        error = np.array([r['error'] for r in scored])
        radius = np.array([r['predicted_error_pixels'] for r in scored])
        residual = np.log1p(error)-np.log1p(radius)
        scenes.append(dict(sequence=name,eligible=len(eligible),labeled=len(scored),
            unlabeled_eligible=len(eligible)-len(scored),
            log_pinball_loss=float(np.maximum(quantile*residual,(quantile-1)*residual).mean()),
            labeled_misses=int(np.sum(error>radius+1e-9)),
            labeled_containment=float(np.mean(error<=radius+1e-9)),
            radius_median=float(np.median(radius)),radius_max=float(radius.max()),
            max_error_to_scale_ratio=float(np.max(error/np.maximum(1.,radius)))))
    return dict(scenes=scenes,macro_scene_pinball_loss=float(np.mean([r['log_pinball_loss'] for r in scenes])),
                labeled=sum(r['labeled'] for r in scenes),labeled_misses=sum(r['labeled_misses'] for r in scenes))


def run(path):
    source=json.loads(path.read_text())
    splits=source['results']['constant']['records']
    results={}
    for name,columns in (('constant',[]),('cycle_quantile',[0]),('image_evidence_quantile',list(range(7)))):
        model=fit(splits['training'],columns)
        records={}
        for split,rows in splits.items():
            records[split]=[dict(sequence=r['sequence'],frame=r['frame'],track=r['track'],eligible=r['eligible'],error=r['error'],
                predicted_error_pixels=predict(model,r) if r['eligible'] else None) for r in rows]
        results[name]=dict(model=model,metrics={s:summarize(r,model['quantile']) for s,r in records.items()},records=records)
    return dict(scope='retrospective upper-tail prediction baseline, not calibrated 90% coverage or camera-DNF validation',
        policy='same frozen features and original-four/replication-four split as prior error regression; fixed .9 log-error quantile, L1 penalty .1, per-scene total weight one, training-only standardization; all three models reported; no tuning or reference-label features',
        limitations='already-inspected development data; predicted quantiles are not finite-sample confidence bounds; diagnostic visibility labels; unlabeled eligible points remain; tail loss and radius size must be read together',
        source_sha256=digest(path),script_sha256=digest(Path(__file__)),scipy_version=scipy.__version__,results=results)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();result=run(args.source)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for name,r in result['results'].items():
        print(name,{s:(m['macro_scene_pinball_loss'],m['labeled_misses'],m['labeled']) for s,m in r['metrics'].items()})
