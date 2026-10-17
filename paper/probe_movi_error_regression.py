"""Fixed small error-prediction baselines; no calibrated uncertainty claim."""

import argparse
import json
from pathlib import Path

import numpy as np

from paper.decode_movi_geometry_pilot import digest

FEATURES = ('log_cycle','log_sampson','missing_sampson','log_displacement','frame_fraction','source_edge_distance','target_edge_distance')


def features(source, target, frame, cycle, sampson):
    source, target = np.asarray(source,float), np.asarray(target,float)
    edge = lambda p: float(min(p[0],p[1],255-p[0],255-p[1]))/255
    values = [np.log1p(cycle),np.log1p(sampson) if sampson is not None else 0.,
              float(sampson is None),np.log1p(np.linalg.norm(target-source)),frame/23.,edge(source),edge(target)]
    if not np.isfinite(values).all():
        raise ValueError('finite image-only features required')
    return values


def fit(rows, columns):
    labeled = [r for r in rows if r['eligible'] and r['error'] is not None]
    names = sorted({r['sequence'] for r in labeled})
    if not names:
        raise ValueError('labeled training scenes required')
    weights = np.array([1/sum(t['sequence']==r['sequence'] for t in labeled) for r in labeled])
    x = np.array([r['features'] for r in labeled])[:,columns]
    y = np.log1p([r['error'] for r in labeled])
    mean = np.average(x,axis=0,weights=weights)
    scale = np.sqrt(np.average((x-mean)**2,axis=0,weights=weights))
    scale = np.where(scale>1e-12,scale,1.)
    z = np.column_stack([np.ones(len(x)),(x-mean)/scale])
    penalty = np.eye(z.shape[1]);penalty[0,0]=0
    beta = np.linalg.solve(z.T@(weights[:,None]*z)+penalty,z.T@(weights*y))
    return dict(columns=columns,mean=mean.tolist(),scale=scale.tolist(),coefficients=beta.tolist(),
                training_scenes=names,training_points=len(labeled),ridge_penalty=1.,scene_total_weight=1.)


def predict(model, row):
    values = np.asarray(row['features'])[model['columns']]
    z = np.r_[1.,(values-model['mean'])/model['scale']]
    log_error = max(0.,float(z@model['coefficients']))
    return float(np.expm1(log_error))


def metrics(rows):
    scenes = []
    for name in sorted({r['sequence'] for r in rows}):
        all_rows = [r for r in rows if r['sequence']==name]
        eligible = [r for r in all_rows if r['eligible']]
        scored = [r for r in eligible if r['error'] is not None]
        errors = np.array([r['error'] for r in scored])
        predictions = np.array([r['predicted_error_pixels'] for r in scored])
        scenes.append(dict(sequence=name,planned=len(all_rows),eligible=len(eligible),labeled=len(scored),
            unlabeled_eligible=len(eligible)-len(scored),
            log_error_mse=float(np.mean((np.log1p(predictions)-np.log1p(errors))**2)),
            mean_absolute_error_pixels=float(np.mean(np.abs(predictions-errors))),
            maximum_error_to_scale_ratio=float(np.max(errors/np.maximum(1.,predictions))),
            predicted_error_median=float(np.median(predictions)),predicted_error_max=float(np.max(predictions))))
    return dict(scenes=scenes,macro_scene_log_error_mse=float(np.mean([r['log_error_mse'] for r in scenes])))


def run(root):
    paths = dict(old_reverse=root/'movi-reverse-tracking-v1/manifest.json',
        old_support=root/'movi-temporal-verifier-v1/crossfit-support-v1.json',
        old_reference=root/'movi-temporal-verifier-v1/motion-track-error-v1.json',
        new_evidence=root/'movi-replication-shard01-v1/evidence-v1/manifest.json',
        new_reference=root/'movi-replication-shard01-v1/score-v1.json')
    data = {k:json.loads(p.read_text()) for k,p in paths.items()}
    if (data['old_reference']['source_sha256']['support'] != digest(paths['old_support'])
            or data['old_reference']['source_sha256']['temporal'] != data['old_reverse']['forward_sha256']
            or data['new_reference']['source_sha256']['evidence'] != digest(paths['new_evidence'])):
        raise ValueError('input lineage mismatch')
    key = lambda r:(r['sequence'],r['frame'],r['track'])
    support = {key(r):r for r in data['old_support']['records']}
    splits = {}
    for split, evidence, reference in (
        ('training',data['old_reverse']['records'],data['old_reference']['records']),
        ('replication',data['new_evidence']['records'],data['new_reference']['points'])):
        refs = {key(r):r for r in reference}
        if len(evidence)!=768 or {key(r) for r in evidence} != set(refs):
            raise ValueError('complete roster required')
        rows = []
        for r in evidence:
            old = split == 'training'
            eligible = r['forward_visible' if old else 'predicted_visible'] and r['reverse_visible']
            x = features(r['source_xy'],r['forward_xy' if old else 'predicted_xy'],r['frame'],r['cycle_error_pixels'],
                         support[key(r)]['sampson_pixels'] if old else r['motion_sampson_pixels']) if eligible else None
            rows.append(dict(sequence=r['sequence'],frame=r['frame'],track=r['track'],eligible=bool(eligible),
                             features=x,error=refs[key(r)].get('tracking_error_pixels')))
        splits[split] = rows
    if {r['sequence'] for r in splits['training']} & {r['sequence'] for r in splits['replication']}:
        raise ValueError('scene overlap')
    results = {}
    for name,columns in (('constant',[]),('cycle_ridge',[0]),('image_evidence_ridge',list(range(len(FEATURES))))):
        model = fit(splits['training'],columns)
        predictions = {split:[dict(r,predicted_error_pixels=predict(model,r) if r['eligible'] else None) for r in rows]
                       for split,rows in splits.items()}
        results[name] = dict(model=model,metrics={split:metrics(rows) for split,rows in predictions.items()},records=predictions)
    return dict(scope='retrospective learned image-error baseline development, not independent validation, calibrated radii, or camera DNF',
        policy='original four scenes train; replication scenes report only; eligible requires forward/reverse visibility; seven image-only features; log1p error target; per-scene total weight one; training-only standardization; fixed ridge penalty one, unpenalized intercept; predictions clipped at zero log error; no hyperparameter search; constant and cycle-only baselines retained',
        limitations='all scenes previously inspected; few independent videos; diagnostic reference visibility and unlabeled eligible observations remain; mean error prediction is not an upper confidence bound; learned uncertainty alone is not novel',
        features=list(FEATURES),source_sha256={k:digest(p) for k,p in paths.items()},script_sha256=digest(Path(__file__)),results=results)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();result=run(args.root)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for name,r in result['results'].items():
        print(name,{k:v['macro_scene_log_error_mse'] for k,v in r['metrics'].items()})
