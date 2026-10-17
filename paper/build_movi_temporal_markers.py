"""Alternative prefix-only temporal tracks for the fixed MOVi marker slots."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from paper.build_movi_marker_geometry import marker_rows
from paper.decode_movi_geometry_pilot import digest
from paper.run_movi_cameras import load_rgb


def run(decoded, cameras, repo, checkpoint, prior, output):
    import torch
    prior_record=json.loads(prior.read_text())
    assert digest(checkpoint)==prior_record['checkpoint_sha256']
    for name,sha in prior_record['source_files_sha256'].items():assert digest(repo/name)==sha
    output.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',scope='development proposal-initialization comparison; not camera quality or DNF',
        policy='same 64 fixed pixels and three query frames per method; CoTracker sees only frames 0..9; visibility-false prefix positions unavailable; no query frames or reference arrays enter fitting; all original IDs retained for pairing',
        script_sha256=digest(Path(__file__)),checkpoint_sha256=digest(checkpoint),prior_sha256=digest(prior),scenes=[])
    def save():(output/'manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save();sys.path.insert(0,str(repo))
    from cotracker.predictor import CoTrackerPredictor
    torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False
    model=CoTrackerPredictor(checkpoint=str(checkpoint),offline=True,v2=False,window_len=60).eval().cuda()
    try:
        for name in ('3835','5397','4385','4970'):
            source_path=cameras/f'marker-geometry-{name}-v1.json'
            source=json.loads(source_path.read_text())
            video,provenance=load_rgb(decoded,name)
            assert provenance==source['provenance']
            seeds=np.array([[0,*t['query_xy']] for t in source['tracks']],np.float32)
            assert seeds.shape==(64,3) and np.all(seeds[:,0]==0)
            batch=torch.from_numpy(video[:10].copy()).permute(0,3,1,2)[None].float().cuda()
            with torch.inference_mode(): xy,visible=model(batch,queries=torch.from_numpy(seeds)[None].cuda(),backward_tracking=False)
            xy=xy[0].cpu().numpy();visible=visible[0].cpu().numpy()
            assert xy.shape==(10,64,2) and visible.shape==(10,64) and np.isfinite(xy).all()
            tracks=[dict(track=i,query_frame=0,query_xy=seeds[i,1:].tolist(),records=[dict(frame=t,
                xy=xy[t,i].tolist() if visible[t,i] else None) for t in range(1,10)]) for i in range(64)]
            records=[];fits=[]
            for method in ('dage','da3'):
                path=cameras/'runs'/f'{name}-{method}'/'prediction.npz'
                assert digest(path)==source['cameras'][method]['prediction_sha256']
                with np.load(path,allow_pickle=False) as a: poses,k=a['extrinsics'],a['intrinsics_normalized']
                for track in tracks:
                    fit,proposals=marker_rows(track,poses,k);fits.append(dict(method=method,track=track['track'],**fit))
                    for proposal in proposals:
                        original=next(r for r in source['records'] if r['method']==method and r['track']==track['track'] and r['frame']==proposal['frame'])
                        records.append(dict(original,**proposal))
            result=dict(sequence=name,scope=report['scope'],policy=report['policy'],provenance=provenance,
                cameras=source['cameras'],source_geometry_sha256=digest(source_path),tracks=tracks,fits=fits,
                records=records,counts=dict(Counter(r['status'] for r in records)),planned_presentations=384)
            path=output/f'marker-geometry-{name}-v1.json'
            path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
            report['scenes'].append(dict(sequence=name,file=path.name,sha256=digest(path),counts=result['counts']))
            save();print(name,result['counts'],flush=True);del batch;torch.cuda.empty_cache()
        report['status']='complete'
    except Exception as error:
        report.update(status='failed',error=repr(error));raise
    finally:save()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('decoded','cameras','repo','checkpoint','prior','output'):parser.add_argument(name,type=Path)
    args=parser.parse_args();run(args.decoded,args.cameras,args.repo,args.checkpoint,args.prior,args.output)
