"""Frozen temporal verifier on the identical MOVi marker roster."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from paper.decode_movi_geometry_pilot import digest
from paper.run_movi_cameras import load_rgb


def run(decoded, root, repo, checkpoint, prior, output):
    import torch
    old=json.loads(prior.read_text())
    assert digest(checkpoint)==old['checkpoint_sha256']
    for name,sha in old['source_files_sha256'].items():assert digest(repo/name)==sha
    corpus=json.loads((root/'judge-corpus-v1/manifest.json').read_text())
    output.mkdir(parents=True,exist_ok=False)
    result=dict(status='running',scope='frozen temporal correspondence baseline on development camera markers; not staticness or DNF',
        protocol='all four fixed scenes, all 64 original frame-zero pixel queries, all 24 RGB frames; same frozen CoTracker3 predictor and upstream defaults; no camera or reference data enters tracker; unchanged 2/8px decision thresholds; unavailable proposals retained',
        script_sha256=digest(Path(__file__)),corpus_sha256=digest(root/'judge-corpus-v1/manifest.json'),
        prior_sha256=digest(prior),checkpoint_sha256=digest(checkpoint),source_files_sha256=old['source_files_sha256'],scenes=[],records=[])
    def save():(output/'manifest.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    save();sys.path.insert(0,str(repo))
    from cotracker.predictor import CoTrackerPredictor
    torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False
    model=CoTrackerPredictor(checkpoint=str(checkpoint),offline=True,v2=False,window_len=60).eval().cuda()
    try:
        for name in ('3835','5397','4385','4970'):
            path=root/f'marker-geometry-{name}-v1.json'
            assert digest(path)==corpus['source_geometry_sha256'][name]
            geometry=json.loads(path.read_text())
            video,provenance=load_rgb(decoded,name)
            assert provenance==geometry['provenance']
            assert [t['track'] for t in geometry['tracks']]==list(range(64))
            query=np.array([[0,*t['query_xy']] for t in geometry['tracks']],np.float32)
            batch=torch.from_numpy(video.copy()).permute(0,3,1,2)[None].float().cuda()
            with torch.inference_mode(): xy,visibility=model(batch,queries=torch.from_numpy(query)[None].cuda(),backward_tracking=False)
            xy=xy[0].cpu().numpy();visibility=visibility[0].cpu().numpy()
            assert xy.shape==(24,64,2) and visibility.shape==(24,64) and np.isfinite(xy).all() and visibility.dtype==bool
            for row in geometry['records']:
                t,i=row['frame'],row['track']
                distance=None;answer='unavailable'
                if row['status']=='renderable':
                    distance=float(np.linalg.norm(xy[t,i]-row['candidate_xy']))
                    answer='occluded' if not visibility[t,i] else 'close' if distance<=2 else 'far' if distance>=8 else 'uncertain'
                result['records'].append(dict(id=row['id'],sequence=name,method=row['method'],frame=t,track=i,
                    predicted_xy=xy[t,i].tolist(),predicted_visible=bool(visibility[t,i]),candidate_distance_pixels=distance,predicted=answer))
            result['scenes'].append(dict(sequence=name,provenance=provenance,geometry_sha256=digest(path)))
            save();print(name,len(result['records']),flush=True)
            del batch;torch.cuda.empty_cache()
        assert len(result['records'])==len({r['id'] for r in result['records']})==1536
        result['status']='complete'
    except Exception as error:
        result.update(status='failed',error=repr(error));raise
    finally:save()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('decoded','root','repo','checkpoint','prior','output'):parser.add_argument(name,type=Path)
    args=parser.parse_args();run(args.decoded,args.root,args.repo,args.checkpoint,args.prior,args.output)
