"""Frozen temporal point/visibility baseline on development correspondence cases."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from paper.decode_movi_geometry_pilot import digest


def run(staged, corpus, repo, checkpoint, prior, output):
    prior_record=json.loads(prior.read_text())
    assert digest(checkpoint)==prior_record['checkpoint_sha256']
    for name,sha in prior_record['source_files_sha256'].items():
        assert digest(repo/name)==sha
    staging=json.loads((staged/'manifest.json').read_text())
    manifest=json.loads((corpus/'manifest.json').read_text())
    assert manifest['source_manifest_sha256']==digest(staged/'manifest.json')
    rows=[r for r in manifest['records'] if r['partition'] in ('train','validation') and r['status']=='labeled']
    output.mkdir(parents=True,exist_ok=False)
    report=dict(status='starting',scope='development temporal correspondence baseline; not camera accuracy, staticness, or DNF',
        protocol='frozen CoTracker3 offline checkpoint and upstream defaults; source query supplied; group tracks by sequence and source frame; at most 61 RGB frames from source; no future beyond lag 60; full native video passed to upstream resizing; visibility false predicts occluded, otherwise <=2px close, >=8px far, middle uncertain',
        limitation='same previously inspected train/validation cases; supplied initial annotated queries; model visibility is a prediction, not truth; fixed short windows differ from full-video tracking; no calibration files opened',
        script_sha256=digest(Path(__file__)),corpus_sha256=digest(corpus/'manifest.json'),prior_sha256=digest(prior),
        checkpoint_sha256=digest(checkpoint),source_files_sha256=prior_record['source_files_sha256'],windows=[],records=[])
    def save():
        (output/'manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save()
    import torch
    sys.path.insert(0,str(repo))
    from cotracker.predictor import CoTrackerPredictor
    torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False
    model=CoTrackerPredictor(checkpoint=str(checkpoint),offline=True,v2=False,window_len=60).eval().cuda()
    for name in sorted({r['sequence'] for r in rows}):
        assert manifest['partitions'][name] in ('train','validation')
        entry=staging['sequences'][name];path=staged/entry['file']
        assert digest(path)==entry['file_sha256']
        with np.load(path,allow_pickle=False) as archive: video=archive['video']
        sequence_rows=[r for r in rows if r['sequence']==name]
        for first in sorted({r['query_frame'] for r in sequence_rows}):
            subset=[r for r in sequence_rows if r['query_frame']==first]
            tracks=sorted({r['track'] for r in subset})
            queries=[]
            for track in tracks:
                q=[r['source_xy'] for r in subset if r['track']==track]
                assert all(x==q[0] for x in q)
                queries.append([0,*q[0]])
            last=max(r['frame'] for r in subset)
            assert first<last<=first+60
            window=dict(sequence=name,first=first,last=last,status='running',tracks=tracks,source_sha256=digest(path))
            report['windows'].append(window);report['status']='evaluating';save()
            try:
                batch=torch.from_numpy(video[first:last+1].copy()).permute(0,3,1,2)[None].float().cuda()
                query=torch.tensor(queries,dtype=torch.float32,device='cuda')[None]
                started=time.monotonic()
                with torch.inference_mode(): xy,visible=model(batch,queries=query,backward_tracking=False)
                xy=xy[0].cpu().numpy();visible=visible[0].cpu().numpy()
                assert np.isfinite(xy).all() and visible.dtype==bool
                for row in subset:
                    t,i=row['frame']-first,tracks.index(row['track'])
                    distance=float(np.linalg.norm(xy[t,i]-row['candidate_xy']))
                    answer='occluded' if not visible[t,i] else 'close' if distance<=2 else 'far' if distance>=8 else 'uncertain'
                    report['records'].append(dict(id=row['id'],sequence=name,partition=row['partition'],
                        predicted_xy=xy[t,i].tolist(),predicted_visible=bool(visible[t,i]),candidate_distance_pixels=distance,
                        predicted=answer,target_for_scoring_only=row['label']))
                window.update(status='complete',seconds=time.monotonic()-started)
                del batch,query;torch.cuda.empty_cache()
            except Exception as error:
                window.update(status='failed',error=repr(error));save();raise
            save();print(name,first,window['status'],len(report['records']),flush=True)
    assert len(report['records'])==len(rows)==len({r['id'] for r in report['records']})
    report['status']='complete';save()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('staged','corpus','repo','checkpoint','prior','output'):parser.add_argument(name,type=Path)
    args=parser.parse_args();run(args.staged,args.corpus,args.repo,args.checkpoint,args.prior,args.output)
